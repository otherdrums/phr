"""
FusedQuantizedAdam — 8-bit AdamW optimizer powered by Triton.

Stores optimizer moment buffers (m, v) as int8 with per-block scales.
~75% memory reduction vs standard Adam (2 bytes/param vs 8 bytes/param).

When offloading is enabled, m/v/scales are stored in pinned CPU memory
and streamed to GPU in chunks (~100MB each) during step().
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


BLOCK_SIZE = 256


@triton.jit
def _fused_adam_8bit_kernel(
    p_ptr,
    g_ptr,
    m_ptr,
    v_ptr,
    m_scale_ptr,
    v_scale_ptr,
    lr,
    beta1,
    beta2,
    eps,
    bias_correction1,
    bias_correction2,
    weight_decay,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)

    m_i8 = tl.load(m_ptr + offs, mask=mask)
    v_i8 = tl.load(v_ptr + offs, mask=mask)
    m_scale = tl.load(m_scale_ptr + pid)
    v_scale = tl.load(v_scale_ptr + pid)

    m_fp = m_i8.to(tl.float32) * m_scale
    v_fp = v_i8.to(tl.float32) * v_scale

    if weight_decay > 0.0:
        p = p - lr * weight_decay * p

    m_new = beta1 * m_fp + (1.0 - beta1) * g
    v_new = beta2 * v_fp + (1.0 - beta2) * g * g

    m_absmax = tl.max(tl.abs(m_new))
    v_absmax = tl.max(tl.abs(v_new))
    new_m_scale = tl.maximum(m_absmax / 127.0, 1e-14)
    new_v_scale = tl.maximum(v_absmax / 127.0, 1e-14)

    m_rounded = tl.where(m_new >= 0.0, m_new / new_m_scale + 0.5, m_new / new_m_scale - 0.5).to(tl.int32)
    m_i8_new = tl.minimum(tl.maximum(m_rounded, -127), 127)
    v_rounded = (v_new / new_v_scale + 0.5).to(tl.int32)
    v_i8_new = tl.minimum(tl.maximum(v_rounded, 1), 127)

    m_hat = m_new / bias_correction1
    v_hat = v_new / bias_correction2
    p_new = p - lr * m_hat / (tl.sqrt(v_hat) + eps)

    tl.store(p_ptr + offs, p_new.to(tl.float32), mask=mask)
    tl.store(m_ptr + offs, m_i8_new, mask=mask)
    tl.store(v_ptr + offs, v_i8_new, mask=mask)
    tl.store(m_scale_ptr + pid, new_m_scale.to(tl.float32))
    tl.store(v_scale_ptr + pid, new_v_scale.to(tl.float32))


class FusedQuantizedAdam(torch.optim.Optimizer):
    """8-bit AdamW with per-block quantization.

    Args:
        params:      iterable of parameters
        lr:          learning rate
        betas:       (beta1, beta2) momentum coefficients
        eps:         epsilon for numerical stability
        weight_decay: L2 weight decay (AdamW-style)
        block_size:  elements per quantization block (default 256)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        block_size=256,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, block_size=block_size,
        )
        super().__init__(params, defaults)
        self._step_count = 0
        self._offload_mgr = None
        self._offload_enabled = False

    def enable_offload(self, manager):
        """Connect to an OffloadManager for chunked optimizer state streaming."""
        self._offload_mgr = manager
        self._offload_enabled = True
        manager._init_opt_offload(self)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        step = self._step_count

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            block = group["block_size"]

            bias1 = 1.0 - beta1 ** step
            bias2 = 1.0 - beta2 ** step

            # ── Offload path: GPU chunked Triton ──
            if self._offload_enabled and self._offload_mgr.num_chunks > 0:
                for chunk_idx in range(self._offload_mgr.num_chunks):
                    self._offload_mgr.prefetch_chunk(chunk_idx, self)

                    for pid in self._offload_mgr._opt_chunks[chunk_idx]["pids"]:
                        p = self._offload_mgr._opt_pid_to_param.get(pid)
                        if p is None or p.grad is None:
                            continue
                        state = self.state[p]
                        if state is None:
                            continue

                        grad = p.grad
                        p_data = p.data.contiguous()
                        g_data = grad.contiguous()
                        N = p_data.numel()
                        num_blocks = (N + block - 1) // block

                        _fused_adam_8bit_kernel[(num_blocks,)](
                            p_data, g_data,
                            state["m"], state["v"],
                            state["m_scale"], state["v_scale"],
                            lr, beta1, beta2, eps,
                            bias1, bias2, wd, N,
                            BLOCK=block,
                        )
                        if p.data.data_ptr() != p_data.data_ptr():
                            p.data.copy_(p_data)

                    self._offload_mgr.evict_chunk(chunk_idx, self)

                continue  # skip per-param loop

            # ── Non-offload path: per-param ──
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FusedQuantizedAdam does not support sparse gradients")

                state = self.state[p]
                if "m" not in state:
                    _init_state(state, p, block)

                p_data = p.data.contiguous()
                g_data = grad.contiguous()
                N = p_data.numel()
                num_blocks = (N + block - 1) // block

                _fused_adam_8bit_kernel[(num_blocks,)](
                    p_data, g_data,
                    state["m"], state["v"],
                    state["m_scale"], state["v_scale"],
                    lr, beta1, beta2, eps,
                    bias1, bias2, wd, N,
                    BLOCK=block,
                )
                if p.data.data_ptr() != p_data.data_ptr():
                    p.data.copy_(p_data)

        return loss

    def zero_grad(self, set_to_none: bool = False):
        super().zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _cpu_adam_step(self, group, bias1, bias2, block):
        """Quantized AdamW update on CPU using pinned state buffers.

        Vectorized per-param (no per-block Python loop).  Intermediate
        tensors are bounded at ~1 MB via batch-block processing.
        """
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]
        BATCH_BLOCKS = 1024

        for p in group["params"]:
            if p.grad is None:
                continue
            pid = id(p)
            cpu = self._offload_mgr._opt_state_cpu.get(pid)
            if cpu is None:
                continue

            N = p.numel()
            nb = (N + block - 1) // block
            pad = (block - (N % block)) % block

            # Transfer param + grad to CPU
            p_cpu = p.data.float().to("cpu")
            g_cpu = p.grad.float().to("cpu")

            # Pad to clean multiple of block_size for tensor ops
            if pad > 0:
                p_flat = torch.cat([p_cpu.flatten(), torch.zeros(pad)])
                g_flat = torch.cat([g_cpu.flatten(), torch.zeros(pad)])
            else:
                p_flat = p_cpu.flatten()
                g_flat = g_cpu.flatten()

            p_blk = p_flat.view(nb, block)
            g_blk = g_flat.view(nb, block)

            # Work with padded copies of m/v for clean reshape
            m_cpu = cpu["m"]
            v_cpu = cpu["v"]
            if pad > 0:
                m_pad = torch.cat([m_cpu.float(), torch.zeros(pad)]).to(torch.int8)
                v_pad = torch.cat([v_cpu.float(), torch.zeros(pad)]).to(torch.int8)
            else:
                m_pad = m_cpu
                v_pad = v_cpu

            m_blk = m_pad.float().view(nb, block)
            v_blk = v_pad.float().view(nb, block)
            m_scale = cpu["m_scale"]
            v_scale = cpu["v_scale"]

            for b_start in range(0, nb, BATCH_BLOCKS):
                b_end = min(b_start + BATCH_BLOCKS, nb)
                p_b = p_blk[b_start:b_end]
                g_b = g_blk[b_start:b_end]
                m_b = m_blk[b_start:b_end]
                v_b = v_blk[b_start:b_end]
                ms_b = m_scale[b_start:b_end]
                vs_b = v_scale[b_start:b_end]

                m_fp = m_b * ms_b.view(-1, 1)
                v_fp = v_b * vs_b.view(-1, 1)

                if wd > 0.0:
                    p_blk[b_start:b_end] = p_b - lr * wd * p_b

                m_new = beta1 * m_fp + (1.0 - beta1) * g_b
                v_new = beta2 * v_fp + (1.0 - beta2) * g_b * g_b

                m_absmax = m_new.abs().amax(dim=1)
                v_absmax = v_new.abs().amax(dim=1)
                new_ms = torch.clamp(m_absmax / 127.0, min=1e-14)
                new_vs = torch.clamp(v_absmax / 127.0, min=1e-10)

                m_rounded = torch.where(
                    m_new >= 0.0,
                    m_new / new_ms.view(-1, 1) + 0.5,
                    m_new / new_ms.view(-1, 1) - 0.5,
                )
                m_blk[b_start:b_end] = torch.clamp(m_rounded, -127, 127)
                v_rounded = (v_new / new_vs.view(-1, 1) + 0.5)
                v_blk[b_start:b_end] = torch.clamp(v_rounded, 1, 127)

                m_hat = m_new / bias1
                v_hat = v_new / bias2
                p_blk[b_start:b_end] = p_b - lr * m_hat / (torch.sqrt(v_hat) + eps)

                m_scale[b_start:b_end] = new_ms
                v_scale[b_start:b_end] = new_vs

            # Store back (strip padding)
            m_cpu.copy_(m_blk.flatten()[:N].to(torch.int8))
            v_cpu.copy_(v_blk.flatten()[:N].to(torch.int8))

            # Transfer updated param back to GPU
            p.data.copy_(p_blk.flatten()[:N].view_as(p.data))


def _init_state(state, p, block_size, offload=False):
    N = p.numel()
    num_blocks = (N + block_size - 1) // block_size
    if offload:
        state["m"] = torch.zeros(N, dtype=torch.int8, pin_memory=True)
        state["v"] = torch.zeros(N, dtype=torch.int8, pin_memory=True)
        state["m_scale"] = torch.ones(num_blocks, dtype=torch.float32, pin_memory=True)
        state["v_scale"] = torch.ones(num_blocks, dtype=torch.float32, pin_memory=True)
    else:
        state["m"] = torch.zeros(N, dtype=torch.int8, device=p.device)
        state["v"] = torch.zeros(N, dtype=torch.int8, device=p.device)
        state["m_scale"] = torch.ones(num_blocks, dtype=torch.float32, device=p.device)
        state["v_scale"] = torch.ones(num_blocks, dtype=torch.float32, device=p.device)
