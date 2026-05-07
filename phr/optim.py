"""
FusedQuantizedAdam — 8-bit AdamW optimizer powered by Triton.

Stores optimizer moment buffers (m, v) as int8 with per-block scales.
~75% memory reduction vs standard Adam (2 bytes/param vs 8 bytes/param).

The Triton kernel handles dequantize → update → requantize → param update
in a single launch per parameter tensor.
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
    """
    8-bit AdamW with per-block quantization.

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

                grid = (num_blocks,)

                _fused_adam_8bit_kernel[grid](
                    p_data,
                    g_data,
                    state["m"],
                    state["v"],
                    state["m_scale"],
                    state["v_scale"],
                    lr,
                    beta1,
                    beta2,
                    eps,
                    bias1,
                    bias2,
                    wd,
                    N,
                    BLOCK=block,
                )

                if p.data.data_ptr() != p_data.data_ptr():
                    p.data.copy_(p_data)

        return loss


def _init_state(state, p, block_size):
    N = p.numel()
    num_blocks = (N + block_size - 1) // block_size

    state["m"] = torch.zeros(N, dtype=torch.int8, device=p.device)
    state["v"] = torch.zeros(N, dtype=torch.int8, device=p.device)
    state["m_scale"] = torch.ones(num_blocks, dtype=torch.float32, device=p.device)
    state["v_scale"] = torch.ones(num_blocks, dtype=torch.float32, device=p.device)
