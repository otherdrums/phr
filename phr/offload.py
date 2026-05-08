"""
OffloadManager — GPU↔CPU streaming for PHR training parameters.

W_p streaming:  frozen uint8 indices live in pinned CPU RAM.
               GPU buffer pool limits concurrent GPU memory to O(1) layers.
               Copies run synchronously on the default stream.

Optimizer state streaming:  m, v, and per-block scales are stored as
               individual pinned CPU tensors, grouped into chunks of
               ~100 MB each.  During optimizer.step(), each chunk's
               states are batched via torch.cat → single GPU copy →
               per-param views — then evicted the same way after the
               Triton kernels.  Peak GPU for states ~100 MB per chunk.
"""

import torch


CHUNK_BYTES = 96 * 1024 * 1024  # target ~100 MB per chunk GPU buffer

# JIT-compiled C++ CPU kernel for fused quantized AdamW (lazy init)
_cpu_adam_ext = None


def _get_cpu_adam_kernel():
    """JIT-compile the fused CPU AdamW kernel on first use."""
    global _cpu_adam_ext
    if _cpu_adam_ext is not None:
        return _cpu_adam_ext

    from torch.utils.cpp_extension import load_inline

    _cpu_adam_source = r"""
#include <torch/extension.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

void cpu_fused_adam_8bit_kernel(
    torch::Tensor m,
    torch::Tensor v,
    torch::Tensor m_scale,
    torch::Tensor v_scale,
    torch::Tensor p,
    torch::Tensor g,
    torch::Tensor p_start,
    torch::Tensor p_size,
    torch::Tensor blk_start,
    torch::Tensor blk_count,
    int64_t num_params,
    float lr, float beta1, float beta2, float eps,
    float bias1, float bias2, float wd,
    int64_t block_size)
{
    TORCH_CHECK(m.is_contiguous() && v.is_contiguous(), "flat m/v must be contiguous");
    TORCH_CHECK(p.is_contiguous() && g.is_contiguous(), "flat p/g must be contiguous");

    int8_t* __restrict__ m_ptr = m.data_ptr<int8_t>();
    int8_t* __restrict__ v_ptr = v.data_ptr<int8_t>();
    float* __restrict__ ms_ptr = m_scale.data_ptr<float>();
    float* __restrict__ vs_ptr = v_scale.data_ptr<float>();
    float* __restrict__ p_ptr = p.data_ptr<float>();
    const float* __restrict__ g_ptr = g.data_ptr<float>();
    const int64_t* __restrict__ ps_ptr = p_start.data_ptr<int64_t>();
    const int64_t* __restrict__ pn_ptr = p_size.data_ptr<int64_t>();
    const int64_t* __restrict__ bs_ptr = blk_start.data_ptr<int64_t>();
    const int64_t* __restrict__ bn_ptr = blk_count.data_ptr<int64_t>();

    for (int64_t pi = 0; pi < num_params; pi++) {
        int64_t ps = ps_ptr[pi];
        int64_t pn = pn_ptr[pi];
        int64_t bs = bs_ptr[pi];
        int64_t bn = bn_ptr[pi];

        for (int64_t b = 0; b < bn; b++) {
            float ms = ms_ptr[bs + b];
            float vs = vs_ptr[bs + b];
            int64_t lo_idx = b * block_size;
            int64_t hi_idx = std::min(lo_idx + block_size, pn);

            // Pass 1: dequantize, apply weight decay, compute moment updates
            float m_absmax = 0.0f;
            float v_absmax = 0.0f;
            float m_new_tmp[256];
            float v_new_tmp[256];
            for (int64_t j = lo_idx; j < hi_idx; j++) {
                int64_t i = ps + j;
                float mf = (float)m_ptr[i] * ms;
                float vf = (float)v_ptr[i] * vs;

                if (wd > 0.0f)
                    p_ptr[i] -= lr * wd * p_ptr[i];

                float m_new = beta1 * mf + (1.0f - beta1) * g_ptr[i];
                float v_new = beta2 * vf + (1.0f - beta2) * g_ptr[i] * g_ptr[i];
                int tmp_idx = j - lo_idx;
                m_new_tmp[tmp_idx] = m_new;
                v_new_tmp[tmp_idx] = v_new;
                m_absmax = std::max(m_absmax, std::fabs(m_new));
                v_absmax = std::max(v_absmax, std::fabs(v_new));
            }

            float new_ms = std::max(m_absmax / 127.0f, 1e-14f);
            float new_vs = std::max(v_absmax / 127.0f, 1e-10f);

            // Pass 2: requantize, apply AdamW update
            for (int64_t j = lo_idx; j < hi_idx; j++) {
                int64_t i = ps + j;
                int tmp_idx = j - lo_idx;
                float m_new = m_new_tmp[tmp_idx];
                float v_new = v_new_tmp[tmp_idx];

                float m_rounded = (m_new >= 0.0f)
                    ? (m_new / new_ms + 0.5f)
                    : (m_new / new_ms - 0.5f);
                m_ptr[i] = (int8_t)std::clamp((int)m_rounded, -127, 127);

                float v_rounded = v_new / new_vs + 0.5f;
                v_ptr[i] = (int8_t)std::clamp((int)v_rounded, 1, 127);

                float m_hat = m_new / bias1;
                float v_hat = v_new / bias2;
                p_ptr[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
            }

            ms_ptr[bs + b] = new_ms;
            vs_ptr[bs + b] = new_vs;
        }
    }
}
"""
    _cpu_adam_ext = load_inline(
        name="cpu_adam_ext",
        cpp_sources=_cpu_adam_source,
        functions=["cpu_fused_adam_8bit_kernel"],
        with_cuda=False,
        extra_cflags=["-O3"],
    )
    return _cpu_adam_ext


class OffloadManager:
    """Manages CPU↔GPU tensor transfers for PHR model components."""

    def __init__(self, prefetch_depth=1):
        if not torch.cuda.is_available():
            raise RuntimeError("OffloadManager requires CUDA")

        self._prefetch_depth = max(prefetch_depth, 0)
        self._stream = torch.cuda.Stream()  # for async eviction DMA

        # --- W_p tracking ---
        self._cpu_buffers = {}       # name → pinned CPU tensor (canonical copy)
        self._wp_params = {}         # name → nn.Parameter reference
        self._wp_devices = {}        # name → target CUDA device
        self._prefetch_buffers = {}  # name → gpu_tensor — in-flight (sync copy)
        self._on_gpu = set()         # names currently GPU-resident
        self._wp_gpu_pool = []       # list of free GPU tensors for reuse

        # --- Layer ordering for prefetch scheduling ---
        self._layer_seq = []
        self._layer_index = {}

        # --- Optimizer state streaming (chunked) ---
        self._opt_state_cpu = {}     # pid → {m, v, m_scale, v_scale} pinned CPU
        self._opt_chunks = []        # [{"pids": [...], "offsets": {pid: {key: (start,size)}}, "total": {key: int}}]
        self._opt_pid_to_param = {}  # pid → param tensor
        self._pending_eviction = None  # {pid: {key: cpu_tensor}} from previous chunk

    # ═══════════════════════════════════════════════════════════════
    # W_p registration & lifecycle
    # ═══════════════════════════════════════════════════════════════

    def register_wp(self, name, wp_param):
        self._wp_params[name] = wp_param
        if wp_param.device.type == "cuda":
            gpu_data = wp_param.data
            target_device = wp_param.device
            cpu_buf = torch.empty(gpu_data.size(), dtype=gpu_data.dtype, pin_memory=True)
            cpu_buf.copy_(gpu_data)
            self._cpu_buffers[name] = cpu_buf
            self._wp_devices[name] = target_device
            wp_param.data = cpu_buf.clone()
        else:
            self._cpu_buffers[name] = wp_param.data
            self._wp_devices[name] = None
        self._on_gpu.discard(name)

    def set_layer_sequence(self, names):
        self._layer_seq = list(names)
        self._layer_index = {n: i for i, n in enumerate(names)}

    def _acquire_wp_buffer(self, cpu_buf, target_device):
        for i, gpu in enumerate(self._wp_gpu_pool):
            if gpu.shape == cpu_buf.shape:
                self._wp_gpu_pool.pop(i)
                gpu.copy_(cpu_buf)
                return gpu
        return cpu_buf.to(target_device)

    def _release_wp_buffer(self, gpu_tensor):
        max_pool = max(self._prefetch_depth + 2, 3)
        if len(self._wp_gpu_pool) < max_pool:
            self._wp_gpu_pool.append(gpu_tensor)

    def prefetch_wp(self, name):
        if name in self._on_gpu or name in self._prefetch_buffers:
            return
        if name not in self._cpu_buffers or name not in self._wp_params:
            return
        target_device = self._wp_devices.get(name) or torch.device("cuda")
        cpu_buf = self._cpu_buffers[name]
        gpu_tensor = None
        for i, gpu in enumerate(self._wp_gpu_pool):
            if gpu.shape == cpu_buf.shape:
                gpu_tensor = self._wp_gpu_pool.pop(i)
                break
        if gpu_tensor is None:
            gpu_tensor = cpu_buf.to(target_device)
        else:
            gpu_tensor.copy_(cpu_buf)
        self._prefetch_buffers[name] = gpu_tensor

    def ensure_wp(self, name):
        if name in self._on_gpu:
            self._schedule_prefetch(name)
            return
        if name in self._prefetch_buffers:
            gpu_tensor = self._prefetch_buffers.pop(name)
            self._wp_params[name].data = gpu_tensor
            self._on_gpu.add(name)
            self._schedule_prefetch(name)
            return
        if name in self._cpu_buffers:
            cpu_buf = self._cpu_buffers[name]
            target_device = self._wp_devices.get(name) or torch.device("cuda")
            gpu_tensor = self._acquire_wp_buffer(cpu_buf, target_device)
            self._wp_params[name].data = gpu_tensor
            if self._wp_devices.get(name) is None:
                self._wp_devices[name] = target_device
            self._on_gpu.add(name)
            self._schedule_prefetch(name)

    def evict_wp(self, name):
        if name not in self._on_gpu:
            return
        wp = self._wp_params[name]
        self._release_wp_buffer(wp.data)
        wp.data = self._cpu_buffers[name].clone()
        self._on_gpu.discard(name)

    def make_evict_cb(self, name):
        def _cb():
            self.evict_wp(name)
        return _cb

    def _schedule_prefetch(self, current_name):
        idx = self._layer_index.get(current_name, -1)
        if idx < 0:
            return
        for d in range(1, self._prefetch_depth + 1):
            nxt = idx + d
            if nxt < len(self._layer_seq):
                self.prefetch_wp(self._layer_seq[nxt])

    # ═══════════════════════════════════════════════════════════════
    # Optimizer state streaming  (chunked)
    # ═══════════════════════════════════════════════════════════════

    @property
    def num_chunks(self):
        return len(self._opt_chunks)

    def _init_opt_offload(self, optimizer):
        """Pre-allocate per-param pinned CPU buffers, chunk groups, and
        flat CPU buffers for the fused C++ kernel.

        Called from enable_offload() before any optimizer step.
        """
        if self._opt_chunks:
            return

        # ── Pass 1: allocate per-param pinned CPU buffers ──
        all_pids = []
        for group in optimizer.param_groups:
            block = group.get("block_size", 256)
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                pid = id(p)
                all_pids.append(pid)
                self._opt_pid_to_param[pid] = p
                N = p.numel()
                nb = (N + block - 1) // block
                self._opt_state_cpu[pid] = {
                    "m":       torch.zeros(N,  dtype=torch.int8,    pin_memory=True),
                    "v":       torch.zeros(N,  dtype=torch.int8,    pin_memory=True),
                    "m_scale": torch.ones(nb,  dtype=torch.float32, pin_memory=True),
                    "v_scale": torch.ones(nb,  dtype=torch.float32, pin_memory=True),
                }

        # ── Pass 2: build flat pinned buffers + metadata for fused C++ kernel ──
        self._build_cpu_flat_buffers(all_pids)

        # ── Pass 3: group pids into GPU chunks ──
        current_chunk = {"pids": [], "offsets": {}, "total": {}}
        for pid in all_pids:
            cpu = self._opt_state_cpu[pid]
            param_bytes = sum(cpu[k].numel() * cpu[k].element_size() for k in ("m", "v"))
            if current_chunk["pids"] and any(
                current_chunk["total"].get(k, 0) + cpu[k].numel() * cpu[k].element_size() > CHUNK_BYTES
                for k in ("m", "v")
            ):
                self._opt_chunks.append(current_chunk)
                current_chunk = {"pids": [], "offsets": {}, "total": {}}

            current_chunk["pids"].append(pid)
            current_chunk["offsets"][pid] = {}
            for key in ("m", "v", "m_scale", "v_scale"):
                size = cpu[key].numel()
                start = current_chunk["total"].get(key, 0)
                current_chunk["offsets"][pid][key] = (start, size)
                current_chunk["total"][key] = start + size

        if current_chunk["pids"]:
            self._opt_chunks.append(current_chunk)

    def _build_cpu_flat_buffers(self, pids):
        """Build flat pinned CPU buffers and metadata for fused CPU kernel."""
        block = 256
        total_m = 0
        total_blocks = 0
        p_starts = []
        p_sizes = []
        blk_starts = []
        blk_counts = []

        for pid in pids:
            p = self._opt_pid_to_param[pid]
            N = p.numel()
            nb = (N + block - 1) // block
            p_starts.append(total_m)
            p_sizes.append(N)
            blk_starts.append(total_blocks)
            blk_counts.append(nb)
            total_m += N
            total_blocks += nb

        self._flat_m = torch.zeros(total_m, dtype=torch.int8, pin_memory=True)
        self._flat_v = torch.zeros(total_m, dtype=torch.int8, pin_memory=True)
        self._flat_ms = torch.ones(total_blocks, dtype=torch.float32, pin_memory=True)
        self._flat_vs = torch.ones(total_blocks, dtype=torch.float32, pin_memory=True)
        self._flat_p = torch.empty(total_m, dtype=torch.float32, pin_memory=True)
        self._flat_g = torch.empty(total_m, dtype=torch.float32, pin_memory=True)

        self._p_starts = torch.tensor(p_starts, dtype=torch.int64)
        self._p_sizes = torch.tensor(p_sizes, dtype=torch.int64)
        self._blk_starts = torch.tensor(blk_starts, dtype=torch.int64)
        self._blk_counts = torch.tensor(blk_counts, dtype=torch.int64)
        self._num_cpu_params = len(pids)
        self._pid_index = {pid: i for i, pid in enumerate(pids)}  # global index map

        # Replace per-param pinned tensors with views into flat buffers
        for i, pid in enumerate(pids):
            ps = p_starts[i]
            pn = p_sizes[i]
            bs = blk_starts[i]
            bn = blk_counts[i]
            self._opt_state_cpu[pid] = {
                "m":       self._flat_m[ps : ps + pn],
                "v":       self._flat_v[ps : ps + pn],
                "m_scale": self._flat_ms[bs : bs + bn],
                "v_scale": self._flat_vs[bs : bs + bn],
            }

    def cpu_fused_adam_step(self, optimizer, bias1, bias2, block):
        """Fused CPU-side quantized AdamW step.

        Batched GPU↔CPU transfers via offload stream (non-blocking),
        then single fused C++ kernel call on CPU.
        """
        # ── 1. Transfer params + grads to flat pinned CPU (async) ──
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                pid = id(p)
                idx = self._pid_index.get(pid)
                if idx is None:
                    continue
                ps = int(self._p_starts[idx])
                pn = int(self._p_sizes[idx])
                with torch.cuda.stream(self._stream):
                    self._flat_p[ps : ps + pn].copy_(p.data.float().view(-1))
                    self._flat_g[ps : ps + pn].copy_(p.grad.float().view(-1))
        self._stream.synchronize()

        # ── 2. Run fused C++ kernel on CPU ──
        kernel = _get_cpu_adam_kernel()
        g0 = optimizer.param_groups[0]
        kernel.cpu_fused_adam_8bit_kernel(
            self._flat_m, self._flat_v,
            self._flat_ms, self._flat_vs,
            self._flat_p, self._flat_g,
            self._p_starts, self._p_sizes,
            self._blk_starts, self._blk_counts,
            self._num_cpu_params,
            g0["lr"], g0["betas"][0], g0["betas"][1],
            g0["eps"], bias1, bias2, g0["weight_decay"],
            block,
        )

        # ── 3. Copy updated params back to GPU (async) ──
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                pid = id(p)
                idx = self._pid_index.get(pid)
                if idx is None:
                    continue
                ps = int(self._p_starts[idx])
                pn = int(self._p_sizes[idx])
                with torch.cuda.stream(self._stream):
                    p.data.copy_(self._flat_p[ps : ps + pn].view_as(p.data.float()).to(p.dtype))
        self._stream.synchronize()

    def prefetch_chunk(self, chunk_idx, optimizer):
        """Copy chunk's pinned CPU states to GPU via non-blocking transfers.

        Copies queue on the default stream.  Triton kernels run on the
        same stream so ordering is guaranteed.  The eviction for the
        previous chunk runs on the offload stream concurrently.
        """
        chunk = self._opt_chunks[chunk_idx]
        device = None
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.device.type == "cuda":
                    device = p.device
                    break
            if device is not None:
                break
        if device is None:
            return

        for pid in chunk["pids"]:
            cpu = self._opt_state_cpu[pid]
            p = self._opt_pid_to_param.get(pid)
            if p is None:
                continue
            state = optimizer.state[p]
            for key in ("m", "v", "m_scale", "v_scale"):
                state[key] = cpu[key].to(device, non_blocking=True)

    def evict_chunk(self, chunk_idx, optimizer):
        """Copy chunk's GPU states to pinned CPU via offload-stream DMA.

        Double-buffered: the PREVIOUS chunk's eviction DMA is cleaned
        up (completed by now since it overlapped with the current
        chunk's compute), then the CURRENT chunk's eviction is fired
        asynchronously.  For single-chunk models we sync immediately
        to avoid stale GPU views during the next forward pass.
        """
        # ── 1. Clean up previous chunk's eviction (DMA done by now) ──
        if self._pending_eviction is not None:
            self._stream.synchronize()
            for pid, cpu_dict in self._pending_eviction.items():
                p = self._opt_pid_to_param.get(pid)
                if p is None:
                    continue
                state = optimizer.state[p]
                for key in ("m", "v", "m_scale", "v_scale"):
                    state[key] = cpu_dict[key]
            self._pending_eviction = None

        # ── 2. Fire current chunk's eviction ──
        chunk = self._opt_chunks[chunk_idx]
        self._stream.wait_stream(torch.cuda.current_stream())

        if self.num_chunks == 1:
            # Single chunk: sync immediately (no next chunk to overlap with)
            for pid in chunk["pids"]:
                cpu = self._opt_state_cpu[pid]
                p = self._opt_pid_to_param.get(pid)
                if p is None:
                    continue
                state = optimizer.state[p]
                for key in ("m", "v", "m_scale", "v_scale"):
                    self._opt_state_cpu[pid][key].copy_(state[key])
                    state[key] = self._opt_state_cpu[pid][key]
        else:
            # Multi-chunk: fire async, overlap with next chunk's compute
            pending = {}
            for pid in chunk["pids"]:
                cpu = self._opt_state_cpu[pid]
                p = self._opt_pid_to_param.get(pid)
                if p is None:
                    continue
                state = optimizer.state[p]
                pinned = {}
                for key in ("m", "v", "m_scale", "v_scale"):
                    with torch.cuda.stream(self._stream):
                        cpu[key].copy_(state[key])
                    pinned[key] = cpu[key]
                pending[pid] = pinned
            self._pending_eviction = pending
