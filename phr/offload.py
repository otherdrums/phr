"""
OffloadManager — GPU↔CPU streaming for PHR training parameters.

Offload levels:
  0  No offloading.  All tensors GPU-resident (default PHR).
  1  W_p streaming.  Frozen uint8 indices live in pinned CPU RAM.
     GPU buffer pool limits concurrent GPU memory to O(prefetch_depth).
  2  Level 1 + optimizer state storage offload.  Flat pinned CPU
     buffers enable batched GPU↔CPU transfers (O(1) copies/step).
  3  Level 1 + optimizer compute offload.  AdamW runs on CPU.
"""

import torch


class OffloadManager:
    """Manages CPU↔GPU tensor transfers for PHR model components."""

    def __init__(self, prefetch_depth=1):
        if not torch.cuda.is_available():
            raise RuntimeError("OffloadManager requires CUDA")

        self._stream = torch.cuda.Stream()
        self._prefetch_depth = max(prefetch_depth, 0)

        # --- W_p tracking ---
        self._cpu_buffers = {}       # name → pinned CPU tensor (canonical copy)
        self._wp_params = {}         # name → nn.Parameter reference
        self._wp_devices = {}        # name → target CUDA device
        self._prefetch_buffers = {}  # name → (gpu_tensor, event) — in-flight
        self._on_gpu = set()         # names currently GPU-resident

        # GPU buffer pool for W_p (reuse instead of re-allocate)
        self._wp_gpu_pool = []       # list of free GPU tensors

        # --- Layer ordering for prefetch scheduling ---
        self._layer_seq = []
        self._layer_index = {}

        # --- Optimizer state offloading (level 2+) ---
        # Flat pinned CPU buffers (one per state type)
        self._opt_flat_cpu = {}      # key → pinned CPU tensor
        # Per-param offsets into flat buffers
        self._opt_offsets = {}       # pid → {key: (offset, size)}

    # ═══════════════════════════════════════════════════════════════
    # W_p registration & lifecycle
    # ═══════════════════════════════════════════════════════════════

    def register_wp(self, name, wp_param):
        """Register a W_p Parameter.  Copies data to pinned CPU, moves param off GPU."""
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
        """Get a GPU tensor from the pool (or allocate)."""
        if self._wp_gpu_pool:
            gpu = self._wp_gpu_pool.pop()
            gpu.copy_(cpu_buf)  # fill with canonical data (sync on default stream)
            return gpu
        return cpu_buf.to(target_device)

    def _release_wp_buffer(self, gpu_tensor):
        """Return a GPU tensor to the pool for later reuse."""
        max_pool = max(self._prefetch_depth + 2, 3)
        if len(self._wp_gpu_pool) < max_pool:
            self._wp_gpu_pool.append(gpu_tensor)
        # (otherwise let Python GC free it)

    def prefetch_wp(self, name):
        """Fire async CPU→GPU copy for W_p[name].  Non-blocking."""
        if name in self._on_gpu or name in self._prefetch_buffers:
            return
        if name not in self._cpu_buffers or name not in self._wp_params:
            return

        target_device = self._wp_devices.get(name) or torch.device("cuda")
        cpu_buf = self._cpu_buffers[name]

        if self._wp_gpu_pool:
            gpu_tensor = self._wp_gpu_pool.pop()
        else:
            gpu_tensor = torch.empty_like(cpu_buf, device=target_device)

        event = torch.cuda.Event()
        with torch.cuda.stream(self._stream):
            gpu_tensor.copy_(cpu_buf, non_blocking=True)
            event.record(self._stream)

        self._prefetch_buffers[name] = (gpu_tensor, event)

    def ensure_wp(self, name):
        """Ensure W_p[name] is on GPU.  Schedules prefetch of nearby layers."""
        if name in self._on_gpu:
            self._schedule_prefetch(name)
            return

        if name in self._prefetch_buffers:
            gpu_tensor, event = self._prefetch_buffers.pop(name)
            torch.cuda.current_stream().wait_event(event)
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
        """Return W_p GPU buffer to pool, swap Parameter to CPU."""
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
            prv = idx - d
            if prv >= 0:
                self.prefetch_wp(self._layer_seq[prv])

    def sync(self):
        self._stream.synchronize()

    def prefetch_first_wp(self):
        """Prefetch W_p for the first layers in the sequence.

        Called during optimizer.step() so the CPU→GPU copies overlap
        with the Triton kernel and/or eviction transfers on the offload
        stream.  When the next forward pass calls ensure_wp for the
        first layer, the data is already on GPU (no fallback copy).
        """
        for d in range(self._prefetch_depth):
            if d < len(self._layer_seq):
                self.prefetch_wp(self._layer_seq[d])

    # ═══════════════════════════════════════════════════════════════
    # Optimizer state offloading  (level 2)
    # ═══════════════════════════════════════════════════════════════

    def _build_opt_flat_buffers(self, optimizer):
        """One-time init: copy per-param GPU states into flat pinned CPU buffers.

        Replaces optimizer.state entries with CPU views into the flat buffers
        (freeing GPU memory in the process).
        """
        offsets_m = []
        offsets_v = []
        offsets_ms = []
        offsets_vs = []
        sizes_m = []
        sizes_v = []
        sizes_ms = []
        sizes_vs = []
        pids = []

        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p)
                if state is None or "m" not in state:
                    continue
                pid = id(p)
                pids.append(pid)
                sizes_m.append(state["m"].numel())
                sizes_v.append(state["v"].numel())
                sizes_ms.append(state["m_scale"].numel())
                sizes_vs.append(state["v_scale"].numel())

        if not sizes_m:
            return

        total = lambda offsets, sizes: offsets[-1] + sizes[-1] if offsets else 0

        for sizes, offsets in [(sizes_m, offsets_m), (sizes_v, offsets_v),
                                (sizes_ms, offsets_ms), (sizes_vs, offsets_vs)]:
            off = 0
            for s in sizes:
                offsets.append(off)
                off += s

        flat_m  = torch.empty(total(offsets_m, sizes_m),  dtype=torch.int8,    pin_memory=True)
        flat_v  = torch.empty(total(offsets_v, sizes_v),  dtype=torch.int8,    pin_memory=True)
        flat_ms = torch.empty(total(offsets_ms, sizes_ms), dtype=torch.float32, pin_memory=True)
        flat_vs = torch.empty(total(offsets_vs, sizes_vs), dtype=torch.float32, pin_memory=True)

        for i, pid in enumerate(pids):
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if id(p) != pid:
                        continue
                    state = optimizer.state[p]
                    om, sm = offsets_m[i], sizes_m[i]
                    ov, sv = offsets_v[i], sizes_v[i]
                    oms, sms = offsets_ms[i], sizes_ms[i]
                    ovs, svs = offsets_vs[i], sizes_vs[i]

                    flat_m[om : om + sm].copy_(state["m"])
                    flat_v[ov : ov + sv].copy_(state["v"])
                    flat_ms[oms : oms + sms].copy_(state["m_scale"])
                    flat_vs[ovs : ovs + svs].copy_(state["v_scale"])

                    # Replace GPU state tensors with CPU views (frees GPU memory)
                    state["m"]       = flat_m[om : om + sm]
                    state["v"]       = flat_v[ov : ov + sv]
                    state["m_scale"] = flat_ms[oms : oms + sms]
                    state["v_scale"] = flat_vs[ovs : ovs + svs]

                    self._opt_offsets[pid] = {
                        "m": (om, sm), "v": (ov, sv),
                        "m_scale": (oms, sms), "v_scale": (ovs, svs),
                    }
                    break

        self._opt_flat_cpu["m"] = flat_m
        self._opt_flat_cpu["v"] = flat_v
        self._opt_flat_cpu["m_scale"] = flat_ms
        self._opt_flat_cpu["v_scale"] = flat_vs

    def prefetch_optim_states(self, optimizer):
        """Batched CPU→GPU transfer for all optimizer states.

        First call builds flat pinned CPU buffers.  Subsequent calls:
        1. Sync + free the PREVIOUS step's async GPU→CPU copy buffers.
        2. Assign per-param CPU views (data now fresh in pinned buffers).
        3. Allocate fresh flat GPU buffers and assign per-param GPU views.
        """
        if not self._opt_flat_cpu:
            self._build_opt_flat_buffers(optimizer)
        if not self._opt_flat_cpu:
            return

        # ── 1. Free previous-step GPU buffers (copy completed during fwd) ──
        if getattr(self, '_evict_pending', False):
            self._evict_event.synchronize()
            self._evict_pending = False
            self._old_flat_gpu = None

        # ── 2. Assign CPU views (pinned buffer has fresh data from copy) ──
        for key in ("m", "v", "m_scale", "v_scale"):
            flat_cpu = self._opt_flat_cpu[key]
            for group in optimizer.param_groups:
                for p in group["params"]:
                    state = optimizer.state.get(p)
                    if state is None or "m" not in state:
                        continue
                    off = self._opt_offsets.get(id(p))
                    if off is None:
                        continue
                    start, size = off[key]
                    state[key] = flat_cpu[start : start + size]

        # ── 3. Allocate fresh flat GPU buffers ──
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

        self._flat_gpu = {}
        for key in ("m", "v", "m_scale", "v_scale"):
            flat_cpu = self._opt_flat_cpu[key]
            flat_gpu = flat_cpu.to(device)
            self._flat_gpu[key] = flat_gpu
            for group in optimizer.param_groups:
                for p in group["params"]:
                    state = optimizer.state.get(p)
                    if state is None or "m" not in state:
                        continue
                    off = self._opt_offsets.get(id(p))
                    if off is None:
                        continue
                    start, size = off[key]
                    state[key] = flat_gpu[start : start + size]

    def evict_optim_states(self, optimizer):
        """Batched GPU→CPU async copy for all optimizer states.

        First step: builds flat buffers (sync).
        Subsequent steps: fires GPU→CPU copy on the offload stream
        (non-blocking).  The copy overlaps with the next forward pass.
        """
        if not self._opt_flat_cpu:
            self._build_opt_flat_buffers(optimizer)
            return

        flat_gpu = getattr(self, '_flat_gpu', None) or {}
        if not flat_gpu:
            return

        for key in ("m", "v", "m_scale", "v_scale"):
            if key in flat_gpu:
                with torch.cuda.stream(self._stream):
                    self._opt_flat_cpu[key].copy_(flat_gpu[key], non_blocking=True)

        self._evict_event = torch.cuda.Event()
        self._evict_event.record(self._stream)
        self._evict_pending = True

        self._old_flat_gpu = flat_gpu
        self._flat_gpu = {}

    def finalize_opt_offload(self, optimizer=None):
        """Sync pending eviction and free all GPU optimizer state buffers.

        Args:
            optimizer: If provided, per-param state views are replaced
                       with CPU views (freeing GPU memory).  If None,
                       only the event is synced and flat buffers freed.
        """
        if getattr(self, '_evict_pending', False):
            self._evict_event.synchronize()
            self._evict_pending = False
            self._old_flat_gpu = None
            self._flat_gpu = {}

            if optimizer is not None:
                for key in ("m", "v", "m_scale", "v_scale"):
                    flat_cpu = self._opt_flat_cpu.get(key)
                    if flat_cpu is None:
                        continue
                    for group in optimizer.param_groups:
                        for p in group["params"]:
                            state = optimizer.state.get(p)
                            if state is None or "m" not in state:
                                continue
                            off = self._opt_offsets.get(id(p))
                            if off is None:
                                continue
                            start, size = off[key]
                            if state[key].is_cuda or state[key].data_ptr() != \
                                    flat_cpu[start : start + size].data_ptr():
                                state[key] = flat_cpu[start : start + size]

            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    # Optimizer compute offload  (level 3)
    # ═══════════════════════════════════════════════════════════════

    def compute_optim_step_cpu(self, optimizer, bias_correction1, bias_correction2):
        """Run the quantized AdamW update entirely on CPU."""
        for group in optimizer.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            block_size = group["block_size"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = optimizer.state.get(p)
                if state is None or "m" not in state:
                    continue

                pid = id(p)
                off = self._opt_offsets.get(pid)
                if off is None:
                    continue

                flat_m = self._opt_flat_cpu["m"]
                flat_v = self._opt_flat_cpu["v"]
                flat_ms = self._opt_flat_cpu["m_scale"]
                flat_vs = self._opt_flat_cpu["v_scale"]

                # Ensure latest param + grad on CPU
                p_cpu = torch.empty(p.data.size(), dtype=p.data.dtype, pin_memory=True)
                g_cpu = torch.empty(p.grad.size(), dtype=p.grad.dtype, pin_memory=True)
                p_cpu.copy_(p.data)
                g_cpu.copy_(p.grad)

                m_start, m_size = off["m"]
                v_start, v_size = off["v"]
                ms_start, ms_size = off["m_scale"]
                vs_start, vs_size = off["v_scale"]

                m_i8 = flat_m[m_start : m_start + m_size]
                v_i8 = flat_v[v_start : v_start + v_size]
                m_scale = flat_ms[ms_start : ms_start + ms_size]
                v_scale = flat_vs[vs_start : vs_start + vs_size]

                N = p_cpu.numel()
                num_blocks = (N + block_size - 1) // block_size

                p_flat = p_cpu.float().flatten()
                g_flat = g_cpu.float().flatten()

                for blk in range(num_blocks):
                    lo = blk * block_size
                    hi = min(lo + block_size, N)

                    gg = g_flat[lo:hi]

                    m_blk = m_i8[lo:hi].float() * m_scale[blk]
                    v_blk = v_i8[lo:hi].float() * v_scale[blk]

                    if wd > 0.0:
                        p_flat[lo:hi] -= lr * wd * p_flat[lo:hi]

                    m_new = beta1 * m_blk + (1.0 - beta1) * gg
                    v_new = beta2 * v_blk + (1.0 - beta2) * gg * gg

                    m_absmax = m_new.abs().max()
                    v_absmax = v_new.abs().max()
                    new_m_scale = max(m_absmax.item() / 127.0, 1e-14)
                    new_v_scale = max(v_absmax.item() / 127.0, 1e-10)

                    m_rounded = torch.where(
                        m_new >= 0.0,
                        m_new / new_m_scale + 0.5,
                        m_new / new_m_scale - 0.5,
                    )
                    m_i8[lo:hi] = torch.clamp(m_rounded.to(torch.int8), -127, 127)
                    v_rounded = (v_new / new_v_scale + 0.5)
                    v_i8[lo:hi] = torch.clamp(v_rounded.to(torch.int8), 1, 127)

                    m_hat = m_new / bias_correction1
                    v_hat = v_new / bias_correction2
                    p_flat[lo:hi] -= lr * m_hat / (torch.sqrt(v_hat) + eps)

                    m_scale[blk] = new_m_scale
                    v_scale[blk] = new_v_scale

                p.data.copy_(p_flat.reshape(p.data.shape))
