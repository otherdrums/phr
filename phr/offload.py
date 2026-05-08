"""
OffloadManager — GPU↔CPU streaming for PHR training parameters.

Offload levels:
  0  No offloading.  All tensors GPU-resident (default PHR).
  1  W_p streaming.  Frozen uint8 indices live in pinned CPU RAM.
     Double-buffered: at most prefetch_depth+1 layers' W_p on GPU
     simultaneously.  Evicted after forward, re-prefetched in backward.
  2  Level 1 + optimizer state storage offload.  int8 m/v on CPU,
     prefetched to GPU before step(), evicted after.
  3  Level 1 + optimizer compute offload.  int8 m/v on CPU.
     AdamW update runs on CPU (GPU free during step).

Usage:
    mgr = OffloadManager(prefetch_depth=1)
    mgr.register_wp("layer.0", phr_layer.W_p)  # per layer
    mgr.set_layer_sequence(["layer.0", "layer.1", ...])
    phr_layer.attach_offload(mgr, "layer.0")

    optimizer.enable_offload(mgr, level=2)
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

        # Deferred GPU tensor freelist: (tensor, event) — tensor is freed
        # once event completes (no CPU-blocking sync).
        self._pending_free = []

        # --- Layer ordering for prefetch scheduling ---
        self._layer_seq = []
        self._layer_index = {}

        # --- Optimizer state offloading (level 2+) ---
        self._optim_state_cpu = {}   # pid → {m, v, m_scale, v_scale} pinned CPU

    # ═══════════════════════════════════════════════════════════════
    # W_p registration & lifecycle
    # ═══════════════════════════════════════════════════════════════

    def register_wp(self, name, wp_param):
        """Register a W_p Parameter.  Copies data to pinned CPU, moves param off GPU.

        After registration wp_param.data is a CPU tensor and the GPU
        allocation is freed.  Works whether or not the parameter is
        currently on CUDA (stores target device for later restores).
        """
        self._wp_params[name] = wp_param

        if wp_param.device.type == "cuda":
            gpu_data = wp_param.data
            target_device = wp_param.device
            cpu_buf = torch.empty(gpu_data.size(), dtype=gpu_data.dtype, pin_memory=True)
            cpu_buf.copy_(gpu_data)  # synchronous GPU→CPU on default stream
            self._cpu_buffers[name] = cpu_buf
            self._wp_devices[name] = target_device
            wp_param.data = cpu_buf.clone()
        else:
            self._cpu_buffers[name] = wp_param.data
            self._wp_devices[name] = None  # resolved on first ensure_wp

        self._on_gpu.discard(name)

    def set_layer_sequence(self, names):
        """Set the ordered list of layer names for prefetch scheduling.

        Names must match those passed to register_wp / attach_offload.
        """
        self._layer_seq = list(names)
        self._layer_index = {n: i for i, n in enumerate(names)}

    def prefetch_wp(self, name):
        """Fire async CPU→GPU copy for W_p[name].  Non-blocking."""
        if name in self._on_gpu or name in self._prefetch_buffers:
            return
        if name not in self._cpu_buffers or name not in self._wp_params:
            return

        wp = self._wp_params[name]
        target_device = self._wp_devices.get(name) or torch.device("cuda")

        cpu_buf = self._cpu_buffers[name]
        gpu_tensor = torch.empty_like(cpu_buf, device=target_device)
        event = torch.cuda.Event()

        with torch.cuda.stream(self._stream):
            gpu_tensor.copy_(cpu_buf, non_blocking=True)
            event.record(self._stream)

        self._prefetch_buffers[name] = (gpu_tensor, event)

    def ensure_wp(self, name):
        """Ensure W_p[name] is on GPU, blocking if necessary.

        Also drains the deferred-free queue (pops GPU tensors whose
        recorded events have completed) and schedules prefetch of
        nearby layers.
        """
        self._drain_pending_free()

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

        # Fallback — synchronous copy on default stream
        if name in self._cpu_buffers:
            cpu_buf = self._cpu_buffers[name]
            target_device = self._wp_devices.get(name) or torch.device("cuda")
            gpu_tensor = cpu_buf.to(target_device)
            wp = self._wp_params[name]
            wp.data = gpu_tensor
            if self._wp_devices.get(name) is None:
                self._wp_devices[name] = target_device
            self._on_gpu.add(name)
            self._schedule_prefetch(name)

    def evict_wp(self, name):
        """Reclaim GPU memory for W_p[name].

        Records a CUDA event on the default stream and defers freeing
        the GPU tensor until the event completes.  The pin is swapped
        to a CPU clone immediately so the Parameter no longer holds
        GPU memory.  Pending frees are drained on the next ensure_wp
        call (event.query(), non-blocking).
        """
        if name not in self._on_gpu:
            return
        wp = self._wp_params[name]
        gpu_tensor = wp.data
        event = torch.cuda.Event()
        event.record()
        self._pending_free.append((gpu_tensor, event))
        wp.data = self._cpu_buffers[name].clone()
        self._on_gpu.discard(name)

    def _drain_pending_free(self):
        """Pop GPU tensors whose recorded events have completed.

        If the queue exceeds 2× the layer count, force-sync the oldest
        entry to prevent unbounded GPU memory growth.
        """
        max_pending = max(len(self._layer_seq) * 2, 48)
        while len(self._pending_free) > max_pending:
            gpu_tensor, event = self._pending_free.pop(0)
            event.synchronize()
        while self._pending_free:
            _, event = self._pending_free[0]
            if event.query():
                self._pending_free.pop(0)
            else:
                break

    def make_evict_cb(self, name):
        """Return a zero-argument callback that evicts W_p[name]."""
        def _cb():
            self.evict_wp(name)
        return _cb

    def _schedule_prefetch(self, current_name):
        """Prefetch W_p for layers adjacent to current_name.

        Prefetches in both directions to cover forward (idx+d) and
        backward pass (idx-d) without needing to know which phase we are in.
        """
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
        """Synchronize the offload stream (wait for all pending transfers)."""
        self._stream.synchronize()

    # ═══════════════════════════════════════════════════════════════
    # Optimizer state offloading  (level 2)
    # ═══════════════════════════════════════════════════════════════

    def prefetch_optim_states(self, optimizer):
        """Move all optimizer state tensors from pinned CPU to GPU."""
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p)
                if state is None or "m" not in state:
                    continue
                pid = id(p)
                if pid not in self._optim_state_cpu:
                    continue  # first step: states already on GPU
                device = p.device
                for key in ("m", "v", "m_scale", "v_scale"):
                    state[key] = self._optim_state_cpu[pid][key].to(device)

    def evict_optim_states(self, optimizer):
        """Copy optimizer state tensors to pinned CPU, reclaim GPU memory."""
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p)
                if state is None or "m" not in state:
                    continue
                pid = id(p)
                if pid not in self._optim_state_cpu:
                    self._optim_state_cpu[pid] = {}
                    for key in ("m", "v", "m_scale", "v_scale"):
                        self._optim_state_cpu[pid][key] = torch.empty(
                            state[key].size(), dtype=state[key].dtype, pin_memory=True
                        )
                for key in ("m", "v", "m_scale", "v_scale"):
                    self._optim_state_cpu[pid][key].copy_(state[key])
                    state[key] = state[key].cpu()

    # ═══════════════════════════════════════════════════════════════
    # Optimizer compute offload  (level 3)
    # ═══════════════════════════════════════════════════════════════

    def compute_optim_step_cpu(self, optimizer, bias_correction1, bias_correction2):
        """Run the quantized AdamW update entirely on CPU.

        GPU is free during this call — can overlap with next forward pass
        when the training loop is structured for it.
        """
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
                cpu = self._optim_state_cpu.get(pid)
                if cpu is None:
                    continue

                # Ensure latest param + grad on CPU
                p_cpu = torch.empty(p.data.size(), dtype=p.data.dtype, pin_memory=True)
                g_cpu = torch.empty(p.grad.size(), dtype=p.grad.dtype, pin_memory=True)
                p_cpu.copy_(p.data)
                g_cpu.copy_(p.grad)

                m_i8 = cpu["m"]
                v_i8 = cpu["v"]
                m_scale = cpu["m_scale"]
                v_scale = cpu["v_scale"]

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

                # Write updated param back to GPU
                p.data.copy_(p_flat.reshape(p.data.shape))
