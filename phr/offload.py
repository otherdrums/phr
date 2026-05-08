"""
OffloadManager — GPU↔CPU streaming for PHR training parameters.

W_p streaming:  frozen uint8 indices live in pinned CPU RAM.
               GPU buffer pool limits concurrent GPU memory to O(1) layers.
               Copies run synchronously on the default stream to avoid
               races with cuBLAS internal streams.

Optimizer state streaming:  m, v, and per-block scales are stored as
               individual pinned CPU tensors.  During optimizer.step(),
               each param's states are prefetched to GPU just before its
               Triton kernel launch and evicted immediately after —
               peak GPU memory for optimizer states is bounded at
               O(max_param_size), not O(total_states).
"""

import torch


class OffloadManager:
    """Manages CPU↔GPU tensor transfers for PHR model components."""

    def __init__(self, prefetch_depth=1):
        if not torch.cuda.is_available():
            raise RuntimeError("OffloadManager requires CUDA")

        self._prefetch_depth = max(prefetch_depth, 0)

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

        # --- Optimizer state streaming ---
        # _opt_state_cpu[pid] = {"m": pinned_tensor, "v": pinned_tensor,
        #                         "m_scale": pinned_tensor, "v_scale": pinned_tensor}
        self._opt_state_cpu = {}
        self._opt_offsets = None  # set to True when initialized

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
        """Get a GPU tensor with canonical W_p data.

        Reuses a pooled buffer of matching shape, or allocates new.
        Copy runs synchronously on the default stream.
        """
        for i, gpu in enumerate(self._wp_gpu_pool):
            if gpu.shape == cpu_buf.shape:
                self._wp_gpu_pool.pop(i)
                gpu.copy_(cpu_buf)
                return gpu
        return cpu_buf.to(target_device)

    def _release_wp_buffer(self, gpu_tensor):
        """Return a GPU tensor to the pool for later reuse."""
        max_pool = max(self._prefetch_depth + 2, 3)
        if len(self._wp_gpu_pool) < max_pool:
            self._wp_gpu_pool.append(gpu_tensor)

    def prefetch_wp(self, name):
        """Synchronous CPU→GPU copy for W_p[name]."""
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
        """Ensure W_p[name] is on GPU.  Schedules prefetch of upcoming layers."""
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

    # ═══════════════════════════════════════════════════════════════
    # Optimizer state streaming  (per-param, no flat GPU buffer)
    # ═══════════════════════════════════════════════════════════════

    def _init_opt_offload(self, optimizer):
        """Pre-allocate individual pinned CPU tensors for each param's states.

        Called from enable_offload() before any optimizer step.
        After this, _init_state should not use GPU memory.
        """
        if self._opt_offsets is not None:
            return  # already initialized
        self._opt_offsets = True

        for group in optimizer.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                pid = id(p)
                N = p.numel()
                block = group.get("block_size", 256)
                nb = (N + block - 1) // block
                self._opt_state_cpu[pid] = {
                    "m":       torch.zeros(N,  dtype=torch.int8,    pin_memory=True),
                    "v":       torch.zeros(N,  dtype=torch.int8,    pin_memory=True),
                    "m_scale": torch.ones(nb,  dtype=torch.float32, pin_memory=True),
                    "v_scale": torch.ones(nb,  dtype=torch.float32, pin_memory=True),
                }

    def prefetch_param_states(self, state, pid, device):
        """Copy this param's states from pinned CPU to GPU."""
        cpu = self._opt_state_cpu.get(pid)
        if cpu is None:
            return
        for key in ("m", "v", "m_scale", "v_scale"):
            state[key] = cpu[key].to(device)

    def evict_param_states(self, state, pid):
        """Copy this param's states from GPU to pinned CPU, free GPU memory."""
        cpu = self._opt_state_cpu.get(pid)
        if cpu is None:
            return
        for key in ("m", "v", "m_scale", "v_scale"):
            cpu[key].copy_(state[key])
            state[key] = state[key].cpu()
