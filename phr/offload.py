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


class OffloadManager:
    """Manages CPU↔GPU tensor transfers for PHR model components."""

    def __init__(self, prefetch_depth=1):
        if not torch.cuda.is_available():
            raise RuntimeError("OffloadManager requires CUDA")

        self._prefetch_depth = max(prefetch_depth, 0)
        self._stream = torch.cuda.Stream()  # for async eviction copies

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
        """Pre-allocate per-param pinned CPU buffers and chunk groups.

        Called from enable_offload() before any optimizer step.
        """
        if self._opt_chunks:
            return

        # ── Pass 1: allocate per-param pinned CPU buffers ──
        for group in optimizer.param_groups:
            block = group.get("block_size", 256)
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                pid = id(p)
                self._opt_pid_to_param[pid] = p
                N = p.numel()
                nb = (N + block - 1) // block
                self._opt_state_cpu[pid] = {
                    "m":       torch.zeros(N,  dtype=torch.int8,    pin_memory=True),
                    "v":       torch.zeros(N,  dtype=torch.int8,    pin_memory=True),
                    "m_scale": torch.ones(nb,  dtype=torch.float32, pin_memory=True),
                    "v_scale": torch.ones(nb,  dtype=torch.float32, pin_memory=True),
                }

        # ── Pass 2: group pids into chunks (by byte budget) ──
        current_chunk = {"pids": [], "offsets": {}, "total": {}}
        for group in optimizer.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                pid = id(p)
                cpu = self._opt_state_cpu[pid]
                # Compute byte size of this param's states
                param_bytes = 0
                for key in ("m", "v", "m_scale", "v_scale"):
                    param_bytes += cpu[key].numel() * cpu[key].element_size()

                # Start new chunk if this param would exceed budget
                if current_chunk["pids"] and any(
                    current_chunk["total"].get(k, 0) + cpu[k].numel() * cpu[k].element_size() > CHUNK_BYTES
                    for k in ("m", "v")
                ):
                    self._opt_chunks.append(current_chunk)
                    current_chunk = {"pids": [], "offsets": {}, "total": {}}

                # Add to current chunk
                off = len(current_chunk["pids"])
                current_chunk["pids"].append(pid)
                current_chunk["offsets"][pid] = {}
                for key in ("m", "v", "m_scale", "v_scale"):
                    size = cpu[key].numel()
                    start = current_chunk["total"].get(key, 0)
                    current_chunk["offsets"][pid][key] = (start, size)
                    current_chunk["total"][key] = start + size

        if current_chunk["pids"]:
            self._opt_chunks.append(current_chunk)

    def prefetch_chunk(self, chunk_idx, optimizer):
        """Copy chunk's pinned CPU states to GPU via non-blocking transfers.

        All params' .to(device) calls are queued on the default stream,
        then we sync once.  Each param gets its own GPU tensor (no shared
        flat buffer), keeping peak GPU memory low.
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

        All copies run concurrently on the offload stream (after waiting
        for the default stream's Triton kernels).  We sync once per
        chunk, then free GPU memory by swapping to CPU views.
        """
        chunk = self._opt_chunks[chunk_idx]

        # Wait for default stream (Triton kernels) before copying
        self._stream.wait_stream(torch.cuda.current_stream())

        for pid in chunk["pids"]:
            cpu = self._opt_state_cpu[pid]
            p = self._opt_pid_to_param.get(pid)
            if p is None:
                continue
            state = optimizer.state[p]
            for key in ("m", "v", "m_scale", "v_scale"):
                with torch.cuda.stream(self._stream):
                    cpu[key].copy_(state[key])

        # Sync once — all copies are now complete
        self._stream.synchronize()

        # Safe to free GPU memory now
        for pid in chunk["pids"]:
            cpu = self._opt_state_cpu[pid]
            p = self._opt_pid_to_param.get(pid)
            if p is None:
                continue
            state = optimizer.state[p]
            for key in ("m", "v", "m_scale", "v_scale"):
                state[key] = cpu[key]  # CPU view (data is now in pinned buffer)
