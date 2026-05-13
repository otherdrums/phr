"""StreamTrainer — unified forward+train for continuous token streams.

Replicates the harness training micro-step exactly:
    forward → loss / acc_steps → backward → optimizer.step

Supports single-example prompts and batched inputs.  Inference passes
skip backward and optimizer, producing logits only.

In ZPackR mode (cv2lrt=None), block-level WeightDict compression provides
the convergence signal — no external LR scheduler is needed.
"""

import torch
import torch.nn.functional as F


class StreamTrainer:
    """Unified train/inference processor mirroring the harness micro-step.

    Zero structural overhead vs the harness — same model, same optimizer,
    optional Velvet controller for PackR mode, same gradient path.

    In ZPackR mode, leave cv2lrt=None.  The WeightDict's block-level
    compression ratios handle convergence natively via ZPackRLinear.post_step().
    """

    def __init__(
        self,
        model,
        optimizer,
        cv2lrt=None,
        acc_steps=4,
        device=None,
        post_opt_step_fn=None,     # callable() fired after each optimizer.step()
    ):
        self.model = model
        self.optimizer = optimizer
        self.cv2lrt = cv2lrt
        self.acc_steps = acc_steps
        self.device = device or next(model.parameters()).device
        self._post_opt_step_fn = post_opt_step_fn

        # Internal counters
        self._micro_step = 0       # forward+backward calls since last optimizer.step
        self._global_step = 0      # total optimizer.step() calls
        self._warmup_done = False  # CV2LRT warmup phase completed

        # Running training stats (reset with reset_stats())
        self._train_loss = 0.0
        self._train_correct = 0
        self._train_total = 0

    # ── Training step ──

    def step(self, input_ids, attention_mask, label):
        """One training micro-step — identical to harness lines 56-80 of training.py.

        Args:
            input_ids:      [L] or [B, L] int tensor
            attention_mask: [L] or [B, L] int tensor
            label:          scalar, [1], or [B] int tensor

        Returns:
            loss_val: float, the unscaled (pre-accumulation) loss
            logits:   [B, C] tensor of class logits
        """
        # Ensure batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if isinstance(label, int):
            label = torch.tensor([label], device=self.device)
        elif label.dim() == 0:
            label = label.unsqueeze(0).to(self.device)
        else:
            label = label.to(self.device)

        self.model.train()

        # Forward — identical to harness
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(outputs.logits, label) / self.acc_steps

        # Backward — gradient flows to W_f via PHRMatmulFunction
        loss.backward()

        unscaled = loss.item() * self.acc_steps
        self._train_loss += unscaled
        self._train_correct += (outputs.logits.argmax(-1) == label).sum().item()
        self._train_total += label.size(0)

        self._micro_step += 1

        # Optimizer step on accumulation boundary
        if self._micro_step % self.acc_steps == 0:
            self.optimizer.step()
            if self.cv2lrt is not None and self._warmup_done:
                self.cv2lrt.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._global_step += 1
            if self._post_opt_step_fn is not None:
                self._post_opt_step_fn(self._global_step)

        return unscaled, outputs.logits

    # ── CV2LRT warmup ──

    def warmup_step(self, global_step: int, warmup_steps: int):
        """Apply CV2LRT linear warmup LR schedule (called per micro-step during warmup)."""
        if self.cv2lrt is not None:
            self.cv2lrt.warmup_step(global_step, warmup_steps)
            if global_step >= warmup_steps:
                self._warmup_done = True

    # ── Inference ──

    @torch.no_grad()
    def eval(self, input_ids, attention_mask=None):
        """Pure inference forward — no backward, no optimizer step.

        Args:
            input_ids:      [L] or [B, L]
            attention_mask: [L] or [B, L] (optional)

        Returns:
            logits: [B, C]
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        self.model.eval()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    # ── Stats ──

    @property
    def running_acc(self):
        if self._train_total == 0:
            return 0.0
        return 100.0 * self._train_correct / self._train_total

    @property
    def running_loss(self):
        if self._micro_step == 0:
            return 0.0
        return self._train_loss / self._micro_step

    def reset_stats(self):
        self._train_loss = 0.0
        self._train_correct = 0
        self._train_total = 0
