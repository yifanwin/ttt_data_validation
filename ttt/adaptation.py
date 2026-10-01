"""Episodic action-expert-only MolmoBot test-time training."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from olmo.torch_util import move_to_device


@dataclass(frozen=True)
class TTTSettings:
    steps: int = 20
    batch_size: int = 16
    microbatch_size: int = 1
    lr: float = 1e-5
    max_grad_norm: float = 1.0
    seed: int = 0


class ActionExpertTTT:
    """Keep one immutable base copy and one fresh optimizer per adaptation."""

    def __init__(self, model: torch.nn.Module, model_config, settings: TTTSettings = TTTSettings()):
        if settings.batch_size % settings.microbatch_size:
            raise ValueError("batch_size must be divisible by microbatch_size")
        if settings.microbatch_size != 1:
            raise ValueError("v1 uses microbatch_size=1 with gradient accumulation")
        if not hasattr(model, "action_expert"):
            raise TypeError("model has no action_expert")
        self.model = model
        self.model_config = model_config
        self.settings = settings
        # BF16 storage loses most 1e-5 updates; keep only the trainable module in FP32.
        self.model.action_expert.float()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("action_expert."))
        self.trainable = [p for p in model.parameters() if p.requires_grad]
        if not self.trainable:
            raise ValueError("action_expert has no trainable parameters")
        self.base = {name: tensor.detach().cpu().clone() for name, tensor in model.action_expert.state_dict().items()}
        self.optimizer: torch.optim.Optimizer | None = None
        self.preprocessor = model_config.build_preprocessor(for_inference=False, is_training=False)
        self.collator = model_config.build_collator(
            self.preprocessor.get_output_shapes(), pad_mode=None, include_metadata=False,
        )
        self.reset_count = 0

    def restore(self) -> None:
        # load_state_dict copies all trainable parameters and action-expert buffers.
        self.model.action_expert.load_state_dict(self.base, strict=True)
        self.model.zero_grad(set_to_none=True)
        self.optimizer = None
        self.model.eval()
        self.reset_count += 1

    def _one_loss(self, dataset, rng: np.random.Generator):
        index = int(rng.integers(len(dataset)))
        example = dataset.get(index, rng)
        processed = self.preprocessor(example)
        batch = self.collator([processed])
        inputs = {key: batch[key] for key in (
            "input_ids", "position_ids", "images", "image_masks", "token_pooling",
            "attention_mask", "states", "actions", "action_is_pad",
        ) if key in batch and batch[key] is not None}
        inputs = move_to_device(inputs, next(self.model.parameters()).device)
        if tuple(inputs["actions"].shape[-2:]) != (16, 20) or inputs["states"].shape[-1] != 22:
            raise ValueError("unexpected action/state shape in TTT batch")
        device = next(self.model.parameters()).device
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = self.model(**inputs)
            loss = output.internal["action_flow_loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite action_flow_loss")
        per_dim = [float(output.metrics[f"flow_loss_dim_{i}"].float().item()) for i in range(20)]
        return loss, per_dim

    def adapt(self, dataset) -> dict:
        if len(dataset) == 0:
            raise ValueError("selected dataset is empty")
        self.restore()
        settings = self.settings
        rng = np.random.default_rng(settings.seed)
        torch.manual_seed(settings.seed)
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.manual_seed_all(settings.seed)
            torch.cuda.reset_peak_memory_stats(device)
        self.optimizer = torch.optim.AdamW(self.trainable, lr=settings.lr, weight_decay=0.0)
        history = []
        start = time.perf_counter()
        try:
            for step in range(settings.steps):
                self.optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                per_dim_sum = np.zeros(20, dtype=np.float64)
                # Effective batch is fixed; no silent step or sample reduction on OOM.
                for _ in range(settings.batch_size):
                    loss, per_dim = self._one_loss(dataset, rng)
                    (loss / settings.batch_size).backward()
                    loss_sum += float(loss.detach().float().item())
                    per_dim_sum += per_dim
                grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable, settings.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("non-finite action_expert gradient")
                self.optimizer.step()
                history.append({
                    "step": step + 1,
                    "loss": loss_sum / settings.batch_size,
                    "flow_loss_dim": (per_dim_sum / settings.batch_size).tolist(),
                    "grad_norm": float(grad_norm.float().item()),
                })
        except Exception:
            self.restore()
            raise
        self.model.eval()
        device = next(self.model.parameters()).device
        changed = 0
        total = 0
        max_abs_delta = 0.0
        with torch.no_grad():
            for name, parameter in self.model.action_expert.named_parameters():
                delta = (parameter - self.base[name].to(parameter.device)).abs()
                changed += int((delta != 0).sum().item())
                total += delta.numel()
                max_abs_delta = max(max_abs_delta, float(delta.max().item()))
        peak_cuda_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        self.model.zero_grad(set_to_none=True)
        self.optimizer = None  # Adam moments are episode-local; release them before rollout.
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "steps": settings.steps, "batch_size": settings.batch_size,
            "microbatch_size": settings.microbatch_size, "lr": settings.lr,
            "elapsed_sec": time.perf_counter() - start,
            "peak_cuda_bytes": peak_cuda_bytes,
            "trainable_elements_changed": changed,
            "trainable_elements_total": total,
            "max_abs_parameter_delta": max_abs_delta,
            "history": history,
        }
