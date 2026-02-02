"""Local trainer with Transformer + LoRA (minimal implementation)."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any

from .utils import compress_gradients, hash_bytes

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LoraConfig = None
    TaskType = None
    get_peft_model = None


@dataclass
class TrainResult:
    compressed_gradients: bytes
    proof_hash: str
    model_hash: str
    metrics: dict[str, Any]

class LocalTrainer:
    def __init__(
        self,
        use_dp: bool = False,
        model_name: str = "sshleifer/tiny-gpt2",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        max_steps: int = 3,
        batch_size: int = 2,
        lr: float = 2e-4,
        task_type: str = "CAUSAL_LM",
        device: str | None = None,
    ):
        self.use_dp = use_dp
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.lr = lr
        self.task_type = task_type
        self.device = device

    def train(self, model_params: bytes, epoch_id: str) -> TrainResult:
        if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("torch + transformers are required for training")
        if LoraConfig is None or get_peft_model is None or TaskType is None:
            raise RuntimeError("peft is required for LoRA training")

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model.to(device)

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            task_type=getattr(TaskType, self.task_type),
        )
        model = get_peft_model(model, lora_config)

        if model_params:
            try:
                state = torch.load(io.BytesIO(model_params), map_location="cpu")
                model.load_state_dict(state, strict=False)
            except Exception:
                pass

        texts = [
            "federated learning on the edge",
            "secure aggregation and proofs",
            "training with lora adapters",
            "resource monitoring and sandbox",
        ]
        dataset = [tokenizer(t, return_tensors="pt") for t in texts]
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        model.train()
        gradients_blob = b""
        for step in range(self.max_steps):
            batch = dataset[step % len(dataset)]
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()

            if step == 0:
                gradients_blob = self._collect_gradients(model)
                if self.use_dp:
                    gradients_blob = self._add_noise(gradients_blob)

            optimizer.step()
            optimizer.zero_grad()

        model_bytes = self._serialize_model(model)
        model_hash = hash_bytes(model_bytes)
        compressed = compress_gradients(gradients_blob)
        proof_hash = hash_bytes(gradients_blob)
        metrics = {
            "epoch_id": epoch_id,
            "loss": float(loss.detach().cpu().item()),
            "gradient_bytes": len(gradients_blob),
            "compressed_bytes": len(compressed),
            "device": device,
        }
        return TrainResult(
            compressed_gradients=compressed,
            proof_hash=proof_hash,
            model_hash=model_hash,
            metrics=metrics,
        )

    @staticmethod
    def _add_noise(data: bytes) -> bytes:
        noise = os.urandom(len(data))
        return bytes(b ^ n for b, n in zip(data, noise))

    @staticmethod
    def _collect_gradients(model: "torch.nn.Module") -> bytes:
        buffer = io.BytesIO()
        grads = {
            name: param.grad.detach().cpu()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        torch.save(grads, buffer)
        return buffer.getvalue()

    @staticmethod
    def _serialize_model(model: "torch.nn.Module") -> bytes:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.getvalue()
