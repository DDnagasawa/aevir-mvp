"""Flower client integration for L1 edge."""

from __future__ import annotations

from typing import Any

try:
    import flwr as fl  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fl = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None

from .local_trainer import LocalTrainer
from .utils import hash_bytes


class L1FlowerClient:  # pragma: no cover - requires flwr runtime
    def __init__(self, trainer: LocalTrainer, epoch_id: str):
        if fl is None or np is None:
            raise RuntimeError("flwr and numpy are required for Flower client mode")
        self.trainer = trainer
        self.epoch_id = epoch_id
        self.parameters = [np.zeros((1,), dtype=np.float32)]

    def get_parameters(self, config: dict[str, Any] | None = None):
        return self.parameters

    def fit(self, parameters, config: dict[str, Any] | None = None):
        self.parameters = parameters
        raw = b"".join([p.tobytes() for p in parameters])
        result = self.trainer.train(raw, self.epoch_id)
        metrics = {"proof_hash": result.proof_hash, "model_hash": result.model_hash}
        return self.parameters, len(raw), metrics

    def evaluate(self, parameters, config: dict[str, Any] | None = None):
        raw = b"".join([p.tobytes() for p in parameters])
        loss = float(int(hash_bytes(raw), 16) % 1000) / 1000.0
        return loss, len(raw), {"loss": loss}


def start_flower_client(address: str, trainer: LocalTrainer, epoch_id: str) -> None:
    if fl is None or np is None:
        raise RuntimeError("flwr and numpy are required for Flower client mode")
    client = L1FlowerClient(trainer=trainer, epoch_id=epoch_id)
    fl.client.start_numpy_client(server_address=address, client=client)

