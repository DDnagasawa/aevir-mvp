"""Data sandbox (path isolation + audit log)."""

from __future__ import annotations

import json
import os
import time
from typing import Any
from dataclasses import dataclass


@dataclass
class SandboxContext:
    root_dir: str
    run_dir: str
    audit_log: str
    pending_path: str


class DataSandbox:
    def __init__(self, base_dir: str = "./l1_edge/sandbox", allowed_roots: list[str] | None = None):
        self.base_dir = os.path.realpath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        self.allowed_roots = [os.path.realpath(p) for p in (allowed_roots or [self.base_dir])]

    def create_run_dir(self, run_id: str) -> SandboxContext:
        run_dir = os.path.join(self.base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        audit_log = os.path.join(run_dir, "audit.log")
        pending_path = os.path.join(self.base_dir, "pending.json")
        return SandboxContext(
            root_dir=self.base_dir,
            run_dir=run_dir,
            audit_log=audit_log,
            pending_path=pending_path,
        )

    def write_audit(self, audit_log: str, event: str, payload: dict) -> None:
        record = {
            "ts": int(time.time()),
            "event": event,
            "payload": payload,
        }
        with open(audit_log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _resolve_path(self, base: str, relative_path: str) -> str:
        path = os.path.realpath(os.path.join(base, relative_path))
        if not any(path.startswith(root + os.sep) or path == root for root in self.allowed_roots):
            raise ValueError("sandbox path is outside allowed roots")
        return path

    def write_file(self, run_dir: str, relative_path: str, data: bytes) -> str:
        path = self._resolve_path(run_dir, relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    def read_file(self, run_dir: str, relative_path: str) -> bytes:
        path = self._resolve_path(run_dir, relative_path)
        with open(path, "rb") as handle:
            return handle.read()

    def save_pending(self, pending_path: str, records: list[dict[str, Any]]) -> None:
        with open(pending_path, "w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False)

    def load_pending(self, pending_path: str) -> list[dict[str, Any]]:
        if not os.path.exists(pending_path):
            return []
        with open(pending_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def clear_pending(self, pending_path: str) -> None:
        if os.path.exists(pending_path):
            os.remove(pending_path)
