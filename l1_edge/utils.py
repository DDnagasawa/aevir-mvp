"""L1 utilities (hashing, hardware ID, gradient compression)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import uuid
import zlib


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_gpu_serial() -> str:
    return os.getenv("GPU_SERIAL", "UNKNOWN")


def get_mac_address() -> str:
    mac = uuid.getnode()
    return ":".join(f"{(mac >> shift) & 0xFF:02x}" for shift in range(40, -1, -8))


def get_mac_hash() -> str:
    return hash_bytes(get_mac_address().encode("utf-8"))


def compute_ash_id() -> str:
    gpu_serial = get_gpu_serial()
    mac_hash = get_mac_hash()
    return hash_bytes(f"{gpu_serial}|{mac_hash}".encode("utf-8"))


def compress_gradients(gradients: bytes) -> bytes:
    return zlib.compress(gradients, level=6)


def read_file_bytes(path: str | None) -> bytes:
    if not path:
        return b""
    with open(path, "rb") as handle:
        return handle.read()


def encode_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def decode_base64(data: str) -> bytes:
    return base64.b64decode(data.encode("utf-8"))


def sign_payload(payload: dict, secret: str | None) -> str:
    if not secret:
        return ""
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), packed, hashlib.sha256).hexdigest()
