"""Minimal HTTP JSON client for L1 communication."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class ApiClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def post_json(self, path: str, payload: dict) -> dict | None:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = response.read().decode("utf-8")
                return json.loads(data) if data else {}
        except urllib.error.HTTPError as exc:
            return {"status": "error", "code": exc.code, "message": exc.reason}
        except urllib.error.URLError as exc:
            return {"status": "error", "message": str(exc)}

