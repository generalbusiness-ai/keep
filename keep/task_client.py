"""HTTP client for the keep.generalbusiness.ai Task API.

Submits delegatable processing tasks (summarize, OCR) to the hosted service
and polls for results.  Used by Keeper.process_pending() when a remote
backend is configured.

Reuses the same [remote] config (api_url, api_key, project) that
RemoteKeeper reads from keep.toml or environment variables.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Retry config for submit
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0  # seconds

# Timeouts
DEFAULT_TIMEOUT = 30.0
POLL_TIMEOUT = 10.0


class TaskClientError(Exception):
    """Error communicating with the Task API."""


class TaskClient:
    """HTTP client for the keep.generalbusiness.ai Task API."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        project: str | None = None,
    ):
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._project = project

        # Refuse non-HTTPS for remote APIs (bearer token would be sent in cleartext)
        if not self._api_url.startswith("https://"):
            from urllib.parse import urlparse
            host = urlparse(self._api_url).hostname or ""
            if host not in ("localhost", "127.0.0.1", "::1"):
                raise ValueError(
                    f"Task API URL must use HTTPS (got {self._api_url}). "
                    "Use HTTPS to protect API credentials, or use localhost for local development."
                )

        from .types import user_agent

        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": user_agent(),
        }
        if project:
            headers["X-Project"] = project

        self._client = httpx.Client(
            base_url=self._api_url,
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
        )
        self._available: bool | None = None  # cached after first check

    def discover_processors(self) -> list[str]:
        """GET /v1/processors -> list of task_type strings."""
        try:
            resp = self._client.get("/v1/processors")
            resp.raise_for_status()
            data = resp.json()
            return [p["task_type"] for p in data.get("processors", [])]
        except (httpx.HTTPError, KeyError) as e:
            logger.warning("Failed to discover processors: %s", e)
            return []

    def submit(
        self,
        task_type: str,
        content: str,
        metadata: dict | None = None,
        *,
        action_name: str | None = None,
        action_params: dict | None = None,
    ) -> str:
        """POST /v1/tasks -> task_id.

        Retries up to MAX_RETRIES times with exponential backoff on
        transient errors (5xx, timeouts, connection errors).
        """
        payload: dict = {"task_type": task_type, "content": content}
        if metadata:
            payload["metadata"] = metadata
        if action_name or action_params is not None:
            payload["action"] = {
                "name": action_name or task_type,
                "params": dict(action_params or {}),
            }

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.post("/v1/tasks", json=payload)
                if resp.status_code == 429:
                    # Rate limited — back off and retry
                    retry_after = min(float(resp.headers.get("Retry-After", "5")), 60.0)
                    logger.info("Rate limited, retrying after %.1fs", retry_after)
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["task_id"]
            except (httpx.HTTPStatusError) as e:
                if e.response.status_code < 500:
                    # Client error (4xx except 429) — don't retry
                    raise TaskClientError(
                        f"Task submission rejected: {e.response.status_code} {e.response.text}"
                    ) from e
                last_error = e
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e

            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.info(
                    "Task submit attempt %d failed, retrying in %.1fs: %s",
                    attempt + 1, delay, last_error,
                )
                time.sleep(delay)

        raise TaskClientError(
            f"Task submission failed after {MAX_RETRIES} attempts: {last_error}"
        ) from last_error

    def poll(self, task_id: str) -> dict:
        """GET /v1/tasks/{task_id} -> {status, result, error}.

        Returns dict with at least 'status' key.  When status is
        'completed', 'result' contains the TaskResult fields.
        When status is 'failed', 'error' contains the error message.
        """
        try:
            resp = self._client.get(
                f"/v1/tasks/{task_id}",
                timeout=POLL_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "status": data.get("status", "unknown"),
                "output": data.get("output"),
                "result": data.get("result"),
                "error": data.get("error"),
                "task_type": data.get("task_type"),
            }
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {
                    "status": "not_found",
                    "output": None,
                    "result": None,
                    "error": "Task not found",
                }
            raise TaskClientError(
                f"Poll failed: {e.response.status_code}"
            ) from e
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            raise TaskClientError(f"Poll failed: {e}") from e

    def acknowledge(self, task_id: str) -> None:
        """DELETE /v1/tasks/{task_id} — acknowledge a completed task."""
        try:
            resp = self._client.delete(f"/v1/tasks/{task_id}")
            # 404 is fine — task already cleaned up
            if resp.status_code != 404:
                resp.raise_for_status()
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            # Non-critical — task will be cleaned up by server retention
            logger.warning("Failed to acknowledge task %s: %s", task_id, e)

    @property
    def available(self) -> bool:
        """True if the service is reachable and configured."""
        if self._available is not None:
            return self._available
        try:
            processors = self.discover_processors()
            self._available = len(processors) > 0
        except Exception:
            self._available = False
        return self._available

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
