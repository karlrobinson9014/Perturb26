"""Fast task fetch for the live miner.

Upstream polls ``GET /task`` with a 10 s timeout and 2 s sleeps, and a single
timeout raises out of its loop, which skips the whole round. Measured from the
dev machine (notes/timing.md): the new task is visible ~3 s after the boundary,
GET /task takes ~2 s with spikes to 8 s and occasional 10 s timeouts. Here each
request gets a short timeout, failures are retried after a short pause, and the
loop runs until a deadline instead of a fixed attempt count.

A task is new when its id differs from the last processed one and, if the id
carries the boundary epoch as its prefix (``1792372440-hf-...``), that epoch is
not older than the current boundary. The second check keeps a failed
pre-boundary snapshot from making the miner attack the previous round's task.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable

logger = logging.getLogger("miner26")

FETCH_REQUEST_TIMEOUT_SECONDS = float(os.getenv("PERTURB26_FETCH_TIMEOUT_SECONDS", "3"))
FETCH_RETRY_SECONDS = float(os.getenv("PERTURB26_FETCH_RETRY_SECONDS", "0.5"))
DOWNLOAD_REQUEST_TIMEOUT_SECONDS = float(os.getenv("PERTURB26_DOWNLOAD_TIMEOUT_SECONDS", "4"))


def task_epoch(task_id: str) -> int | None:
    """Boundary epoch encoded as the task id's prefix, or None when the id has no numeric prefix."""
    prefix = str(task_id).split("-", 1)[0]
    return int(prefix) if prefix.isdigit() else None


def is_new_task(task_id: str, last_task_id: str | None, boundary: float, epoch_tolerance: float = 5.0) -> bool:
    if not task_id or task_id == last_task_id:
        return False
    epoch = task_epoch(task_id)
    return epoch is None or epoch >= boundary - epoch_tolerance


def _request_timeout(deadline: float, per_request: float, now: float) -> float:
    return max(0.5, min(per_request, deadline - now))


def fetch_new_task(
    get_task: Callable[..., object],
    *,
    base_url: str,
    last_task_id: str | None,
    boundary: float,
    deadline: float,
    request_timeout: float = FETCH_REQUEST_TIMEOUT_SECONDS,
    retry_seconds: float = FETCH_RETRY_SECONDS,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
):
    """Poll ``get_task(base_url=..., timeout_seconds=...)`` until a new task appears or ``deadline`` passes.

    ``get_task`` is upstream ``perturbnet.api_client.get_current_task`` (returns an object with
    ``task_id`` / ``image_url`` or None, raises on HTTP errors and timeouts). Returns the task or None.
    """
    attempt = 0
    while True:
        attempt += 1
        now = clock()
        try:
            task = get_task(base_url=base_url, timeout_seconds=_request_timeout(deadline, request_timeout, now))
        except Exception as exc:  # timeouts, connection errors, HTTP 5xx: retry
            logger.info(f"task fetch attempt={attempt} failed after {clock() - now:.1f}s: {exc}")
            task = None
        else:
            if task is not None and is_new_task(task.task_id, last_task_id, boundary):
                if attempt > 1:
                    logger.info(f"new task after attempt={attempt} since_boundary={clock() - boundary:.1f}s")
                return task
        if clock() + retry_seconds >= deadline:
            logger.warning(f"no new task by deadline (since_boundary={clock() - boundary:.1f}s, attempts={attempt})")
            return None
        sleep(retry_seconds)


def download_with_retries(
    download: Callable[..., str],
    image_url: str,
    *,
    deadline: float,
    request_timeout: float = DOWNLOAD_REQUEST_TIMEOUT_SECONDS,
    retry_seconds: float = FETCH_RETRY_SECONDS,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """``download(image_url, timeout_seconds=...)`` (upstream ``image_url_to_b64``) with retries until ``deadline``.

    Raises the last error when the deadline passes without a successful download.
    """
    attempt = 0
    while True:
        attempt += 1
        now = clock()
        try:
            return download(image_url, timeout_seconds=_request_timeout(deadline, request_timeout, now))
        except Exception as exc:
            logger.info(f"image download attempt={attempt} failed after {clock() - now:.1f}s: {exc}")
            if clock() + retry_seconds >= deadline:
                raise
        sleep(retry_seconds)
