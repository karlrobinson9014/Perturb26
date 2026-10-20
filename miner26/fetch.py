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
import statistics
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable

logger = logging.getLogger("miner26")

FETCH_REQUEST_TIMEOUT_SECONDS = float(os.getenv("PERTURB26_FETCH_TIMEOUT_SECONDS", "3"))
FETCH_RETRY_SECONDS = float(os.getenv("PERTURB26_FETCH_RETRY_SECONDS", "0.5"))
DOWNLOAD_REQUEST_TIMEOUT_SECONDS = float(os.getenv("PERTURB26_DOWNLOAD_TIMEOUT_SECONDS", "4"))
DOWNLOAD_HEDGE_SECONDS = float(os.getenv("PERTURB26_DOWNLOAD_HEDGE_SECONDS", "1.5"))  # until 3 downloads are timed
DOWNLOAD_MAX_PARALLEL = int(os.getenv("PERTURB26_DOWNLOAD_MAX_PARALLEL", "3"))
_DOWNLOAD_SECONDS: deque[float] = deque(maxlen=20)  # durations of recent winning downloads (hedge_delay)


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


def hedge_delay() -> float:
    """When to start a parallel download attempt: ~3x the median recent download, within [0.8, 2.5] s.

    The task image host answers in 0.3-0.6 s (notes/timing.md), but a slow transfer can hold one request
    far past its timeout: ``requests``' timeout bounds each socket wait, not the whole download, and one
    live dry-run round (2026-10-20, round 10) spent ~11 s in a single attempt.
    """
    if len(_DOWNLOAD_SECONDS) < 3:
        return DOWNLOAD_HEDGE_SECONDS
    return min(2.5, max(0.8, 3.0 * statistics.median(_DOWNLOAD_SECONDS)))


def download_with_retries(
    download: Callable[..., str],
    image_url: str,
    *,
    deadline: float,
    request_timeout: float = DOWNLOAD_REQUEST_TIMEOUT_SECONDS,
    retry_seconds: float = FETCH_RETRY_SECONDS,
    hedge_after: float | None = None,
    max_parallel: int = DOWNLOAD_MAX_PARALLEL,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,  # kept for call compatibility; waits use the futures
) -> str:
    """Hedged ``download(image_url, timeout_seconds=...)`` (upstream ``image_url_to_b64``) until ``deadline``.

    Starts one attempt; whenever ``hedge_after`` seconds pass without a result (default ``hedge_delay()``),
    starts another in parallel, up to ``max_parallel`` in flight. A failed attempt is replaced after
    ``retry_seconds``. The first successful download wins; the others are abandoned (their threads finish
    or time out on their own). Raises the last error, or TimeoutError, when the deadline passes first.
    """
    hedge = hedge_delay() if hedge_after is None else hedge_after
    pool = ThreadPoolExecutor(max_workers=max(1, max_parallel), thread_name_prefix="img-dl")
    running: dict = {}  # future -> (attempt number, start time)
    attempts = 0
    last_error: BaseException | None = None
    next_start = clock()

    def start() -> None:
        nonlocal attempts
        attempts += 1
        now = clock()
        timeout = _request_timeout(deadline, request_timeout, now)
        running[pool.submit(download, image_url, timeout_seconds=timeout)] = (attempts, now)

    try:
        while True:
            now = clock()
            if now >= deadline:
                break
            if len(running) < max_parallel and now >= next_start:
                start()
                next_start = now + hedge
                if attempts > 1:
                    logger.info(f"image download: attempt={attempts} started in parallel ({len(running)} in flight)")
            wake = min(deadline, next_start if len(running) < max_parallel else deadline)
            if not running:  # every attempt failed; pause until the next one may start
                time.sleep(max(0.0, min(wake - clock(), retry_seconds)))
                continue
            done, _ = wait(list(running), timeout=max(0.0, wake - clock()), return_when=FIRST_COMPLETED)
            for future in done:
                attempt, started = running.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # timeouts, connection errors, HTTP errors: replace the attempt
                    last_error = exc
                    logger.info(f"image download attempt={attempt} failed after {clock() - started:.1f}s: {exc}")
                    next_start = min(next_start, clock() + retry_seconds)
                    continue
                took = clock() - started
                _DOWNLOAD_SECONDS.append(took)
                if attempt > 1 or attempts > 1:
                    logger.info(f"image download: attempt={attempt} of {attempts} won after {took:.1f}s")
                return result
        if last_error is not None:
            raise last_error
        raise TimeoutError(f"image download: no attempt finished by the deadline ({attempts} started)")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
