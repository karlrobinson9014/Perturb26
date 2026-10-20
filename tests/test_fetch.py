from __future__ import annotations

import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from miner26 import fetch as fetch_mod
from miner26.fetch import download_with_retries, fetch_new_task, hedge_delay, is_new_task, task_epoch
from perturbnet.api_client import get_current_task
from perturbnet.image_io import image_url_to_b64


class FakeApi:
    """Local HTTP server: ``/api/v1/task`` and ``/img.png`` follow a per-request script.

    Each script entry is ``"hang"`` (sleep longer than any client timeout), ``"500"``, or a payload to serve.
    The last entry repeats once the script is exhausted.
    """

    def __init__(self, task_script: list, image_script: list) -> None:
        self.task_script, self.image_script = list(task_script), list(image_script)
        self.task_calls = self.image_calls = 0
        self.stop = threading.Event()
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                if self.path.endswith("/task"):
                    step = api.task_script[min(api.task_calls, len(api.task_script) - 1)]
                    api.task_calls += 1
                    body, ctype = json.dumps(step).encode(), "application/json"
                elif self.path.endswith("/img.png"):
                    step = api.image_script[min(api.image_calls, len(api.image_script) - 1)]
                    api.image_calls += 1
                    body, ctype = step, "image/png"
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                if step == "hang":
                    api.stop.wait(3.0)
                    return
                if step == "trickle":  # a valid image, a few bytes at a time: never idle long enough to time out
                    png = _png_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(png)))
                    self.end_headers()
                    for i in range(0, len(png), 8):
                        if api.stop.wait(0.25):
                            return
                        self.wfile.write(png[i:i + 8])
                        self.wfile.flush()
                    return
                if step == "500":
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b"boom")
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 6), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _task(task_id: str, base: str = "http://x") -> dict:
    return {"task_id": task_id, "imageURL": f"{base}/img.png"}


BOUNDARY = 1792372440


@pytest.fixture
def api_factory():
    made = []

    def make(task_script, image_script=None):
        api = FakeApi(task_script, image_script or [_png_bytes()])
        made.append(api)
        return api

    yield make
    for api in made:
        api.close()


def test_task_epoch_and_newness():
    assert task_epoch(f"{BOUNDARY}-hf-abc") == BOUNDARY
    assert task_epoch("hf-abc") is None
    assert is_new_task(f"{BOUNDARY}-hf-b", f"{BOUNDARY - 120}-hf-a", BOUNDARY)
    assert not is_new_task(f"{BOUNDARY}-hf-b", f"{BOUNDARY}-hf-b", BOUNDARY)
    # previous round's task when the pre-boundary snapshot failed: stale, not new
    assert not is_new_task(f"{BOUNDARY - 120}-hf-a", f"{BOUNDARY - 240}-hf-z", BOUNDARY)
    assert is_new_task("opaque-id", None, BOUNDARY)
    assert not is_new_task("", None, BOUNDARY)


def test_fetch_survives_timeout_and_http_error(api_factory):
    old, new = _task(f"{BOUNDARY - 120}-hf-old"), _task(f"{BOUNDARY}-hf-new")
    api = api_factory(["hang", "500", old, new])
    t0 = time.time()
    task = fetch_new_task(
        get_current_task, base_url=api.base + "/api/v1", last_task_id=old["task_id"], boundary=BOUNDARY,
        deadline=t0 + 10, request_timeout=0.6, retry_seconds=0.05,
    )
    elapsed = time.time() - t0
    assert task is not None and task.task_id == new["task_id"]
    assert api.task_calls == 4
    # one short timeout + three quick requests; upstream would have raised out after a 10 s timeout
    assert elapsed < 2.5


def test_fetch_rejects_stale_task_without_snapshot(api_factory):
    stale, new = _task(f"{BOUNDARY - 120}-hf-old"), _task(f"{BOUNDARY}-hf-new")
    api = api_factory([stale, stale, new])
    task = fetch_new_task(
        get_current_task, base_url=api.base, last_task_id="", boundary=BOUNDARY,
        deadline=time.time() + 5, request_timeout=0.6, retry_seconds=0.05,
    )
    assert task.task_id == new["task_id"] and api.task_calls == 3


def test_fetch_gives_up_at_deadline(api_factory):
    api = api_factory(["hang"])
    t0 = time.time()
    task = fetch_new_task(
        get_current_task, base_url=api.base, last_task_id=None, boundary=BOUNDARY,
        deadline=t0 + 1.5, request_timeout=0.6, retry_seconds=0.05,
    )
    assert task is None
    assert time.time() - t0 < 1.5 + 0.6 + 0.3  # never starts a request after the deadline, each capped


def test_download_retries_then_succeeds(api_factory):
    png = _png_bytes()
    api = api_factory([_task("x")], ["hang", "500", png])
    t0 = time.time()
    b64 = download_with_retries(
        image_url_to_b64, api.base + "/img.png", deadline=t0 + 10, request_timeout=0.6, retry_seconds=0.05
    )
    assert time.time() - t0 < 2.5 and api.image_calls == 3
    assert Image.open(io.BytesIO(base64.b64decode(b64))).size == (8, 6)


def test_download_raises_after_deadline(api_factory):
    api = api_factory([_task("x")], ["500"])
    with pytest.raises(RuntimeError):
        download_with_retries(
            image_url_to_b64, api.base + "/img.png", deadline=time.time() + 0.5, request_timeout=0.3, retry_seconds=0.05
        )
    assert api.image_calls >= 2


def test_fetch_with_fake_clock_counts_attempts():
    calls = []
    now = [100.0]

    def get_task(*, base_url, timeout_seconds):
        calls.append(timeout_seconds)
        now[0] += timeout_seconds  # every request times out
        raise TimeoutError("slow")

    def sleep(s):
        now[0] += s

    task = fetch_new_task(
        get_task, base_url="b", last_task_id=None, boundary=100.0, deadline=110.0,
        request_timeout=3.0, retry_seconds=0.5, clock=lambda: now[0], sleep=sleep,
    )
    assert task is None
    # 3 + 0.5 + 3 + 0.5 + 3 + 0.5 = 110.5 > deadline would be exceeded, so the third request is capped
    assert calls[:2] == [3.0, 3.0] and all(t <= 3.0 for t in calls) and now[0] <= 110.0 + 0.5


def test_download_hedges_past_a_slow_transfer(api_factory):
    # Live dry run 2026-10-20 round 10: one attempt trickled for ~11 s without ever timing out.
    png = _png_bytes()
    api = api_factory([_task("x")], ["trickle", png])
    t0 = time.time()
    b64 = download_with_retries(
        image_url_to_b64, api.base + "/img.png", deadline=t0 + 10, request_timeout=1.0, hedge_after=0.3
    )
    elapsed = time.time() - t0
    assert Image.open(io.BytesIO(base64.b64decode(b64))).size == (8, 6)
    assert api.image_calls == 2 and elapsed < 1.5, (api.image_calls, elapsed)  # the trickle alone takes ~2.5 s


def test_download_fast_path_uses_one_request(api_factory):
    api = api_factory([_task("x")], [_png_bytes()])
    download_with_retries(image_url_to_b64, api.base + "/img.png", deadline=time.time() + 5, hedge_after=1.0)
    assert api.image_calls == 1


def test_download_times_out_when_every_attempt_stalls(api_factory):
    api = api_factory([_task("x")], ["trickle"])
    t0 = time.time()
    with pytest.raises(TimeoutError):
        download_with_retries(image_url_to_b64, api.base + "/img.png", deadline=t0 + 0.8, hedge_after=0.2, max_parallel=2)
    assert time.time() - t0 < 1.5 and api.image_calls == 2


def test_hedge_delay_adapts_to_measured_downloads(monkeypatch):
    monkeypatch.setattr(fetch_mod, "_DOWNLOAD_SECONDS", fetch_mod.deque(maxlen=20))
    assert hedge_delay() == fetch_mod.DOWNLOAD_HEDGE_SECONDS  # not enough samples yet
    fetch_mod._DOWNLOAD_SECONDS.extend([0.4, 0.5, 0.45])
    assert hedge_delay() == pytest.approx(1.35)
    fetch_mod._DOWNLOAD_SECONDS.extend([0.05] * 10)
    assert hedge_delay() == 0.8  # floor
    fetch_mod._DOWNLOAD_SECONDS.extend([3.0] * 20)
    assert hedge_delay() == 2.5  # cap
