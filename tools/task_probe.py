"""Measure live task timing on the Perturb API (read-only: GET /task and the task image).

For each task change it records when the new task_id became visible relative to
the 120 s wall-clock boundary, how long GET /task and the image download take,
and the local clock's skew against the server's Date header (miners and
validators both schedule by wall clock, so skew eats into the attack budget).

    python -m tools.task_probe --boundaries 10 --out results/timing-probe.json
"""
from __future__ import annotations

import argparse
import email.utils
import io
import json
import statistics
import sys
import time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.perturbai.io/api/v1"
CADENCE = 120


def signed_offset(t: float, cadence: int = CADENCE) -> float:
    """Seconds from the nearest boundary (negative = before it)."""
    rem = t % cadence
    return rem if rem < cadence / 2 else rem - cadence


def get_task(session: requests.Session, timeout: float) -> tuple[dict | None, float, float | None]:
    t0 = time.time()
    response = session.get(f"{API}/task", timeout=timeout)
    t1 = time.time()
    skew = None
    if response.headers.get("Date"):
        server = email.utils.parsedate_to_datetime(response.headers["Date"]).timestamp()
        skew = (t0 + t1) / 2 - server  # >0: local clock ahead (Date has 1 s resolution)
    payload = response.json() if response.status_code == 200 and response.text.strip() else None
    return payload, t1 - t0, skew


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boundaries", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "timing-probe.json")
    args = parser.parse_args()

    session = requests.Session()
    events, latencies, skews, errors = [], [], [], 0
    last_id = None
    stop_after = (int(time.time() // CADENCE) + args.boundaries + 1) * CADENCE + 30
    print(f"probing until {time.strftime('%H:%M:%S', time.localtime(stop_after))} ({args.boundaries} boundaries)", flush=True)
    while time.time() < stop_after:
        polled = time.time()
        try:
            payload, latency, skew = get_task(session, args.timeout)
        except requests.RequestException as exc:
            errors += 1
            print(f"  poll error: {exc}", flush=True)
            time.sleep(args.interval)
            continue
        latencies.append(latency)
        if skew is not None:
            skews.append(skew)
        task_id = (payload or {}).get("task_id")
        if task_id and task_id != last_id:
            seen = polled + latency / 2
            event = {"task_id": task_id, "seen_at": seen, "offset_s": round(signed_offset(seen), 3),
                     "poll_latency_s": round(latency, 3), "first_observation": last_id is None}
            prefix = task_id.split("-", 1)[0]
            if prefix.isdigit():
                event["id_epoch_minus_seen_s"] = round(int(prefix) - seen, 3)
            try:
                d0 = time.time()
                image = session.get(payload.get("imageURL") or payload.get("imageUrl"), timeout=args.timeout)
                event["download_s"] = round(time.time() - d0, 3)
                event["image_bytes"] = len(image.content)
                event["image_size"] = list(Image.open(io.BytesIO(image.content)).size)
            except Exception as exc:  # keep probing
                event["download_error"] = str(exc)
            events.append(event)
            last_id = task_id
            print(f"  new task {task_id[:40]} offset={event['offset_s']:+.2f}s poll={latency:.2f}s "
                  f"download={event.get('download_s')}s size={event.get('image_size')}", flush=True)
        time.sleep(max(0.0, args.interval - (time.time() - polled)))

    changes = [e for e in events if not e["first_observation"]]
    offsets = [e["offset_s"] for e in changes]
    downloads = [e["download_s"] for e in changes if "download_s" in e]
    summary = {
        "changes_observed": len(changes), "poll_errors": errors,
        "offset_s": {"min": min(offsets), "median": statistics.median(offsets), "max": max(offsets)} if offsets else None,
        "poll_latency_s": {"median": statistics.median(latencies), "max": max(latencies)} if latencies else None,
        "download_s": {"median": statistics.median(downloads), "max": max(downloads)} if downloads else None,
        "clock_skew_s_median": statistics.median(skews) if skews else None,
        "note": "offset_s: when a new task became visible relative to the nearest 120 s boundary "
                "(poll interval adds up to --interval of delay); clock skew has 1 s resolution",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "events": events}, indent=1) + "\n", encoding="utf-8")
    print("PROBE_RESULT " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
