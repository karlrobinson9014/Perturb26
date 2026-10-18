"""Download benchmark images from the public Perturb adversarial dataset.

The dataset's ``image`` column holds real past task images (clean), so the
benchmark matches the live task distribution. The dataset is stored as many
tiny parquet shards (1-2 rows each); a manifest pins the dataset *revision*
and the shard paths, so it stays reproducible while the dataset keeps growing.
Only the needed columns are range-read from the Hub.

    # dev set used by the loop (manifest is committed, images are not)
    python -m bench.fetch_data --manifest bench/manifests/dev.json --split validation --count 64 --seed 26
    # holdout: benchmarked only by the harness; lives OUTSIDE the repo, secret seed (setup_box.sh)
    python -m bench.fetch_data --manifest "$HOLDOUT_DIR/holdout.json" --split test --count 128 --seed <secret>

If the manifest exists its rows are re-downloaded and sha256-verified;
otherwise shards are drawn with ``--seed`` at the current revision and the
manifest is written. ``--with-adv`` also stores the network's verified
adversarial versions (competitor analysis).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

DATASET = "perturb-ai/efficientnet-v2-l-adv-dataset"


def _retry(fn, *, attempts: int = 5):
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # network / hub hiccups
            if attempt == attempts:
                raise
            print(f"  retry {attempt}: {exc}")
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


def _image_bytes(value) -> bytes | None:
    if isinstance(value, dict) and value.get("bytes"):
        return bytes(value["bytes"])
    return None


def read_row(fs, revision: str, shard: str, row: int, with_adv: bool) -> dict:
    import pyarrow.parquet as pq

    columns = ["image_id", "image", "label", "width", "height"] + (["adversarial"] if with_adv else [])

    def _read():
        with fs.open(f"datasets/{DATASET}@{revision}/{shard}") as handle:
            return pq.read_table(handle, columns=columns).slice(row, 1).to_pylist()[0]

    return _retry(_read)


def main() -> int:
    from huggingface_hub import HfApi, HfFileSystem

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--with-adv", action="store_true")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or None
    manifest_path: Path = args.manifest
    repo_root = Path(__file__).resolve().parents[1]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        api = HfApi(token=token)
        revision = _retry(lambda: api.dataset_info(DATASET).sha)
        files = _retry(lambda: api.list_repo_files(DATASET, repo_type="dataset", revision=revision))
        shards = sorted(f for f in files if f.startswith(f"data/{args.split}/") and f.endswith(".parquet"))
        chosen = sorted(random.Random(args.seed).sample(shards, min(args.count, len(shards))))
        if manifest_path.resolve().is_relative_to(repo_root):
            data_dir = repo_root / "data" / manifest_path.stem
        else:
            data_dir = manifest_path.parent / "data"
        manifest = {
            "dataset": DATASET, "revision": revision, "split": args.split,
            "shards_at_creation": len(shards),
            "data_dir": os.path.relpath(data_dir, manifest_path.parent.resolve()).replace("\\", "/"),
            "items": [{"shard": shard, "row": 0} for shard in chosen],
        }

    data_dir = (manifest_path.parent / manifest["data_dir"]).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem(token=token)
    for index, item in enumerate(manifest["items"], start=1):
        item.setdefault("key", Path(item["shard"]).stem + f"-r{item['row']}")
        target = data_dir / f"{item['key']}.img"
        if target.exists() and item.get("sha256") == hashlib.sha256(target.read_bytes()).hexdigest() and (
            not args.with_adv or "adv_files" in item
        ):
            continue
        record = read_row(fs, manifest["revision"], item["shard"], int(item["row"]), args.with_adv)
        raw = _image_bytes(record.get("image"))
        if raw is None:
            raise RuntimeError(f"{item['shard']} row {item['row']} has no image bytes")
        digest = hashlib.sha256(raw).hexdigest()
        if item.get("sha256") and item["sha256"] != digest:
            raise RuntimeError(f"{item['shard']} changed at pinned revision {manifest['revision'][:8]}?! sha mismatch")
        target.write_bytes(raw)
        item.update({"file": target.name, "sha256": digest, "image_id": record.get("image_id"),
                     "label": record.get("label"), "width": record.get("width"), "height": record.get("height")})
        if args.with_adv:
            adv_files = []
            for k, adv in enumerate(record.get("adversarial") or []):
                adv_raw = _image_bytes(adv)
                if adv_raw:
                    adv_target = data_dir / f"{item['key']}.adv{k:02d}.img"
                    adv_target.write_bytes(adv_raw)
                    adv_files.append(adv_target.name)
            item["adv_files"] = adv_files
        print(f"[{index}/{len(manifest['items'])}] {item['key']} {item['image_id']} {len(raw)}B", flush=True)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path} ({len(manifest['items'])} items, revision {manifest['revision'][:8]}) data: {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
