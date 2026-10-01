#!/usr/bin/env python3
"""Audit read-only NAS archive shards and build a leakage-filtered success index.

Run with MolmoBot's virtualenv Python (h5py and zstandard required). The outer
``arrow_table.json`` points to independent zstd-compressed tar entries; entries
sharing a house/name/inner HDF5 may contain different trajectories, so an archive
offset is part of every record ID. No source archive is unpacked permanently.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import h5py
import zstandard as zstd

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("RBY1PickAndPlaceDataGenConfig", "RBY1PickDataGenConfig")
DATA_ROOT = ROOT / "nas_wenyifan/molmospaces_data"
BENCH = ROOT / (
    "molmospaces_data/assets/benchmarks/molmospaces-bench-v1/procthor-10k/"
    "RBY1PickAndPlaceDataGenConfig/"
    "RBY1PickAndPlaceDataGenConfig_20260209_json_benchmark/benchmark.json"
)
OUT = Path(__file__).resolve().parents[1] / "artifacts"


def decode_json(dataset: h5py.Dataset) -> dict:
    raw = dataset[()]
    if isinstance(raw, bytes):
        return json.loads(raw.decode("utf-8"))
    return json.loads(raw.tobytes().rstrip(b"\0").decode("utf-8"))


def scan_entry(name: str, index: int, item: dict, benchmark_houses: set[int]) -> dict:
    shard = DATA_ROOT / name / "shards" / f"{item['shard_id']:05d}.tar"
    house_match = re.search(r"_house_(\d+)\.tar\.zst$", item["path"])
    assert house_match, item["path"]
    house = int(house_match.group(1))
    record = {
        "dataset": name,
        "entry_index": index,
        "archive_path": item["path"],
        "shard_id": item["shard_id"],
        "offset": item["offset"],
        "size": item["size"],
        "part": item["part"],
        "house_index": house,
        "benchmark_house_overlap": house in benchmark_houses,
        "trajectories": [],
    }
    try:
        with shard.open("rb") as fh:
            fh.seek(item["offset"])
            compressed = fh.read(item["size"])
        if len(compressed) != item["size"]:
            raise IOError(f"short read: {len(compressed)}/{item['size']}")
        tar_bytes = zstd.ZstdDecompressor().decompress(
            compressed, max_output_size=max(item["inflated_size"] + 1024 * 1024, 1024 * 1024)
        )
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".h5"):
                    continue
                h5bytes = archive.extractfile(member).read()
                with h5py.File(io.BytesIO(h5bytes), "r") as h5:
                    mask = h5.get("valid_traj_mask")
                    for key in h5:
                        if not re.fullmatch(r"traj_\d+", key):
                            continue
                        group = h5[key]
                        scene = decode_json(group["obs_scene"])
                        success = group["success"]
                        traj_num = int(key.split("_")[1])
                        valid = bool(mask[traj_num]) if mask is not None and traj_num < len(mask) else None
                        phases = group.get("obs/extra/policy_phase")
                        phase_values = sorted(set(int(x) for x in phases[:])) if phases is not None else []
                        t = {
                            "h5_member": member.name,
                            "h5_sha256": hashlib.sha256(h5bytes).hexdigest(),
                            "traj_key": key,
                            "length": len(success),
                            "valid_traj_mask": valid,
                            "terminal_success": bool(success[-1]) if len(success) else False,
                            "any_success": bool(success[:].any()) if len(success) else False,
                            "terminal_truncated": bool(group["truncated"][-1]) if "truncated" in group and len(success) else None,
                            "task": scene.get("task_type"),
                            "instruction": scene.get("task_description", ""),
                            "object_id": scene.get("object_name"),
                            "target_id": scene.get("place_receptacle_name"),
                            "stage_map": scene.get("policy_phases", {}),
                            "observed_phase_ids": phase_values,
                            "policy_dt_ms": scene.get("policy_dt_ms"),
                        }
                        t["eligible"] = bool(
                            t["terminal_success"]
                            and valid is not False
                            and not record["benchmark_house_overlap"]
                            and t["length"] > 2
                            and t["instruction"]
                        )
                        record["trajectories"].append(t)
        record["status"] = "ok"
    except Exception as exc:  # Keep a per-entry failure visible, never silently include it.
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["trajectories"] = []
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Maximum new available entries to scan")
    parser.add_argument("--snapshot", type=Path, default=None, help="Only scan shard IDs listed in this source_snapshot.json")
    args = parser.parse_args()
    snapshot_ids = None
    if args.snapshot is not None:
        snap = json.loads(args.snapshot.read_text())
        snapshot_ids = {name: {int(filename[:5]) for filename in files if filename.endswith(".tar")} for name, files in snap["shards"].items()}
    OUT.mkdir(exist_ok=True)
    benchmark = json.loads(BENCH.read_text())
    benchmark_houses = {int(x["house_index"]) for x in benchmark}
    path = OUT / "entry_audit.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            done.add((row["dataset"], row["entry_index"]))

    added = 0
    tables = {
        name: json.loads((DATA_ROOT / name / "arrow_table.json").read_text())
        for name in NAMES
    }
    with path.open("a", encoding="utf-8") as sink:
        for name in NAMES:
            table = tables[name]
            for index, item in enumerate(table):
                if "_house_" not in item["path"]:
                    continue
                if (name, index) in done:
                    continue
                if snapshot_ids is not None and item["shard_id"] not in snapshot_ids[name]:
                    continue
                shard = DATA_ROOT / name / "shards" / f"{item['shard_id']:05d}.tar"
                if not shard.is_file():
                    continue
                row = scan_entry(name, index, item, benchmark_houses)
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                added += 1
                if added % 250 == 0:
                    print(f"scanned {added} new entries; last={name}:{index}", flush=True)
                if args.limit is not None and added >= args.limit:
                    break
            if args.limit is not None and added >= args.limit:
                break

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    indexed = {(r["dataset"], r["entry_index"]) for r in rows}
    summary = {
        "benchmark_episodes": len(benchmark),
        "benchmark_houses": len(benchmark_houses),
        "datasets": {},
    }
    success_path = OUT / "success_index.jsonl"
    with success_path.open("w", encoding="utf-8") as sink:
        for name in NAMES:
            table = tables[name]
            house_items = [(i, x) for i, x in enumerate(table) if "_house_" in x["path"]]
            available = [(i, x) for i, x in house_items if (DATA_ROOT / name / "shards" / f"{x['shard_id']:05d}.tar").is_file()]
            current = [r for r in rows if r["dataset"] == name]
            counts = collections.Counter(r["status"] for r in current)
            trajs = [t for r in current for t in r["trajectories"]]
            counts["trajectories"] = len(trajs)
            counts["terminal_success"] = sum(t["terminal_success"] for t in trajs)
            counts["eligible"] = sum(t["eligible"] for t in trajs)
            counts["excluded_benchmark_house"] = sum(len(r["trajectories"]) for r in current if r["benchmark_house_overlap"])
            counts["invalid_mask"] = sum(t["valid_traj_mask"] is False for t in trajs)
            counts["success_nonterminal_only"] = sum(t["any_success"] and not t["terminal_success"] for t in trajs)
            for r in current:
                for t in r["trajectories"]:
                    if t["eligible"]:
                        sink.write(json.dumps({k: v for k, v in r.items() if k != "trajectories"} | t, ensure_ascii=False) + "\n")
            summary["datasets"][name] = {
                "total_house_entries": len(house_items),
                "available_house_entries": len(available),
                "scanned_house_entries": sum((name, i) in indexed for i, _ in available),
                "missing_house_entries": len(house_items) - len(available),
                "total_houses": len({re.search(r"_house_(\d+)", x["path"]).group(1) for _, x in house_items}),
                "benchmark_overlap_houses": len(benchmark_houses & {int(re.search(r"_house_(\d+)", x["path"]).group(1)) for _, x in house_items}),
                "counts": dict(counts),
            }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
