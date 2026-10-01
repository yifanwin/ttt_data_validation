#!/usr/bin/env python3
"""G1：数据充分性与泄漏上限核算（纯 CPU，零 GPU 成本）。

回答三个问题：
  1. 检索库按 (task, stage) 分桶后，每个桶有多少可用 clip？
  2. 检索库与 benchmark 的语义字段（object/target/instruction）重合到什么程度？
  3. 有哪些 benchmark episode 因畸形 instruction 而应被隔离？

只读消费 ttt_data_validation/artifacts/ 下的步骤一产物与 benchmark JSON，
不打开任何源归档，不修改任何文件。
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import gate_config as cfg

# 离线轨迹的阶段名（步骤一已确认 17120 条全部带同一套 stage_map）
STAGE_NAMES = ("pregrasp", "grasp", "lift", "place", "postplace", "done")


def load_success_index() -> list[dict]:
    rows = []
    with cfg.SUCCESS_INDEX.open(encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def load_benchmark() -> list[dict]:
    return json.loads(cfg.BENCHMARK_JSON.read_text(encoding="utf-8"))


def load_exclusions() -> dict:
    return json.loads(cfg.LEAKAGE_EXCLUSIONS.read_text(encoding="utf-8"))


def object_category(name: str | None) -> str | None:
    """从 object_id / pickup_obj_name 提取语义类别前缀。

    形如 'remotecontrol_488ea7924f7cb4d329c991926df62222_1_0_3' → 'remotecontrol'。
    取不到时退回第一个下划线前的片段，保持与 benchmark 侧同口径。
    """
    if not name:
        return None
    match = re.match(r"^([a-zA-Z]+?)_[0-9a-f]{32}", name)
    return (match.group(1) if match else name.split("_")[0]).lower()


def stages_of(row: dict) -> list[str]:
    """返回该轨迹经过的全部阶段名（有序，按 phase id 升序）。

    observed_phase_ids 是整数序列，stage_map 给出名字→编号映射。
    实测：Pick 轨迹只经过 {0,1,2,5}，PnP 经过全部 {0..5}；
    因此必须逐阶段计数，取 min() 会把所有轨迹都压成 pregrasp。
    """
    phase_ids = row.get("observed_phase_ids") or []
    stage_map = row.get("stage_map") or {}
    if not phase_ids or not stage_map:
        return []
    inverse = {int(v): k for k, v in stage_map.items()}
    return [inverse[i] for i in sorted(int(x) for x in phase_ids) if i in inverse]


def bucket_key(row: dict) -> tuple[str, str]:
    """分桶键：task + 该轨迹实际覆盖的最新阶段（最接近任务完成的那一阶段）。"""
    stages = stages_of(row)
    return (row.get("task") or "unknown", stages[-1] if stages else "unknown")


def bench_malformed(rows: list[dict]) -> list[dict]:
    """挑出 instruction 命中畸形模式的 benchmark episode。"""
    hits = []
    for episode in rows:
        text = (episode.get("language") or {}).get("task_description") or ""
        low = text.lower()
        matched = [p for p in cfg.MALFORMED_INSTRUCTION_PATTERNS if p in low]
        if matched:
            hits.append(
                {
                    "house_index": episode["house_index"],
                    "task_description": text,
                    "matched_patterns": matched,
                }
            )
    return hits


def cross_compare(
    index_rows: list[dict], bench_rows: list[dict]
) -> dict:
    """检索库与 benchmark 的语义字段交叉比对，给出泄漏上限。

    三类重合：object 语义类别、target 语义类别、instruction 精确匹配。
    这是「同源同 split」（benchmark 源自 PnP/val）下唯一零成本可做的近重复检查。
    """
    index_objects = collections.Counter(
        object_category(r.get("object_id")) for r in index_rows
    )
    index_targets = collections.Counter(
        object_category(r.get("target_id")) for r in index_rows
    )
    index_instructions = {r.get("instruction", "").strip().lower() for r in index_rows}

    bench_objects = collections.Counter()
    bench_targets = collections.Counter()
    bench_instructions = collections.Counter()
    for episode in bench_rows:
        task = episode.get("task") or {}
        bench_objects[object_category(task.get("pickup_obj_name"))] += 1
        bench_targets[object_category(task.get("place_receptacle_name"))] += 1
        text = ((episode.get("language") or {}).get("task_description") or "").strip().lower()
        bench_instructions[text] += 1

    shared_objects = {
        k: {"index": index_objects.get(k, 0), "benchmark": bench_objects.get(k, 0)}
        for k in sorted(set(bench_objects) & set(index_objects))
        if k
    }
    shared_targets = {
        k: {"index": index_targets.get(k, 0), "benchmark": bench_targets.get(k, 0)}
        for k in sorted(set(bench_targets) & set(index_targets))
        if k
    }
    exact_instruction_hits = sorted(
        text for text in bench_instructions if text and text in index_instructions
    )

    # 泄漏上限：即使 house 级排除生效，语义同类的 episode 仍可能构成近重复。
    # 这里给出的是「不能靠 house 排除消除」的那部分覆盖面。
    covered = 0
    for episode in bench_rows:
        cat = object_category((episode.get("task") or {}).get("pickup_obj_name"))
        if cat and index_objects.get(cat, 0) > 0:
            covered += 1
    bench_episodes_semantically_covered = covered
    return {
        "index_object_categories": len([k for k in index_objects if k]),
        "benchmark_object_categories": len([k for k in bench_objects if k]),
        "shared_object_categories": shared_objects,
        "shared_target_categories": shared_targets,
        "exact_instruction_matches": len(exact_instruction_hits),
        "exact_instruction_examples": exact_instruction_hits[:10],
        "benchmark_episodes_with_object_category_present_in_index": bench_episodes_semantically_covered,
        "benchmark_episode_total": len(bench_rows),
        "note": (
            "object/target/instruction 语义重合是 house 级排除挡不住的部分。"
            "精确 instruction 命中数为 0 不代表无近重复，只说明文本未逐字相同。"
        ),
    }


def bucket_report(index_rows: list[dict]) -> dict:
    counts = collections.Counter(bucket_key(r) for r in index_rows)
    by_task = collections.Counter(r.get("task") or "unknown" for r in index_rows)
    # 逐阶段计数：同一条轨迹会贡献给它经过的每个阶段
    by_stage: collections.Counter = collections.Counter()
    for row in index_rows:
        for stage in stages_of(row):
            by_stage[stage] += 1
    # 阶段覆盖组合：区分 Pick（无 place/postplace）与 PnP
    stage_sets = collections.Counter(
        tuple(stages_of(r)) for r in index_rows
    )
    short = sorted(
        [{"task": t, "stage": s, "count": c} for (t, s), c in counts.items() if c < cfg.G1_MIN_BUCKET_SIZE],
        key=lambda x: x["count"],
    )
    return {
        "total": len(index_rows),
        "by_task": dict(by_task),
        "by_stage_trajectory_count": dict(by_stage),
        "by_task_stage": {f"{t}/{s}": c for (t, s), c in sorted(counts.items())},
        "stage_coverage_patterns": {
            "/".join(k) if k else "unknown": v for k, v in stage_sets.most_common()
        },
        "buckets_below_threshold": short,
        "threshold": cfg.G1_MIN_BUCKET_SIZE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="G1：数据充分性与泄漏上限核算")
    parser.add_argument(
        "--out",
        type=Path,
        default=cfg.ARTIFACT_DIR / "g1_data_sufficiency.json",
    )
    args = parser.parse_args()

    index_rows = load_success_index()
    bench_rows = load_benchmark()
    exclusions = load_exclusions()

    buckets = bucket_report(index_rows)
    leak = cross_compare(index_rows, bench_rows)
    malformed = bench_malformed(bench_rows)

    # 复核步骤一的 house 级排除确实生效
    bench_houses = {int(e["house_index"]) for e in bench_rows}
    index_houses = {int(r["house_index"]) for r in index_rows}
    house_overlap = bench_houses & index_houses

    # 判定
    findings = []
    verdict = cfg.VERDICT_PASS
    if house_overlap:
        verdict = cfg.VERDICT_FAIL
        findings.append(
            f"检索库仍有 {len(house_overlap)} 个 house 与 benchmark 重合，house 级排除未生效"
        )
    if buckets["buckets_below_threshold"]:
        if verdict == cfg.VERDICT_PASS:
            verdict = cfg.VERDICT_WARN
        findings.append(
            f"{len(buckets['buckets_below_threshold'])} 个 (task, stage) 桶的可检索量低于 "
            f"{cfg.G1_MIN_BUCKET_SIZE}，这些桶上的检索可能无命中"
        )
    if not leak["benchmark_episodes_with_object_category_present_in_index"] == 0:
        findings.append(
            f"{leak['benchmark_episodes_with_object_category_present_in_index']}/"
            f"{leak['benchmark_episode_total']} 个 benchmark episode 的物体类别在检索库中出现过；"
            "这是 house 级排除挡不住的语义覆盖，正式评测前需补充近重复核验"
        )
    if malformed:
        findings.append(
            f"{len(malformed)} 个 benchmark episode 的 instruction 命中畸形模式，建议隔离"
        )

    result = {
        "gate": "G1",
        "verdict": verdict,
        "findings": findings,
        "retrieval_bank": buckets,
        "leakage": leak,
        "malformed_benchmark_instructions": {
            "count": len(malformed),
            "examples": malformed[:5],
        },
        "house_level_exclusion": {
            "benchmark_houses": len(bench_houses),
            "index_houses": len(index_houses),
            "overlap_count": len(house_overlap),
            "overlap_houses": sorted(house_overlap)[:20],
            "exclusions_artifact_rule": exclusions.get("rule"),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"G1 判定：{verdict}")
    for f in findings:
        print(f"  - {f}")
    print(f"检索库总量：{buckets['total']}（task 分布 {buckets['by_task']}）")
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
