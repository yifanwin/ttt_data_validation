#!/usr/bin/env python3
"""G2：评测口径对齐检查（纯 CPU 可完成的部分 + 一份需实跑核对的清单）。

检查五组口径，任何一项不一致都会让 TTT 的收益被系统性掩盖或误读：
  1. 维度口径：action 20 / state 22 / horizon 16 / execute 8
  2. 归一化口径：checkpoint stats 的往返误差，尤其 ±100 量纲维
  3. 相机口径：离线 clip 的相机 vs benchmark 记录的相机（分辨率/相机名/数量）
  4. 判据口径：end_on_success 必须为 True，否则瞬时成功被末态覆盖
  5. 指令口径：检索目标 instruction 与在线 observation 的 task_description 来源

只读，不修改任何文件。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

import gate_config as cfg


def check_dimensions(ckpt: dict, contract: dict) -> dict:
    model = ckpt.get("model", {})
    observed = {
        "action_dim": model.get("action_dim"),
        "action_horizon": model.get("action_horizon"),
        "n_action_steps": model.get("n_action_steps"),
        "n_obs_steps": model.get("n_obs_steps"),
    }
    expected = {
        "action_dim": cfg.EXPECT_ACTION_DIM,
        "action_horizon": cfg.EXPECT_ACTION_HORIZON,
        "n_action_steps": cfg.EXPECT_EXECUTE_HORIZON,
    }
    mismatches = {
        k: {"expected": v, "actual": observed.get(k)}
        for k, v in expected.items()
        if observed.get(k) != v
    }

    # 步骤一实测的 batch 形状
    shapes = [
        {
            "dataset": s["dataset"],
            "batch_action_shape": s["batch_action_shape"],
            "batch_state_shape": s["batch_state_shape"],
        }
        for s in contract.get("samples", [])
    ]
    shape_mismatches = [
        s
        for s in shapes
        if s["batch_action_shape"] != [1, cfg.EXPECT_ACTION_HORIZON, cfg.EXPECT_ACTION_DIM]
        or s["batch_state_shape"] != [1, cfg.EXPECT_STATE_DIM]
    ]

    # horizon 与执行步数不等，是 TTT 目标与 rollout 口径的核心错配点
    horizon_gap = (observed.get("action_horizon"), observed.get("n_action_steps"))
    if observed.get("action_horizon") != observed.get("n_action_steps"):
        note = (
            f"训练 chunk={horizon_gap[0]} 步，rollout 每 chunk 只执行 {horizon_gap[1]} 步。"
            "TTT 的 loss 若覆盖全部 16 步，会把容量花在被丢弃的后 8 步上。"
            "建议首版显式对齐到 8 步（加权或 mask 后半段）。"
        )
    else:
        note = "训练 chunk 与执行步数一致，无需对齐。"
    return {
        "observed": observed,
        "mismatches": mismatches,
        "sample_shape_mismatches": shape_mismatches,
        "horizon_alignment_note": note,
    }


def check_normalization(ckpt: dict, contract: dict) -> dict:
    model = ckpt.get("model", {})
    pre = (model.get("robot_preprocessor") or {}).get("stats_by_repo", {}).get(cfg.NORM_REPO_ID, {})
    action_stats = pre.get("action", {})
    q01 = action_stats.get("q01") or []
    q99 = action_stats.get("q99") or []

    # 找出量纲异常的维度：|q| 远超正常关节量级
    outlier_dims = []
    for i, (lo, hi) in enumerate(zip(q01, q99)):
        if abs(lo) > 1.0 or abs(hi) > 1.0:
            span = hi - lo if hi > lo else 0.0
            outlier_dims.append(
                {
                    "dim": i,
                    "q01": lo,
                    "q99": hi,
                    "span": span,
                    "relative_weight": (1.0 / span) if span > 0 else None,
                }
            )

    # 正常维度被 clip 的量级
    normal_spans = [
        hi - lo
        for lo, hi in zip(q01, q99)
        if abs(lo) <= 1.0 and abs(hi) <= 1.0 and hi > lo
    ]
    median_normal_span = sorted(normal_spans)[len(normal_spans) // 2] if normal_spans else None

    # 权重比：异常维的逆尺度 与 正常维中位数逆尺度 之比
    weight_ratio = None
    if outlier_dims and median_normal_span and median_normal_span > 0:
        outlier_span = outlier_dims[0]["span"]
        if outlier_span > 0:
            weight_ratio = median_normal_span / outlier_span

    roundtrips = [
        {
            "dataset": s["dataset"],
            "action_roundtrip_max_abs_error": s["action_roundtrip_max_abs_error"],
            "state_roundtrip_max_abs_error": s["state_roundtrip_max_abs_error"],
            "ok": s["action_roundtrip_max_abs_error"] <= cfg.G2_NORM_ATOL
            and s["state_roundtrip_max_abs_error"] <= cfg.G2_NORM_ATOL,
        }
        for s in contract.get("samples", [])
    ]

    return {
        "action_norm_mode": (model.get("robot_preprocessor") or {}).get("action_norm_mode"),
        "state_norm_mode": (model.get("robot_preprocessor") or {}).get("state_norm_mode"),
        "quantile_outlier_dims": outlier_dims,
        "median_normal_span": median_normal_span,
        "gradient_weight_ratio_outlier_vs_normal": weight_ratio,
        "roundtrip_checks": roundtrips,
        "note": (
            "默认 MSE 会让量纲异常的维度在 loss 中的有效权重与正常关节维相差数个数量级；"
            "quantiles 模式还会把 OOD 动作 clip 到 ±1，静默截断梯度。"
            "建议至少把逐维 loss 分量记入日志。"
        ),
    }


def check_cameras(ckpt: dict, bench: list[dict], contract: dict) -> dict:
    model = ckpt.get("model", {})
    image_cfg = (model.get("mm_preprocessor") or {}).get("image", {})
    offline_shapes = sorted(
        {tuple(s["camera_shapes"][0]) for s in contract.get("samples", [])}
    )
    offline_preproc_shapes = sorted(
        {
            (s["sample_metadata"]["image_group_size"][0][1], s["sample_metadata"]["image_group_size"][0][0])
            for s in contract.get("samples", [])
        }
    )
    # benchmark 记录的相机
    bench_cam_names = sorted({c["name"] for e in bench for c in e.get("cameras", [])})
    bench_res = sorted({tuple(e.get("img_resolution", [])) for e in bench})
    depth_flags = sorted({bool(c.get("record_depth")) for e in bench for c in e.get("cameras", [])})

    offline_names = ["wrist_camera_r", "head_camera", "wrist_camera_l"]
    name_overlap = sorted(set(bench_cam_names) & set(offline_names))

    return {
        "checkpoint_image_config": {
            "max_images": image_cfg.get("max_images"),
            "max_crops": image_cfg.get("max_crops"),
            "crop_mode": image_cfg.get("crop_mode"),
        },
        "offline_camera_names": offline_names,
        "offline_frame_shapes": [list(s) for s in offline_shapes],
        "benchmark_camera_names": bench_cam_names,
        "benchmark_img_resolution": [list(r) for r in bench_res],
        "benchmark_record_depth": depth_flags,
        "shared_camera_names": name_overlap,
        "camera_name_mismatch": sorted(set(bench_cam_names) - set(offline_names)),
        "resolution_mismatch": len(bench_res) == 1
        and offline_shapes
        and list(bench_res[0]) != ["640", "480"]
        and tuple(bench_res[0]) != tuple(offline_shapes[0]),
        "note": (
            "benchmark JSON 记录的相机（640×480、record_depth=false）会覆盖 config 的相机设置；"
            "MolmoBot 训练用 RBY1GoProD455CameraSystem 1024×576。两地口径若不一致，"
            "TTT 学到的动作映射会建立在与 rollout 不同的视觉输入上，"
            "表现为『检索正确但动作不兼容』。执行前必须实跑核对一次。"
        ),
    }


def check_success_criteria() -> dict:
    """成功判据口径：end_on_success 必须为 True。"""
    return {
        "default_end_on_success": False,
        "default_defined_at": "molmo_spaces/configs/abstract_exp_config.py:53",
        "required_value": cfg.REQUIRED_END_ON_SUCCESS,
        "why": (
            "end_on_success=False 时 rollout 跑满 task_horizon，judge_success() 只在循环结束后"
            "用末态判定一次，过程中曾经成功的会被判为失败。同 benchmark 的 E2 run 因此低估 2.1 倍"
            "（2.67% vs 5.61%）。TTT 最可能的收益形式是『更早成功』，用错这个开关会把收益方向掩盖。"
        ),
        "verdict": cfg.VERDICT_FAIL,
    }


def check_instruction_source(bench: list[dict]) -> dict:
    texts = [
        ((e.get("language") or {}).get("task_description") or "") for e in bench
    ]
    empty = sum(1 for t in texts if not t.strip())
    malformed = []
    for t in texts:
        low = t.lower()
        if any(p in low for p in cfg.MALFORMED_INSTRUCTION_PATTERNS):
            malformed.append(t)
    return {
        "online_instruction_field": "benchmark language.task_description（经 obs['task'] 传入策略）",
        "offline_instruction_field": "obs_scene.task_description",
        "benchmark_empty_instructions": empty,
        "malformed_count": len(malformed),
        "malformed_examples": malformed[:5],
        "note": (
            "检索以 instruction 为键，畸形 instruction 会导致检索无命中或命中错误 clip，"
            "会被误读为该批数据覆盖不足。建议隔离这些 episode 并记录数量。"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="G2：评测口径对齐检查")
    parser.add_argument("--out", type=Path, default=cfg.ARTIFACT_DIR / "g2_alignment.json")
    args = parser.parse_args()

    ckpt = yaml.safe_load(cfg.CHECKPOINT_CONFIG.read_text(encoding="utf-8"))
    contract = json.loads(
        (cfg.ROOT / "ttt_data_validation/artifacts/sample_contract.json").read_text(
            encoding="utf-8"
        )
    )
    bench = json.loads(cfg.BENCHMARK_JSON.read_text(encoding="utf-8"))

    dims = check_dimensions(ckpt, contract)
    norm = check_normalization(ckpt, contract)
    cams = check_cameras(ckpt, bench, contract)
    succ = check_success_criteria()
    instr = check_instruction_source(bench)

    findings = []
    verdict = cfg.VERDICT_PASS
    if dims["mismatches"] or dims["sample_shape_mismatches"]:
        verdict = cfg.VERDICT_FAIL
        findings.append(f"维度口径不一致：{dims['mismatches'] or dims['sample_shape_mismatches']}")
    if any(not r["ok"] for r in norm["roundtrip_checks"]):
        verdict = cfg.VERDICT_FAIL
        findings.append("归一化往返误差超出阈值")
    if norm["quantile_outlier_dims"]:
        if verdict == cfg.VERDICT_PASS:
            verdict = cfg.VERDICT_WARN
        findings.append(
            f"action 归一化有 {len(norm['quantile_outlier_dims'])} 个量纲异常维"
            f"（dim {[d['dim'] for d in norm['quantile_outlier_dims']]}），"
            f"与正常维的梯度权重比约 {norm['gradient_weight_ratio_outlier_vs_normal']:.1e}"
        )
    findings.append(
        f"horizon 口径：{dims['horizon_alignment_note']}"
    )
    if cams["camera_name_mismatch"]:
        if verdict == cfg.VERDICT_PASS:
            verdict = cfg.VERDICT_WARN
        findings.append(
            f"benchmark 相机名与离线相机名不一致：{cams['camera_name_mismatch']}"
        )
    findings.append(
        f"判据口径：end_on_success 默认 False，必须显式设为 {cfg.REQUIRED_END_ON_SUCCESS}"
    )
    if instr["malformed_count"]:
        findings.append(f"{instr['malformed_count']} 条畸形 instruction 需隔离")

    result = {
        "gate": "G2",
        "verdict": verdict,
        "findings": findings,
        "dimensions": dims,
        "normalization": norm,
        "cameras": cams,
        "success_criteria": succ,
        "instruction_source": instr,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"G2 判定：{verdict}")
    for f in findings:
        print(f"  - {f}")
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
