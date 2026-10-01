#!/usr/bin/env python3
"""G4：单 episode 端到端冒烟测试（唯一需要 GPU 的闸门）。

验证链路：benchmark JSON → 场景实例化 → reset → 模型前向 → step → success 判定。

三条硬约束（少了任何一条，结论都不可用）：
  1. end_on_success 必须为 True。默认 False 时 judge_success() 只在循环结束后
     用末态判定一次，过程中已成功会被判失败（历史 E2 run 因此低估 2.1 倍）。
  2. 只用本地存在的场景资产，避免撞上缺失 house。
  3. 单 episode，带超时。参考耗时 ~190 s，超时阈值留 4.7 倍余量。

本脚本不内联任何路径常量，全部来自 gate_config.py。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import gate_config as cfg

# end_on_success 通过本地 config 子类注入（见 g4_eval_configs.py），
# 不改 MolmoBot 源码，也不用 PYTHONPATH 黑魔法。
# 子类继承父类的 policy_config 默认值，因此 run_eval.py 的
# isinstance(policy_config, SynthVLAPolicyConfig) 校验照样通过。
G4_EVAL_CONFIG_CLS = "g4_eval_configs:G4MolmoBotRBY1PickPnPEvalConfig"


def resolve_target_episode(bench_path: Path) -> dict:
    """定位 G4 的目标 episode：house_index + idx（benchmark 全局下标）。"""
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    for idx, episode in enumerate(bench):
        if int(episode["house_index"]) == cfg.G4_HOUSE_INDEX:
            return {
                "idx": idx,
                "house_index": int(episode["house_index"]),
                "task_description": (episode.get("language") or {}).get("task_description"),
                "pickup_obj_name": (episode.get("task") or {}).get("pickup_obj_name"),
                "episode_length": (episode.get("source") or {}).get("episode_length"),
            }
    raise SystemExit(
        f"benchmark 中找不到 house_index={cfg.G4_HOUSE_INDEX} 的 episode"
    )


def check_scene_assets(house_index: int) -> dict:
    """确认目标 house 的场景文件本地存在。"""
    scenes = cfg.SCENES_DIR
    required = {
        "json": scenes / f"val_{house_index}.json",
        "xml": scenes / f"val_{house_index}.xml",
        "ceiling_xml": scenes / f"val_{house_index}_ceiling.xml",
        "assets_dir": scenes / f"val_{house_index}_assets",
    }
    status = {k: p.exists() for k, p in required.items()}
    return {
        "scene_dir": str(scenes),
        "house_index": house_index,
        "files": {k: {"path": str(v), "exists": status[k]} for k, v in required.items()},
        "all_present": all(status.values()),
        "known_missing_in_snapshot": house_index in cfg.G4_KNOWN_MISSING_HOUSES,
    }


def build_command(target: dict, output_dir: Path) -> list[str]:
    return [
        str(cfg.MOLMOBOT_PYTHON),
        str(cfg.RUN_EVAL_SCRIPT),
        "--benchmark_path",
        str(cfg.BENCHMARK_DIR),
        "--eval_config_cls",
        G4_EVAL_CONFIG_CLS,
        "--checkpoint_path",
        str(cfg.CHECKPOINT_DIR),
        "--task_horizon",
        str(cfg.G4_TASK_HORIZON),
        "--num_workers",
        str(cfg.G4_NUM_WORKERS),
        "--episode_idx",
        str(target["idx"]),
        "--output_dir",
        str(output_dir),
    ]


def parse_outcome(output_dir: Path, stdout: str) -> dict:
    """从输出目录与 stdout 提取成功数与总数。

    优先读 stdout 的 Success count / Success rate 行（与历史基线同口径）；
    其次回落到输出目录里的 log 文件。
    """
    pattern_count = re.compile(r"Success count:\s*(\d+),\s*Total count:\s*(\d+)")
    pattern_rate = re.compile(r"Success rate:\s*([\d.]+)%")

    match_count = pattern_count.search(stdout)
    match_rate = pattern_rate.search(stdout)

    logs = sorted(output_dir.rglob("*.log")) if output_dir.exists() else []
    for log in logs:
        text = log.read_text(encoding="utf-8", errors="replace")
        if not match_count:
            match_count = pattern_count.search(text)
        if not match_rate:
            match_rate = pattern_rate.search(text)

    return {
        "success_count": int(match_count.group(1)) if match_count else None,
        "total_count": int(match_count.group(2)) if match_count else None,
        "success_rate": float(match_rate.group(1)) if match_rate else None,
        "log_files": [str(p) for p in logs],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="G4：单 episode 端到端冒烟测试")
    parser.add_argument("--out", type=Path, default=cfg.ARTIFACT_DIR / "g4_smoke.json")
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="只做静态检查（场景资产、目标 episode、命令拼装），不启动仿真",
    )
    parser.add_argument(
        "--house-index",
        type=int,
        default=cfg.G4_HOUSE_INDEX,
        help="覆盖 gate_config 里的目标 house",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=cfg.G4_TIMEOUT_SEC,
        help="wrapped 命令的超时秒数",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="覆盖 gate_config 里的 G4_GPU（CUDA_VISIBLE_DEVICES）",
    )
    args = parser.parse_args()

    # 允许 CLI 覆盖目标 house（便于换一个更可能通过的 episode 重试）
    if args.house_index != cfg.G4_HOUSE_INDEX:
        cfg.G4_HOUSE_INDEX = args.house_index

    target = resolve_target_episode(cfg.BENCHMARK_JSON)
    scene = check_scene_assets(target["house_index"])

    output_dir = cfg.SMOKE_OUTPUT_DIR
    command = build_command(target, output_dir)

    result = {
        "gate": "G4",
        "target_episode": target,
        "scene_assets": scene,
        "command": command,
        "eval_config_cls": G4_EVAL_CONFIG_CLS,
        "timeout_sec": args.timeout,
    }

    findings = []
    if not scene["all_present"]:
        missing = [k for k, v in scene["files"].items() if not v["exists"]]
        findings.append(f"目标 house 场景文件缺失：{missing}，无法实例化")

    if args.detect_only or not scene["all_present"]:
        result["mode"] = "detect_only" if args.detect_only else "blocked"
        result["verdict"] = cfg.VERDICT_PASS if scene["all_present"] else cfg.VERDICT_FAIL
        result["findings"] = findings
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"G4 判定：{result['verdict']}（模式：{result['mode']}）")
        for f in findings:
            print(f"  - {f}")
        print(f"目标 episode：idx={target['idx']} house={target['house_index']}")
        print(f"指令：{target['task_description']}")
        if result["verdict"] == cfg.VERDICT_PASS:
            print("场景资产齐备，可实际执行（去掉 --detect-only）")
        print(f"写入 {args.out}")
        return

    # 实跑：cwd 必须是 MolmoBot 根（run_eval.py 依赖 cwd 相对路径），
    # PYTHONPATH 需含 gates 目录，好让 g4_eval_configs 可被 import。
    molmobot_root = cfg.ROOT / "MolmoBot/MolmoBot"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(cfg.GATE_DIR), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    # mujoco 渲染与 CUDA 选择沿用历史基线的环境（见 E3 的 .bash_history）
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("JAX_PLATFORMS", "cpu")
    if cfg.G4_GPU is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu if args.gpu else cfg.G4_GPU)

    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=str(molmobot_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        elapsed = time.time() - started
        stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - started
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        returncode = None
        timed_out = True

    outcome = parse_outcome(output_dir, stdout)

    if timed_out:
        verdict = cfg.VERDICT_FAIL
        findings.append(f"超时（{args.timeout}s），链路未走通")
    elif returncode != 0:
        verdict = cfg.VERDICT_FAIL
        findings.append(f"返回码 {returncode}，链路失败")
    elif outcome["total_count"] is None:
        verdict = cfg.VERDICT_WARN
        findings.append("命令正常结束但未解析到 Success count 行，需人工确认输出目录")
    else:
        verdict = cfg.VERDICT_PASS
        findings.append(
            f"链路走通：{outcome['success_count']}/{outcome['total_count']} 成功，"
            f"耗时 {elapsed:.0f}s"
        )

    result.update(
        {
            "mode": "executed",
            "verdict": verdict,
            "findings": findings,
            "elapsed_sec": round(elapsed, 1),
            "timed_out": timed_out,
            "returncode": returncode,
            "outcome": outcome,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"G4 判定：{verdict}")
    for f in findings:
        print(f"  - {f}")
    print(f"耗时 {elapsed:.0f}s，写入 {args.out}")


if __name__ == "__main__":
    main()
