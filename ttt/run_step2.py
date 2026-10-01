#!/usr/bin/env python3
"""One-episode online TTT smoke, not a data-quality comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "MolmoBot/MolmoBot"
BENCHMARK = (ROOT / "molmospaces_data/assets/benchmarks/molmospaces-bench-v1/procthor-10k"
             / "RBY1PickAndPlaceDataGenConfig/RBY1PickAndPlaceDataGenConfig_20260209_json_benchmark")
CHECKPOINT = ROOT / "nas_wenyifan/models/MolmoBot-RBY1Multitask"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="6", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--episode-idx", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ttt_data_validation/artifacts/step2_smoke")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(PACKAGE), os.environ.get("PYTHONPATH", "")))
    sys.path[:0] = [str(ROOT), str(PACKAGE)]

    from molmo_spaces.evaluation.eval_main import run_evaluation
    from ttt_data_validation.ttt.ttt_policy import EpisodicTTTEvalConfig

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    result = run_evaluation(
        eval_config_cls=EpisodicTTTEvalConfig,
        benchmark_dir=BENCHMARK,
        checkpoint_path=CHECKPOINT,
        task_horizon_steps=400,
        output_dir=args.output_dir,
        num_workers=1,
        episode_idx=args.episode_idx,
        use_wandb=False,
    )
    summary = {
        "episode_idx": args.episode_idx,
        "gpu": args.gpu,
        "model_precision": "bf16_backbone_fp32_action_expert",
        "success_count": result.success_count,
        "total_count": result.total_count,
        "success_rate": result.success_rate,
        "elapsed_sec": time.perf_counter() - start,
        "output_dir": str(result.output_dir),
        "ttt_event_files": [str(p) for p in Path(result.output_dir).rglob("ttt_events_*.jsonl")],
    }
    output = args.output_dir / "step2_summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
