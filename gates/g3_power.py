#!/usr/bin/env python3
"""G3：统计功效与分层抽样核算（纯 CPU，零 GPU 成本）。

回答三个问题：
  1. 以历史基线 10.62%（24/226）为参照，要检出多大的效应需要多少 episode？
  2. 按物体类别/几何形态分层后，方差能降到多少？首版抽样应如何配比？
  3. 在给定 worker 数下，每个条件的墙钟时间是多少？

配对设计（同一批 episode 跑所有条件）比独立两组设计省样本，这里按配对口径估算。
只读，不修改任何文件。
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import gate_config as cfg


# ------------------------------------------------------------------ 基础工具
def normal_ppf(p: float) -> float:
    """标准正态分位数（Acklam 有理逼近，精度 ~1e-9，无需 scipy）。"""
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


Z_ALPHA = normal_ppf(1 - cfg.G3_ALPHA / 2)
Z_POWER = normal_ppf(cfg.G3_POWER)


def paired_sample_size(p0: float, p1: float, discordance: float = 0.5) -> int:
    """配对二值结局的样本量估算（McNemar 风格近似）。

    p0/p1 为两条件的成功率；discordance 为不一致对占比的猜测值。
    在不一致对中，检验力取决于 p1 相对 p0 的净增益方向。
    """
    if p1 <= p0:
        return -1
    # 需要的不一致对数（近似）：n_disc * (delta)^2 需覆盖 (z_a/2 + z_b)^2
    delta = p1 - p0
    n_disc_needed = ((Z_ALPHA + Z_POWER) ** 2) * p0 * (1 - p0) / (delta ** 2)
    if discordance <= 0:
        return -1
    return int(math.ceil(n_disc_needed / discordance))


def independent_sample_size(p0: float, p1: float) -> int:
    """独立两组设计的样本量（每组），用于对照说明配对设计的收益。"""
    if p1 <= p0:
        return -1
    p_bar = (p0 + p1) / 2
    numerator = (Z_ALPHA * math.sqrt(2 * p_bar * (1 - p_bar))
                 + Z_POWER * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return int(math.ceil(numerator / ((p1 - p0) ** 2)))


def wilson_ci(k: int, n: int, z: float = Z_ALPHA) -> tuple[float, float]:
    """Wilson 区间，小样本比正态近似稳。"""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ------------------------------------------------------------------ 功效表
def power_table() -> dict:
    rows = []
    for mult in cfg.G3_EFFECT_MULTIPLIERS:
        p1 = min(0.999, cfg.BASELINE_RATE * mult)
        paired = paired_sample_size(cfg.BASELINE_RATE, p1)
        indep = independent_sample_size(cfg.BASELINE_RATE, p1)
        rows.append(
            {
                "effect_multiplier": mult,
                "target_rate": round(p1, 4),
                "abs_gain_pp": round((p1 - cfg.BASELINE_RATE) * 100, 2),
                "paired_episodes_needed": paired,
                "independent_episodes_per_group": indep,
                "paired_efficiency_gain": (
                    round(indep / paired, 2) if paired > 0 and indep > 0 else None
                ),
            }
        )
    return {
        "baseline": {
            "success": cfg.BASELINE_SUCCESS,
            "total": cfg.BASELINE_TOTAL,
            "rate": round(cfg.BASELINE_RATE, 4),
            "wilson_ci_95": [round(x, 4) for x in wilson_ci(cfg.BASELINE_SUCCESS, cfg.BASELINE_TOTAL)],
        },
        "alpha": cfg.G3_ALPHA,
        "power": cfg.G3_POWER,
        "assumed_discordance": 0.5,
        "rows": rows,
        "note": (
            "配对设计假设不一致对占比 0.5。实际不一致对越少，所需 episode 越多。"
            "独立设计数字给出对照，说明为什么必须配对（同一批 episode 跑所有条件）。"
        ),
    }


# ------------------------------------------------------------------ 分层
def stratify(bench: list[dict]) -> dict:
    """按 E3 逐类成功率把 benchmark 分层，估算分层前后的方差。"""
    strata = {"high": [], "mid": [], "low": [], "unknown": []}
    malformed = []
    for episode in bench:
        text = ((episode.get("language") or {}).get("task_description") or "").strip()
        if not text or any(p in text.lower() for p in cfg.MALFORMED_INSTRUCTION_PATTERNS):
            malformed.append(episode["house_index"])
            continue
        cat = _bench_category(episode)
        rate = cfg.BASELINE_CATEGORY_RATES.get(cat)
        if rate is None:
            strata["unknown"].append(cat or "unknown")
        elif rate >= cfg.G3_STRATA_HIGH_RATE:
            strata["high"].append(cat)
        elif rate <= cfg.G3_STRATA_LOW_RATE:
            strata["low"].append(cat)
        else:
            strata["mid"].append(cat)

    # 分层内方差（伯努利）：p(1-p)，用各层基线的类别均值近似
    def layer_rate(names: list[str]) -> float | None:
        rates = [cfg.BASELINE_CATEGORY_RATES[c] for c in names if c in cfg.BASELINE_CATEGORY_RATES]
        return sum(rates) / len(rates) if rates else None

    layer_stats = {}
    for name in ("high", "mid", "low"):
        names = strata[name]
        rate = layer_rate(names)
        layer_stats[name] = {
            "episodes": len(names),
            "categories": sorted(set(names)),
            "category_count": len(set(names)),
            "approx_rate": round(rate, 4) if rate is not None else None,
            "bernoulli_variance": round(rate * (1 - rate), 4) if rate is not None else None,
        }

    all_cats = strata["high"] + strata["mid"] + strata["low"]
    overall = layer_rate(all_cats)
    return {
        "pooled_rate": round(overall, 4) if overall is not None else None,
        "pooled_variance": round(overall * (1 - overall), 4) if overall is not None else None,
        "layers": layer_stats,
        "unknown_category_episodes": len(strata["unknown"]),
        "isolated_malformed_episodes": len(malformed),
        "isolated_malformed_houses": sorted(set(malformed))[:20],
        "note": (
            "把 0% 类别与 71% 类别混在一起会放大方差。分层抽样可降方差，"
            "但分层方案必须与基线口径一致，否则成功率变化可能只反映类别配比变动。"
        ),
    }


def _bench_category(episode: dict) -> str:
    import re
    name = ((episode.get("task") or {}).get("pickup_obj_name")) or ""
    match = re.match(r"^([a-zA-Z]+?)_[0-9a-f]{32}", name)
    cat = (match.group(1) if match else name.split("_")[0]).lower()
    # E3 报告用 'potato'，benchmark 用 'irishpotato'，统一口径
    return {"irishpotato": "potato"}.get(cat, cat)


def budget_table() -> dict:
    """按候选规模 × worker 数给出墙钟时间。单 episode 参考 190 s。"""
    per_episode_sec = 190.0
    # TTT 条件的额外开销：每 episode 假设 3 轮 TTT，每轮 20 梯度步。
    # 若做冻结主干 feature 缓存，每轮主干前向 1 次 ≈ 单次推理的 1/10；
    # 这里保守按「每轮 TTT 额外 0.5 倍单步前向 × 20 步」粗估。
    ttt_overhead_ratio = 0.15  # 相对单 episode 时长的额外比例（含检索与梯度步）
    rows = []
    for size in cfg.G3_CANDIDATE_SIZES:
        for workers in (1, 2, 4, 8):
            base_hours = size * per_episode_sec / workers / 3600
            ttt_hours = base_hours * (1 + ttt_overhead_ratio)
            rows.append(
                {
                    "episodes": size,
                    "workers": workers,
                    "baseline_hours": round(base_hours, 2),
                    "with_ttt_hours": round(ttt_hours, 2),
                    "four_conditions_hours": round(base_hours + 3 * ttt_hours, 2),
                }
            )
    return {
        "per_episode_sec_reference": per_episode_sec,
        "ttt_overhead_ratio_assumed": ttt_overhead_ratio,
        "rows": rows,
        "conditions": ["no_ttt", "random_ttt", "retrieved_ttt", "shuffled_instruction_ttt"],
        "note": (
            "190 s/episode 来自历史基线实测（场景重建 15–60 s + rollout 2:00–3:10 + 保存 10–40 s）。"
            "TTT 开销比例是粗估，需在 G4 冒烟后用实测替换。"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="G3：统计功效与分层核算")
    parser.add_argument("--out", type=Path, default=cfg.ARTIFACT_DIR / "g3_power.json")
    args = parser.parse_args()

    bench = json.loads(cfg.BENCHMARK_JSON.read_text(encoding="utf-8"))
    power = power_table()
    strata = stratify(bench)
    budget = budget_table()

    # 判定：首版推荐规模 = 能检出 2 倍效应的配对样本量，向上取到候选规模
    target = next(
        (r for r in power["rows"] if r["effect_multiplier"] == 2.0), None
    )
    recommended = None
    if target and target["paired_episodes_needed"] > 0:
        needed = target["paired_episodes_needed"]
        recommended = min(
            [s for s in cfg.G3_CANDIDATE_SIZES if s >= needed]
            or [max(cfg.G3_CANDIDATE_SIZES)]
        )

    findings = []
    verdict = cfg.VERDICT_PASS
    if target:
        findings.append(
            f"检出 {target['effect_multiplier']}× 效应（{cfg.BASELINE_RATE:.2%} → "
            f"{target['target_rate']:.2%}）在配对设计下需 {target['paired_episodes_needed']} episode"
        )
    findings.append(
        f"现有历史基线 {cfg.BASELINE_TOTAL} episode 的 95% CI 为 "
        f"[{power['baseline']['wilson_ci_95'][0]:.2%}, {power['baseline']['wilson_ci_95'][1]:.2%}]，"
        f"宽度约 {(power['baseline']['wilson_ci_95'][1]-power['baseline']['wilson_ci_95'][0])*100:.1f} 个百分点"
    )
    if strata["isolated_malformed_episodes"]:
        findings.append(
            f"{strata['isolated_malformed_episodes']} 个 episode 因畸形/空 instruction 隔离"
        )
    if strata["unknown_category_episodes"]:
        if verdict == cfg.VERDICT_PASS:
            verdict = cfg.VERDICT_WARN
        findings.append(
            f"{strata['unknown_category_episodes']} 个 episode 的物体类别无历史基线，分层时归入 unknown"
        )

    result = {
        "gate": "G3",
        "verdict": verdict,
        "findings": findings,
        "recommended_episodes": recommended,
        "power": power,
        "stratification": strata,
        "budget": budget,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"G3 判定：{verdict}")
    for f in findings:
        print(f"  - {f}")
    print(f"推荐首版规模：{recommended} episode")
    print("功效表：")
    for r in power["rows"]:
        print(
            f"  {r['effect_multiplier']}× → {r['target_rate']:.2%} "
            f"(+{r['abs_gain_pp']}pp)  配对需 {r['paired_episodes_needed']} ep，"
            f"独立需 {r['independent_episodes_per_group']} ep/组"
        )
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
