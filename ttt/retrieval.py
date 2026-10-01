"""Deterministic, metadata-only retrieval over the audited partial snapshot."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


INDEX_PATH = Path(__file__).resolve().parents[1] / "artifacts/success_index.jsonl"
_WORDS = re.compile(r"[a-z0-9]+")
_OBJECT_ID = re.compile(r"^([a-zA-Z]+?)_[0-9a-f]{32}")
_STOP = {"a", "an", "the", "it", "and", "in", "on", "or", "to", "with", "up", "pick", "place"}


def trajectory_id(row: dict) -> str:
    return f"{row['dataset']}:{row['entry_index']}:{row['h5_member']}:{row['traj_key']}"


def object_category(object_id: str | None) -> str | None:
    if not object_id:
        return None
    match = _OBJECT_ID.match(object_id)
    return (match.group(1) if match else object_id.split("_")[0]).lower()


def words(text: str) -> set[str]:
    return set(_WORDS.findall(text.lower())) - _STOP


def instruction_object(instruction: str, categories: set[str]) -> str | None:
    # The pickup clause, not the placement clause, identifies the object.
    pickup = re.split(r"\b(?:and\s+)?place\b", instruction.lower(), maxsplit=1)[0]
    tokens = _WORDS.findall(pickup)
    for category in sorted(categories, key=lambda x: (-len(x), x)):
        if not category:
            continue
        # "remotecontrol" in IDs versus "remote control" in language.
        for width in (1, 2, 3):
            if any("".join(tokens[i:i + width]) in {category, category + "s", category + "es"}
                   for i in range(max(0, len(tokens) - width + 1))):
                return category
    return None


@dataclass(frozen=True)
class RetrievalResult:
    rows: tuple[dict, ...]
    reason: str | None
    query_category: str | None
    candidate_count: int

    @property
    def ids(self) -> list[str]:
        return [trajectory_id(row) for row in self.rows]


class SuccessRetriever:
    def __init__(self, index_path: Path = INDEX_PATH, top_k: int = 16):
        self.top_k = top_k
        self.rows: list[dict] = []
        with Path(index_path).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("eligible") is not True or row.get("terminal_success") is not True:
                    continue
                if row.get("benchmark_house_overlap") is not False:
                    continue
                if row.get("stage_map", {}).get("pregrasp") not in row.get("observed_phase_ids", []):
                    continue
                self.rows.append(row)
        self.categories = {cat for row in self.rows if (cat := object_category(row.get("object_id")))}

    def retrieve(self, initial_obs: dict, instruction: str, task: str | None = None) -> RetrievalResult:
        if not isinstance(initial_obs, dict):
            return RetrievalResult((), "missing_initial_observation", None, 0)
        if not instruction or not instruction.strip():
            return RetrievalResult((), "missing_instruction", None, 0)
        if task is None:
            task = "pick_and_place" if re.search(r"\bplace\b", instruction.lower()) else "pick"
        category = instruction_object(instruction, self.categories)
        if category is None:
            return RetrievalResult((), "unknown_object_category", None, 0)
        candidates = [r for r in self.rows if r.get("task") == task and object_category(r.get("object_id")) == category]
        if not candidates:
            return RetrievalResult((), "no_matching_success_trajectory", category, 0)
        query = words(instruction)

        def rank(row: dict) -> tuple:
            target = words(row.get("instruction", ""))
            union = len(query | target)
            score = len(query & target) / union if union else 0.0
            return (-score, trajectory_id(row))

        ranked = sorted(candidates, key=rank)
        return RetrievalResult(tuple(ranked[: self.top_k]), None, category, len(candidates))
