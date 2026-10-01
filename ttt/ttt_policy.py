"""MolmoSpaces policy/config subclass that adapts before the first action."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from olmo.eval.configure_molmo_spaces import (
    MolmoBotRBY1MultitaskPolicy,
    MolmoBotRBY1PickPnPEvalConfig,
    MolmoBotRBY1PickPnPPolicyConfig,
)

from ttt_data_validation.ttt.adaptation import ActionExpertTTT, TTTSettings
from ttt_data_validation.ttt.retrieval import SuccessRetriever
from ttt_data_validation.ttt.selected_data import SelectedDataError, selected_dataset


class EpisodicTTTPolicy(MolmoBotRBY1MultitaskPolicy):
    def __init__(self, config, task_type):
        super().__init__(config, task_type)
        pc = config.policy_config
        self.retriever = SuccessRetriever(top_k=pc.ttt_top_k)
        self.ttt = ActionExpertTTT(
            self.agent.model, self.agent.model_config,
            TTTSettings(steps=pc.ttt_steps, batch_size=pc.ttt_batch_size,
                        microbatch_size=1, lr=pc.ttt_lr, seed=pc.ttt_seed),
        )
        self._ttt_done = False
        self._event_path = Path(config.output_dir) / f"ttt_events_{os.getpid()}.jsonl"

    def _record(self, event: dict) -> None:
        event = {"utc": datetime.now(timezone.utc).isoformat(), **event}
        self._event_path.parent.mkdir(parents=True, exist_ok=True)
        with self._event_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def reset(self):
        super().reset()
        if hasattr(self, "ttt"):
            self.ttt.restore()
            self._ttt_done = False

    def _populate_action_buffer(self, observation) -> None:
        if not self._ttt_done:
            self._ttt_done = True
            obs = observation[0] if isinstance(observation, list) else observation
            instruction = obs.get("task") or self.task.get_task_description()
            start = time.perf_counter()
            result = self.retriever.retrieve(obs, instruction, task="pick_and_place")
            event = {
                "event": "episode_ttt", "instruction": instruction,
                "retrieval_sec": time.perf_counter() - start,
                "query_category": result.query_category,
                "candidate_count": result.candidate_count,
                "trajectory_ids": result.ids,
                "skip_reason": result.reason,
            }
            if result.rows:
                try:
                    with selected_dataset(list(result.rows), self.agent.model_config) as dataset:
                        event["selected_examples"] = len(dataset)
                        event["adaptation"] = self.ttt.adapt(dataset)
                except SelectedDataError as error:
                    self.ttt.restore()
                    event["skip_reason"] = f"selected_data_error:{error}"
                except Exception as error:
                    self.ttt.restore()
                    event["error"] = f"{type(error).__name__}:{error}"
                    self._record(event)
                    raise
            self._record(event)
        super()._populate_action_buffer(observation)


class EpisodicTTTPolicyConfig(MolmoBotRBY1PickPnPPolicyConfig):
    ttt_top_k: int = 16
    ttt_batch_size: int = 16
    ttt_steps: int = 20
    ttt_lr: float = 1e-5
    ttt_seed: int = 0

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        object.__setattr__(self, "policy_cls", EpisodicTTTPolicy)
        object.__setattr__(self, "policy_factory", EpisodicTTTPolicy)


class EpisodicTTTEvalConfig(MolmoBotRBY1PickPnPEvalConfig):
    policy_config: EpisodicTTTPolicyConfig = EpisodicTTTPolicyConfig()
    end_on_success: bool = True
