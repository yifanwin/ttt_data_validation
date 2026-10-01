"""Step-2 unit and partial-snapshot integration checks."""

from __future__ import annotations

import copy
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ttt_data_validation.ttt.adaptation import ActionExpertTTT, TTTSettings
from ttt_data_validation.ttt.retrieval import SuccessRetriever
from ttt_data_validation.ttt.selected_data import SelectedDataError, _member_bytes, materialize, selected_dataset


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = SuccessRetriever()

    def test_egg_is_reproducible_and_leak_free(self):
        query = "pick up the brown egg and place it on the bowl"
        first = self.retriever.retrieve({}, query)
        second = self.retriever.retrieve({}, query)
        self.assertEqual(first.ids, second.ids)
        self.assertEqual(len(first.ids), 16)
        self.assertEqual(len(first.ids), len(set(first.ids)))
        self.assertTrue(all(not r["benchmark_house_overlap"] for r in first.rows))
        self.assertTrue(all(r["task"] == "pick_and_place" for r in first.rows))

    def test_skip_reasons(self):
        self.assertEqual(self.retriever.retrieve({}, "").reason, "missing_instruction")
        self.assertEqual(self.retriever.retrieve({}, "pick up the quasar").reason, "unknown_object_category")
        self.assertEqual(self.retriever.retrieve(None, "pick up egg").reason, "missing_initial_observation")


class SelectedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from olmo.models.model_config import BaseModelConfig
        cls.config = BaseModelConfig.load(
            Path(__file__).resolve().parents[2] / "nas_wenyifan/models/MolmoBot-RBY1Multitask/config.yaml",
            key="model", validate_paths=False,
        )
        cls.retriever = SuccessRetriever()

    def test_pick_and_pnp_sample_contract(self):
        for query in ("pick up the brown egg and place it on the bowl", "pick up the brown egg"):
            with self.subTest(query=query):
                result = self.retriever.retrieve({}, query)
                self.assertTrue(result.rows)
                with selected_dataset([result.rows[0]], self.config) as dataset:
                    example = dataset.get(0, np.random.default_rng(0))
                    self.assertEqual(example["action"].shape, (16, 20))
                    self.assertEqual(example["state"].shape, (22,))
                    self.assertEqual(len(example["image"]), 3)

    def test_missing_shard_and_hash_fail_clearly(self):
        row = self.retriever.retrieve({}, "pick up the brown egg and place it on the bowl").rows[0]
        with tempfile.TemporaryDirectory() as folder:
            missing = copy.deepcopy(row)
            missing["shard_id"] = 99999
            with self.assertRaisesRegex(SelectedDataError, "missing shard"):
                materialize([missing], Path(folder))
            corrupt = copy.deepcopy(row)
            corrupt["h5_sha256"] = "0" * 64
            with self.assertRaisesRegex(SelectedDataError, "hash mismatch"):
                materialize([corrupt], Path(folder))

    def test_missing_video_member_fails_clearly(self):
        with tarfile.open(fileobj=io.BytesIO(), mode="w") as archive:
            with self.assertRaisesRegex(SelectedDataError, "missing archive member"):
                _member_bytes(archive, "house_1/episode_00000001_head_camera_batch_1.mp4")

    def test_same_house_different_entries_are_isolated(self):
        rows = self.retriever.rows
        grouped: dict[tuple, dict[int, dict]] = {}
        pair = None
        for row in rows:
            key = (row["dataset"], row["house_index"])
            by_entry = grouped.setdefault(key, {})
            by_entry[row["entry_index"]] = row
            if len(by_entry) >= 2:
                pair = list(by_entry.values())[:2]
                break
        self.assertIsNotNone(pair)
        with tempfile.TemporaryDirectory() as folder:
            index = materialize(pair, Path(folder))
            self.assertEqual(len(index), 2)
            self.assertEqual(len({p for value in index.values() for p in value}), 2)


class _FakeDataset:
    def __len__(self):
        return 2

    def get(self, idx, rng):
        return {"sample_id": idx}


class _FakeConfig:
    def build_preprocessor(self, **kwargs):
        class Processor:
            def __call__(self, example):
                return example

            def get_output_shapes(self):
                return {}
        return Processor()

    def build_collator(self, *args, **kwargs):
        def collate(examples):
            return {
                "input_ids": torch.ones(1, 1),
                "actions": torch.ones(1, 16, 20),
                "states": torch.ones(1, 22),
            }
        return collate


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(1, 1)
        self.action_expert = torch.nn.Linear(1, 1)

    def forward(self, input_ids, actions, states):
        prediction = self.action_expert(self.backbone(input_ids))
        loss = (prediction - actions.mean()).square().mean()
        metrics = {f"flow_loss_dim_{i}": loss.detach() for i in range(20)}
        return SimpleNamespace(internal={"action_flow_loss": loss}, metrics=metrics)


class AdaptationTests(unittest.TestCase):
    def test_update_freeze_and_exact_restore(self):
        model = _FakeModel()
        backbone_before = [p.detach().clone() for p in model.backbone.parameters()]
        trainer = ActionExpertTTT(model, _FakeConfig(), TTTSettings(steps=2, batch_size=2))
        expert_before = [p.detach().clone() for p in model.action_expert.parameters()]
        stats = trainer.adapt(_FakeDataset())
        self.assertEqual(len(stats["history"]), 2)
        self.assertIsNone(trainer.optimizer)
        self.assertTrue(all(p.grad is None for p in model.action_expert.parameters()))
        self.assertGreater(stats["trainable_elements_changed"], 0)
        self.assertGreater(stats["max_abs_parameter_delta"], 0.0)
        self.assertTrue(all(np.isfinite(x["loss"]) and np.isfinite(x["grad_norm"]) for x in stats["history"]))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(expert_before, model.action_expert.parameters())))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(backbone_before, model.backbone.parameters())))
        trainer.restore()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expert_before, model.action_expert.parameters())))
        self.assertIsNone(trainer.optimizer)


if __name__ == "__main__":
    unittest.main()
