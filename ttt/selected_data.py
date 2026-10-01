"""Materialize only retrieved audited trajectories for MolmoBot's original loader."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import zstandard as zstd

from olmo.data.synthmanip_config import SynthmanipDatasetConfig
from olmo.data.synthmanip_dataset import SynthmanipDataset
from olmo.data.synthmanip_presets import (
    ACTION_DATASET_KEYS, ACTION_SPECS, CAMERA_PRESETS, STATE_INDICES, STATE_SPECS,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "nas_wenyifan/molmospaces_data"
CAMERAS = CAMERA_PRESETS["RBY1_full_with_head_gopro"]


class SelectedDataError(RuntimeError):
    pass


def _member_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    try:
        member = archive.getmember(name)
    except KeyError as error:
        raise SelectedDataError(f"missing archive member: {name}") from error
    if not member.isfile():
        raise SelectedDataError(f"not a file: {name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise SelectedDataError(f"unreadable archive member: {name}")
    return stream.read()


def _validate_row(row: dict) -> None:
    if row.get("eligible") is not True or row.get("terminal_success") is not True:
        raise SelectedDataError("selected row is not audited successful data")
    if row.get("benchmark_house_overlap") is not False:
        raise SelectedDataError("selected row overlaps benchmark house")
    if not re.fullmatch(r"traj_\d+", row["traj_key"]):
        raise SelectedDataError("invalid trajectory key")
    if Path(row["h5_member"]).is_absolute() or ".." in Path(row["h5_member"]).parts:
        raise SelectedDataError("unsafe HDF5 member path")


def materialize(rows: list[dict], root: Path) -> dict:
    """Return valid_trajectory_index; never modifies source HDF5 or shards."""
    if not rows:
        raise SelectedDataError("no selected rows")
    selected: dict[tuple, list[dict]] = {}
    for row in rows:
        _validate_row(row)
        key = (row["dataset"], row["entry_index"])
        selected.setdefault(key, []).append(row)
    index: dict[str, dict] = {}
    for (dataset, entry_index), group in selected.items():
        first = group[0]
        fields = ("shard_id", "offset", "size", "part", "archive_path")
        if any(any(row[k] != first[k] for k in fields) for row in group):
            raise SelectedDataError("inconsistent archive locator")
        shard = DATA_ROOT / dataset / "shards" / f"{first['shard_id']:05d}.tar"
        if not shard.is_file():
            raise SelectedDataError(f"missing shard: {shard}")
        if shard.stat().st_size < first["offset"] + first["size"]:
            raise SelectedDataError(f"truncated shard: {shard}")
        with shard.open("rb") as stream:
            stream.seek(first["offset"])
            packed = stream.read(first["size"])
        if len(packed) != first["size"]:
            raise SelectedDataError(f"short read: {shard}")
        try:
            with zstd.ZstdDecompressor().stream_reader(io.BytesIO(packed)) as reader:
                payload = reader.read()
        except zstd.ZstdError as error:
            raise SelectedDataError(f"archive decode failed: {shard}") from error
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            by_h5: dict[str, list[dict]] = {}
            for row in group:
                by_h5.setdefault(row["h5_member"], []).append(row)
            for h5_member, h5_rows in by_h5.items():
                h5_bytes = _member_bytes(archive, h5_member)
                digest = hashlib.sha256(h5_bytes).hexdigest()
                if any(digest != row["h5_sha256"] for row in h5_rows):
                    raise SelectedDataError(f"HDF5 hash mismatch: {h5_member}")
                # Each archive entry has a private loader directory, even when house/HDF5 names repeat.
                house = f"house_{first['house_index']}_entry_{entry_index}_part_{first['part']}"
                destination = root / "train" / house
                destination.mkdir(parents=True, exist_ok=True)
                h5_path = destination / Path(h5_member).name
                h5_path.write_bytes(h5_bytes)
                traj_lengths = {}
                with h5py.File(h5_path, "r+") as h5:
                    for row in h5_rows:
                        traj_key = row["traj_key"]
                        if traj_key not in h5:
                            raise SelectedDataError(f"missing trajectory: {h5_member}/{traj_key}")
                        if len(h5[traj_key]["success"]) != row["length"]:
                            raise SelectedDataError(f"length mismatch: {h5_member}/{traj_key}")
                        sensor = h5[traj_key]["obs"]["sensor_data"]
                        batch_name = h5_path.stem.removeprefix("trajectories_")
                        traj_num = int(traj_key.split("_")[1])
                        for camera in CAMERAS:
                            video_name = f"episode_{traj_num:08d}_{camera}_{batch_name}.mp4"
                            video_member = str(Path(h5_member).parent / video_name)
                            video = _member_bytes(archive, video_member)
                            (destination / video_name).write_bytes(video)
                            if camera in sensor:
                                stored = sensor[camera][:].tobytes().decode("utf-8").rstrip("\x00")
                                if stored != video_name:
                                    raise SelectedDataError(f"unexpected camera mapping: {traj_key}/{camera}")
                            else:
                                sensor.create_dataset(camera, data=np.frombuffer(video_name.encode(), dtype=np.uint8))
                        traj_lengths[traj_key] = row["length"]
                rel = f"{house}/{h5_path.name}"
                index.setdefault(house, {})[rel] = traj_lengths
    index_path = root / "train" / "valid_trajectory_index.json"
    index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
    return index


@contextmanager
def selected_dataset(rows: list[dict], model_config):
    with tempfile.TemporaryDirectory(prefix="ttt_selected_") as folder:
        root = Path(folder)
        materialize(rows, root)
        config = SynthmanipDatasetConfig(
            data_path=str(root), camera_names=CAMERAS,
            action_move_group_names=list(ACTION_SPECS["RBY1_multitask"]),
            action_spec=ACTION_SPECS["RBY1_multitask"],
            action_keys=ACTION_DATASET_KEYS["RBY1_multitask"],
            state_spec=STATE_SPECS["RBY1_multitask"],
            state_indices=STATE_INDICES["RBY1_multitask"],
            action_horizon=model_config.action_horizon,
            input_window_size=model_config.n_obs_steps,
            robot_processor_config=model_config.robot_preprocessor,
            split="train", load_policy_phase=True,
        )
        yield SynthmanipDataset(config)
