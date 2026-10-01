#!/usr/bin/env python3
"""Read-only smoke check of two archived examples through MolmoBot's own loader."""
import io
import json
import tarfile
import tempfile
from pathlib import Path

import h5py
import numpy as np
import yaml
import zstandard as zstd

from olmo.data.robot_processing import RobotProcessorConfig
from olmo.data.synthmanip_config import SynthmanipDatasetConfig
from olmo.data.synthmanip_dataset import SynthmanipDataset
from olmo.data.synthmanip_presets import ACTION_SPECS, ACTION_DATASET_KEYS, CAMERA_PRESETS, STATE_INDICES, STATE_SPECS
from olmo.models.model_config import BaseModelConfig

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / 'nas_wenyifan/models/MolmoBot-RBY1Multitask/config.yaml'
DATA = ROOT / 'nas_wenyifan/molmospaces_data'
OUT = Path(__file__).resolve().parents[1] / 'artifacts/sample_contract.json'


def one(name, model_cfg, proc_cfg, train_pre, collator):
    table = json.loads((DATA / name / 'arrow_table.json').read_text())
    item = next(x for x in table if '_house_' in x['path'] and x['part'] == 0 and (DATA / name / 'shards' / f"{x['shard_id']:05d}.tar").is_file())
    shard = DATA / name / 'shards' / f"{item['shard_id']:05d}.tar"
    with shard.open('rb') as src:
        src.seek(item['offset'])
        compressed = src.read(item['size'])
    archive_bytes = zstd.ZstdDecompressor().decompress(compressed, max_output_size=item['inflated_size'] + 1024 * 1024)
    with tempfile.TemporaryDirectory(prefix='ttt_contract_') as tmp:
        data_root = Path(tmp)
        train = data_root / 'train'
        index = {}
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode='r:') as archive:
            for member in archive:
                if not member.isfile() or not (member.name.endswith('.h5') or member.name.endswith('.mp4')):
                    continue
                dest = train / member.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.extractfile(member).read())
                if dest.suffix == '.h5':
                    with h5py.File(dest, 'r+') as h5:
                        for traj_key in [k for k in h5 if k.startswith('traj_')]:
                            traj_num = int(traj_key.split('_')[1])
                            sensor = h5[traj_key]['obs']['sensor_data']
                            batch = dest.stem.removeprefix('trajectories_')
                            for camera in CAMERA_PRESETS['RBY1_full_with_head_gopro']:
                                video_name = f'episode_{traj_num:08d}_{camera}_{batch}.mp4'
                                if camera not in sensor and (dest.parent / video_name).exists():
                                    sensor.create_dataset(camera, data=np.frombuffer(video_name.encode(), dtype=np.uint8))
                        index.setdefault(dest.parent.name, {})[member.name] = {k: len(h5[k]['success']) for k in h5 if k.startswith('traj_')}
        # Member order is arbitrary: complete camera links after all MP4s exist.
        for h5_path in train.glob('house_*/*.h5'):
            with h5py.File(h5_path, 'r+') as h5:
                batch = h5_path.stem.removeprefix('trajectories_')
                for traj_key in [k for k in h5 if k.startswith('traj_')]:
                    traj_num = int(traj_key.split('_')[1])
                    sensor = h5[traj_key]['obs']['sensor_data']
                    for camera in CAMERA_PRESETS['RBY1_full_with_head_gopro']:
                        video_name = f'episode_{traj_num:08d}_{camera}_{batch}.mp4'
                        if camera not in sensor and (h5_path.parent / video_name).exists():
                            sensor.create_dataset(camera, data=np.frombuffer(video_name.encode(), dtype=np.uint8))
        (train / 'valid_trajectory_index.json').write_text(json.dumps(index))
        cfg = SynthmanipDatasetConfig(
            data_path=str(data_root),
            camera_names=CAMERA_PRESETS['RBY1_full_with_head_gopro'],
            action_move_group_names=list(ACTION_SPECS['RBY1_multitask']),
            action_spec=ACTION_SPECS['RBY1_multitask'],
            action_keys=ACTION_DATASET_KEYS['RBY1_multitask'],
            state_spec=STATE_SPECS['RBY1_multitask'],
            state_indices=STATE_INDICES['RBY1_multitask'],
            action_horizon=model_cfg['action_horizon'],
            input_window_size=model_cfg['n_obs_steps'],
            robot_processor_config=proc_cfg,
            split='train',
        )
        ds = SynthmanipDataset(cfg)
        sample = ds.get(0, np.random.default_rng(0))
        pre = proc_cfg.build_preprocessor()
        post = proc_cfg.build_postprocessor()
        raw_action = post.unnormalize_action(sample['action'], 'synthmanip')
        roundtrip = pre.normalize_action(raw_action, 'synthmanip')
        expected_keys = {'image', 'question', 'answers', 'style', 'state', 'action', 'action_is_pad', 'metadata'}
        assert expected_keys.issubset(sample)
        assert sample['action'].shape == (16, 20)
        assert sample['state'].shape == (22,)
        assert len(sample['image']) == 3
        assert np.allclose(sample['action'], roundtrip, atol=1e-5)
        raw_state = post.unnormalize_state(sample['state'], 'synthmanip')
        state_roundtrip = pre.normalize_state(raw_state, 'synthmanip')
        assert np.allclose(sample['state'], state_roundtrip, atol=1e-5)
        processed = train_pre(sample)
        batch = collator([processed])
        assert 'actions' in batch and 'states' in batch
        result = {
            'dataset': name,
            'archive': item['path'],
            'h5_members': [x for house in index.values() for x in house],
            'sample_keys': sorted(sample),
            'processed_keys': sorted(processed),
            'batch_keys': sorted(batch),
            'batch_action_shape': list(batch['actions'].shape),
            'batch_state_shape': list(batch['states'].shape),
            'camera_shapes': [list(x.shape) for x in sample['image']],
            'action_shape': list(sample['action'].shape),
            'state_shape': list(sample['state'].shape),
            'action_is_pad_count': int(sample['action_is_pad'].sum()),
            'instruction': sample['question'],
            'action_roundtrip_max_abs_error': float(np.max(np.abs(sample['action'] - roundtrip))),
            'state_roundtrip_max_abs_error': float(np.max(np.abs(sample['state'] - state_roundtrip))),
            'raw_action_min_max': [float(raw_action.min()), float(raw_action.max())],
            'normalized_action_min_max': [float(sample['action'].min()), float(sample['action'].max())],
            'sample_metadata': {k: v for k, v in sample['metadata'].items() if k != 'file_path'},
        }
        return result


def main():
    config = yaml.safe_load(CKPT.read_text())['model']
    proc_cfg = RobotProcessorConfig(**config['robot_preprocessor'])
    post_cfg = RobotProcessorConfig(**config['robot_postprocessor'])
    assert proc_cfg.stats_by_repo == post_cfg.stats_by_repo
    assert config['action_dim'] == 20 and config['action_horizon'] == 16
    assert len(proc_cfg.stats_by_repo['synthmanip']['action']['q01']) == 20
    assert len(proc_cfg.stats_by_repo['synthmanip']['observation.state']['min']) == 22
    model_obj = BaseModelConfig.load(CKPT, key='model', validate_paths=False)
    train_pre = model_obj.build_preprocessor(for_inference=False, is_training=False)
    collator = model_obj.build_collator(train_pre.get_output_shapes(), pad_mode=None, include_metadata=False)
    result = {'checkpoint_config': str(CKPT), 'model_specs': {k: config[k] for k in ['action_dim', 'action_horizon', 'n_action_steps', 'n_obs_steps']}, 'samples': []}
    for name in ['RBY1PickAndPlaceDataGenConfig', 'RBY1PickDataGenConfig']:
        result['samples'].append(one(name, config, proc_cfg, train_pre, collator))
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
