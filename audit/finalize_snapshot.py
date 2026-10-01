#!/usr/bin/env python3
"""Freeze a finished entry audit to the source_snapshot.json shard inventory."""
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / 'artifacts'
snapshot = json.loads((OUT / 'source_snapshot.json').read_text())
bench = json.loads((ROOT / 'molmospaces_data/assets/benchmarks/molmospaces-bench-v1/procthor-10k/RBY1PickAndPlaceDataGenConfig/RBY1PickAndPlaceDataGenConfig_20260209_json_benchmark/benchmark.json').read_text())
bh = {int(x['house_index']) for x in bench}
all_rows = [json.loads(x) for x in (OUT / 'entry_audit.jsonl').open()]
expected = {}
for name, files in snapshot['shards'].items():
    table = json.loads((ROOT / 'nas_wenyifan/molmospaces_data' / name / 'arrow_table.json').read_text())
    ids = {int(f[:5]) for f in files if f.endswith('.tar')}
    expected[name] = {(name, i) for i, item in enumerate(table) if '_house_' in item['path'] and item['shard_id'] in ids}
    for filename, old in files.items():
        path = ROOT / 'nas_wenyifan/molmospaces_data' / name / 'shards' / filename
        if path.stat().st_size != old['size'] or path.stat().st_mtime_ns != old['mtime_ns']:
            raise RuntimeError(f'Shard changed since source snapshot: {path}')
want = set().union(*expected.values())
selected = [r for r in all_rows if (r['dataset'], r['entry_index']) in want]
keys = [(r['dataset'], r['entry_index']) for r in selected]
if len(keys) != len(set(keys)) or set(keys) != want:
    raise RuntimeError(f'Incomplete or duplicated snapshot audit: expected {len(want)}, found {len(set(keys))}')
path = OUT / 'entry_audit.jsonl'
tmp = path.with_suffix('.jsonl.tmp')
with tmp.open('w') as f:
    for r in selected:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
tmp.replace(path)
summary = {'source_snapshot_utc': snapshot['captured_utc'], 'benchmark_episodes': len(bench), 'benchmark_houses': len(bh), 'datasets': {}}
with (OUT / 'success_index.jsonl').open('w') as sink:
    for name in snapshot['shards']:
        table = json.loads((ROOT / 'nas_wenyifan/molmospaces_data' / name / 'arrow_table.json').read_text())
        entries = [x for x in table if '_house_' in x['path']]
        rows = [x for x in selected if x['dataset'] == name]
        trajs = [t for r in rows for t in r['trajectories']]
        counts = collections.Counter(r['status'] for r in rows)
        counts['trajectories'] = len(trajs)
        counts['terminal_success'] = sum(t['terminal_success'] for t in trajs)
        counts['eligible'] = sum(t['eligible'] for t in trajs)
        counts['excluded_benchmark_house'] = sum(len(r['trajectories']) for r in rows if r['benchmark_house_overlap'])
        counts['invalid_mask'] = sum(t['valid_traj_mask'] is False for t in trajs)
        counts['success_nonterminal_only'] = sum(t['any_success'] and not t['terminal_success'] for t in trajs)
        counts['missing_instruction'] = sum(not t['instruction'] for t in trajs)
        counts['exact_h5_traj_duplicate'] = sum(n - 1 for n in collections.Counter((t['h5_sha256'], t['traj_key']) for t in trajs).values())
        for r in rows:
            for t in r['trajectories']:
                if t['eligible']:
                    sink.write(json.dumps({k: v for k, v in r.items() if k != 'trajectories'} | t, ensure_ascii=False) + '\n')
        houses = {int(re.search(r'_house_(\d+)', x['path']).group(1)) for x in entries}
        summary['datasets'][name] = {
            'total_house_entries': len(entries),
            'snapshot_house_entries': len(expected[name]),
            'scanned_house_entries': len(rows),
            'unavailable_house_entries': len(entries) - len(expected[name]),
            'total_houses': len(houses),
            'benchmark_overlap_houses': len(houses & bh),
            'counts': dict(counts),
        }
(OUT / 'audit_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(summary, indent=2, ensure_ascii=False))
