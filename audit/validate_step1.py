#!/usr/bin/env python3
"""Validate step-one JSONL artifacts; never opens or modifies source data."""
import collections
import json
from pathlib import Path

out = Path(__file__).resolve().parents[1] / 'artifacts'
summary = json.loads((out / 'audit_summary.json').read_text())
rows = [json.loads(x) for x in (out / 'entry_audit.jsonl').open()]
selected = [json.loads(x) for x in (out / 'success_index.jsonl').open()]
assert len({(r['dataset'], r['entry_index']) for r in rows}) == len(rows), 'duplicate archive entries'
locators = set()
for r in selected:
    loc = (r['dataset'], r['entry_index'], r['h5_member'], r['traj_key'])
    assert loc not in locators, f'duplicate locator {loc}'
    locators.add(loc)
    assert r['status'] == 'ok' and r['eligible'] and r['terminal_success']
    assert r['valid_traj_mask'] is not False
    assert not r['benchmark_house_overlap']
    assert r['length'] > 2 and r['instruction']
for name, stats in summary['datasets'].items():
    assert stats['scanned_house_entries'] == sum(r['dataset'] == name for r in rows)
    assert stats['counts'].get('eligible', 0) == sum(r['dataset'] == name for r in selected)
    assert stats['scanned_house_entries'] == stats['snapshot_house_entries']
print(json.dumps({'entry_rows': len(rows), 'selected_rows': len(selected), 'selected_by_dataset': dict(collections.Counter(x['dataset'] for x in selected)), 'status': 'passed'}, ensure_ascii=False))
