#!/usr/bin/env python3
"""Record read-only source provenance without hashing multi-GB weights/shards."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parents[1] / 'artifacts/source_snapshot.json'
rel = [
 'nas_wenyifan/models/MolmoBot-RBY1Multitask/config.yaml',
 'nas_wenyifan/molmospaces_data/RBY1PickAndPlaceDataGenConfig/arrow_table.json',
 'nas_wenyifan/molmospaces_data/RBY1PickDataGenConfig/arrow_table.json',
 'molmospaces_data/assets/benchmarks/molmospaces-bench-v1/procthor-10k/RBY1PickAndPlaceDataGenConfig/RBY1PickAndPlaceDataGenConfig_20260209_json_benchmark/benchmark.json',
]
snapshot = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'files': {}, 'source_revisions': {}, 'shards': {}}
for p in rel:
 f=root/p;stat=f.stat();snapshot['files'][p]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
weight=root/'nas_wenyifan/models/MolmoBot-RBY1Multitask/model.pt';s=weight.stat();snapshot['files'][str(weight.relative_to(root))]={'sha256':'not_computed_20GB','size':s.st_size,'mtime_ns':s.st_mtime_ns}
for p in ['MolmoBot/MolmoBot','gc_ttt','molmospaces']:
 h=subprocess.check_output(['git','-C',str(root/p),'rev-parse','HEAD'],text=True).strip();snapshot['source_revisions'][p]=h
for name in ['RBY1PickAndPlaceDataGenConfig','RBY1PickDataGenConfig']:
 d=root/'nas_wenyifan/molmospaces_data'/name/'shards';snapshot['shards'][name]={x.name:{'size':x.stat().st_size,'mtime_ns':x.stat().st_mtime_ns} for x in sorted(d.glob('*.tar'))}
out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(snapshot,indent=2,ensure_ascii=False)+'\n')
print(out)
