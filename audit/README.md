# 步骤一审计脚本

数据和模型只读。审计结果固定写回上级目录的 `artifacts/`，不会因脚本移入 `audit/` 而改变。

```bash
MolmoBot/MolmoBot/.venv/bin/python ttt_data_validation/audit/audit_step1.py --snapshot ttt_data_validation/artifacts/source_snapshot.json
python ttt_data_validation/audit/finalize_snapshot.py
python ttt_data_validation/audit/validate_step1.py
PYTHONPATH=MolmoBot/MolmoBot MolmoBot/MolmoBot/.venv/bin/python ttt_data_validation/audit/check_sample_contract.py
```

要扩大覆盖范围，先运行 `freeze_sources.py` 生成新快照；不要把后来出现的 NAS shard 混入旧快照。详见 [`STEP1_AUDIT.md`](../STEP1_AUDIT.md)。
