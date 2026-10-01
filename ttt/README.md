# 步骤二：检索式 episodic TTT

## 文件职责

| 文件 | 作用 |
|---|---|
| `retrieval.py` | 从步骤一的成功索引做确定性检索，返回轨迹 ID 或跳过原因 |
| `selected_data.py` | 按需解码 Top-K 归档，核验 HDF5 哈希、补视频映射，交给 MolmoBot 原始加载器 |
| `adaptation.py` | 冻结 VLM，只更新 `action_expert`，复用原始 16 步 flow loss 并执行参数恢复 |
| `ttt_policy.py` | MolmoSpaces 在线策略与 config 子类；首帧 TTT，`end_on_success=True` |
| `run_step2.py` | 单 benchmark episode 冒烟入口（不是统计评测） |

## 运行

从项目根目录运行：

```bash
PYTHONPATH="$PWD:$PWD/MolmoBot/MolmoBot" MolmoBot/MolmoBot/.venv/bin/python -m unittest -v ttt_data_validation.tests.test_step2
MolmoBot/MolmoBot/.venv/bin/python ttt_data_validation/ttt/run_step2.py --gpu 6 --episode-idx 24
```

默认 Top-16、有效 batch 16（microbatch 1、梯度累积）、20 次 AdamW 更新、学习率 `1e-5`。checkpoint 仍按 BF16 加载，但仅 `action_expert` 改为 FP32 参数存储，避免小学习率更新被 BF16 舍入；训练前后记录实际改变的元素数。只使用 `artifacts/success_index.jsonl` 中的审计通过子集；无命中时保持 checkpoint 原策略，归档损坏时报出明确原因。训练缓存写入临时目录，退出后自动删除。

结果写入 `artifacts/step2_smoke/`，包含 `step2_summary.json`、评测输出及逐 episode 的 `ttt_events_*.jsonl`。单例成功或失败**不能**推断数据质量；正式配对比较属于步骤三。
