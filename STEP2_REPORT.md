# 步骤二：检索式 episodic TTT 实现与验收

> 2026-09-30。工程闭环已跑通；**单 episode 结果不构成数据质量结论**。只使用步骤一审计的 17,120 条成功轨迹子集，未修改上游 MolmoBot、GC-TTT 或 NAS 原件。

## 实现结果

- `ttt/retrieval.py`：从冻结成功索引做任务、物体类别和 `pregrasp` 覆盖筛选，以指令词项相似度稳定排序，返回 Top-16；无命中给出原因并使用原策略。在线阶段信号不可用，因此只在 episode 第一帧适应，不做阶段触发或未经验证的几何匹配。
- `ttt/selected_data.py`：按需解压命中归档，核验 HDF5 SHA256，补齐三路视频映射后交给 MolmoBot 原加载器；同名 house/文件按归档条目隔离，缓存退出即删。
- `ttt/adaptation.py`、`ttt/ttt_policy.py`：冻结 VLM/视觉主干，只训练 FP32 存储的 `action_expert`；复用原始 16 步 flow-matching action loss。每次适应前恢复原始参数并重建 AdamW，更新后释放梯度与优化器状态，推理仍走原动作反归一化和 8 步执行链路。
- 目录按 `audit/`、`gates/`、`ttt/`、`tests/`、`artifacts/` 分组；入口及复现命令见 [README.md](README.md) 和 [ttt/README.md](ttt/README.md)。

## 验收记录

| 检查 | 结果 |
|---|---|
| 自动化测试 | **7/7 通过**：Pick/PnP 真实样本构批、确定性检索、无命中、缺 shard/视频、HDF5 哈希错误、同名归档隔离、冻结/更新/精确恢复 |
| 真实模型单 batch | 原 checkpoint + 审计轨迹的 flow loss 前向与反传通过，loss/梯度均为有限值 |
| CUDA 4 单例 | benchmark 全局 idx 24、house 107；Top-16 来自 82 条 egg 候选，20 次更新全部执行，随后 rollout 和成功判定完整落盘；`end_on_success=True` |
| 参数检查 | 536,031,764 个 action-expert 元素中 **535,441,704** 个发生变化；训练前后参数可由 reset 恢复；VLM 不在 optimizer 中 |
| 运行开销 | CUDA 4 的 TTT 更新约 **125.3 s**，PyTorch 进程峰值已分配显存约 **22.6 GB**；整次单例约 **711.5 s** |
| 单例结果 | **0/1 成功**，401 帧；仅证明链路执行完毕，不代表数据无效或 TTT 有害 |

CUDA 4 的[运行汇总](artifacts/step2_smoke/step2_summary.json)、[逐次 TTT 事件](artifacts/step2_smoke/EpisodicTTTEvalConfig/20260930_152849/ttt_events_2780297.jsonl)与原始评测输出位于 `artifacts/step2_smoke/`。同一 FP32 配置在 CUDA 5 也完成 **0/1**，且检索轨迹和参数变化计数一致；早期 BF16 存储试跑为 **1/1**。这些单例差异不能用于数据质量排序。

## 仍未解决的边界

- NAS 只含部分 shard；跨 house 视觉近重复未验证。当前库只适合开发和冒烟，正式质量结论需步骤三的隔离及配对评测。
- G2 指出的动作第 10、18 维归一化异常及训练 16 步/执行 8 步差异被保留并逐维记录；首版没有改变原 loss。
- 在线 `policy_phase` 始终不可作为阶段触发信号；几何匹配、固定 K 重训和 8 步损失消融均未实现。
- 历史 G4 基线与本单例不是足以判断改进的配对统计对照；步骤三应统一 seeds、判据和 episode 集，并报告不确定度。
