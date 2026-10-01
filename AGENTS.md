# 项目协作指南

## 项目目标

- 本项目通过 MolmoBot-RBY1Multitask 的检索式 episodic TTT，验证 RBY1 移动操作数据对当前模型和任务的适应价值。
- 开始工作前阅读 `README.md`、`IMPLEMENTATION_PLAN.md`，按任务查阅 `STEP1_AUDIT.md`、`STEP2_REPORT.md` 和各子目录 README。
- 当前文档记录步骤一、二已完成，步骤三受控 benchmark 尚未实施；不要把单例冒烟结果表述为数据质量结论。

## 目录职责

- `audit/`：数据契约审计、来源冻结和快照校验。
- `gates/`：G1 数据充分性、G2 接口对齐、G3 统计功效、G4 仿真冒烟；统一配置在 `gate_config.py`，闸门产物在 `gates/artifacts/`。
- `ttt/`：确定性检索、按需数据解码、action expert 适应、在线策略和单例入口。
- `tests/test_step2.py`：`unittest` 测试，包含真实数据子集集成检查，不是完全脱离外部资源的纯单元测试。
- `artifacts/`：冻结快照、成功轨迹索引、审计结果；`artifacts/step2_smoke/` 为已忽略的冒烟输出目录。

## 数据与模型边界

- 父目录中的 `MolmoBot/`、`gc_ttt/`、`nas_wenyifan/`、`molmospaces_data/` 都是外部只读依赖；不修改源码、模型、benchmark 或原始归档，不复制大文件进本项目。
- 只使用冻结快照对应、审计通过的成功轨迹索引；已有文档记录可访问子集为 17,120 条，不代表 NAS 全量完整。
- 新增 NAS shard 必须通过新快照和审计纳入，不得混入旧快照。维持 benchmark house 排除，并区分语义覆盖与近重复泄漏。
- 派生训练缓存使用临时目录，并保证正常退出及异常路径的清理。持久结果写入本项目对应产物目录。
- 不擅自覆盖冻结索引或历史验收产物；改变数据、配置或评测协议时记录来源和影响。

## 运行与验证

下列命令从父目录 `MoMaTrajGen/` 执行，而不是当前包目录：

```bash
cd /data0/wenyifan/MoMaTrajGen
PYTHONPATH="$PWD:$PWD/MolmoBot/MolmoBot" MolmoBot/MolmoBot/.venv/bin/python -m unittest -v ttt_data_validation.tests.test_step2

# GPU 单例冒烟：先确认目标 GPU 可用，再显式指定编号。
MolmoBot/MolmoBot/.venv/bin/python ttt_data_validation/ttt/run_step2.py --gpu 6 --episode-idx 24
```

CPU 闸门和 G4 静态检测从本项目执行：

```bash
cd gates
python3 g1_data_sufficiency.py
python3 g2_alignment.py
python3 g3_power.py
python3 g4_smoke.py --detect-only
```

- 闸门脚本可能写回结果，运行前确认是否需要保留已有产物；GPU 仿真和真实数据检查不是无成本操作。
- 涉及 MolmoBot/MolmoSpaces 的验证优先使用上述虚拟环境，不假定系统 Python 已安装依赖。
- 按改动范围运行相关测试；无法运行时明确缺失资源及未验证项，不声称测试通过。

## 实现与评测约束

- 遵循现有 Python 风格：四空格缩进、`snake_case`、类型注解、`pathlib.Path`；保持模块职责清晰，避免无关重构。
- 保持已验证接口：动作维度 20、状态维度 22、训练 action horizon 16，复用 checkpoint 的归一化与原始 flow loss。
- 首版仅在 episode 首帧触发一次 TTT；在线 `policy_phase` 不可用，不把它当成有效阶段信号。
- 每个 episode 恢复原始模型参数并重建 optimizer；冻结视觉/VLM 主干，仅更新 FP32 存储的 `action_expert`。无检索命中时保持基线策略，损坏归档须明确报错。
- 当前默认 Top-K=16、有效 batch=16、microbatch=1、20 次更新、学习率 `1e-5`，这些只是已冒烟的工程默认值，不是最优配置。
- 正式比较须固定相同 episode、seed、预算和成功判据，显式设置 `end_on_success=True`；同时比较无 TTT、Random-TTT 和 Retrieved-TTT，报告配对统计、置信区间、安全指标及额外时延。
- 记录已知风险：归一化第 10、18 动作维异常、训练 16 步与 rollout 执行 8 步差异、相机差异、近重复检查及功效限制。不得无依据消除这些限制或发布数据质量排序。

## 交付要求

- 修改行为或运行方式时同步相关 README；阶段状态和验收结论变化时更新对应计划或报告。
- 不提交临时缓存、模型权重、大型视频或无关生成文件。
- 较复杂任务完成后先输出 Markdown 中文总结，说明执行过程、改动结果、实际验证和剩余限制。
