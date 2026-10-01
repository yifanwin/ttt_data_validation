# 移动操作数据质量验证管线：三步实现计划

> 状态：步骤一已审计 2026-09-30 冻结的**可访问 NAS 快照**，详见 [STEP1_AUDIT.md](STEP1_AUDIT.md)；步骤二已实现并完成 CUDA 4 单例验收，详见 [STEP2_REPORT.md](STEP2_REPORT.md)；步骤三尚未实施。全量 NAS 仍缺 shard。工作目录为 `ttt_data_validation/`；模型、离线数据和 benchmark 只读引用原路径。

## 目标与判定问题

用 **MolmoBot-RBY1Multitask + RBY1 Pick-and-Place benchmark** 检验一批移动操作数据：在相同 benchmark episode、初始条件和模型下，检索该批数据进行短时训练（TTT）后，是否比**不适应**及**随机数据 TTT**更有利于任务成功。TTT 的收益只是数据可用性的*间接证据*，须结合数据覆盖、动作对齐和检索命中情况判断，不能把成功率变化直接等同于数据质量。

| 大步骤 | 主要工作 | 交付物 | 完成标准 / 决策门槛 |
|---|---|---|---|
| **1. 接口与数据契约核验** | 在独立工作目录记录并锁定 checkpoint、MolmoBot 代码、两份离线数据及 benchmark 的版本/路径；核对相机、指令、proprio、动作维度/顺序/单位、时间步、action horizon、成功标记和 episode/house 标识。将原始数据整理为可索引 episode，并定义 `task/stage/object/target/success` 的提取或标注规则。只让 `success=True` 进入首版 experience bank；按 house、场景、episode 和近重复轨迹检查评测泄漏。验证训练输入使用 checkpoint 自带的 state/action normalization，推理输出按原 action space 反归一化。 | 数据契约、样本审计表、独立的成功轨迹索引、训练/评测隔离清单、最小端到端样本检查记录。 | 能从样本稳定构造 MolmoBot 训练 batch；20 维动作及 16 步 action horizon 等实际规格与 checkpoint 一致；归一化/反归一化、相机与状态映射通过往返/形状检查；评测轨迹及其近重复不进入经验库。缺失 stage 或几何字段时先明确可复现的降级检索规则，不凭空假设其存在。 |
| **2. 检索与 episodic TTT 闭环（已完成）** | 基于审计索引做 critic-free 任务/物体/`pregrasp` 检索，按指令相似度返回 Top-K；无命中则保留基线策略。仅在 episode 首帧适应一次：恢复原始参数、重建 optimizer、冻结 VLM/视觉主干，只更新 FP32 存储的 `action_expert`，复用原 16 步 flow loss。在线阶段信号不可用，物体几何坐标未核验，故阶段触发及几何匹配不属于首版。 | `ttt/` 中的检索、按需缓存、TTT 更新、在线策略和运行入口；`tests/` 中的测试；[步骤二报告](STEP2_REPORT.md)。 | 7/7 测试通过；CUDA 4 单例完成 `reset → retrieve → adapt → rollout`，Top-16、20 次更新及实际参数变化均落盘；单例成败不用于数据质量判定。 |
| **3. 受控 benchmark 与数据质量报告** | 固定 benchmark episode/seed、任务时长和评测协议，先跑无 TTT 基线，再跑 Random-TTT（`ratio=0`）和 Retrieved-TTT（`ratio=1`）；资源允许再做 `ratio=0.5`、Top-K/步数/学习率及固定 K 与可观测阶段检测消融。按每批数据分别建库和评测，并保留相同预算；统计 overall、pick、place、完整 Pick&Place 成功率，碰撞、TTT 延迟、检索覆盖率/数据量及更新步数。按 episode 配对比较并报告置信区间，结合失败案例检查“检索正确但动作不兼容”等机制。 | 可复现配置与结果清单、逐 episode 指标、对比表、数据质量结论及适用边界。 | 基线与 TTT 使用同一批评测 episode；报告同时给出收益、方差/置信区间和额外时延；若检索无覆盖、存在泄漏或动作映射未验证，则不发布质量排序；只在 Retrieved-TTT 相对无 TTT 和 Random-TTT 均有可信增益且安全指标未恶化时，判为“该数据对当前任务和模型有正向适应价值”。 |

## 首版冻结配置与待核验点

- **模型与路径**：`nas_wenyifan/models/MolmoBot-RBY1Multitask`；源码 `MolmoBot/MolmoBot`；参考实现 `gc_ttt/utils/evaluation.py`、`gc_ttt/utils/datasets.py`。GC-TTT 只借鉴检索和 reset-and-adapt 结构，不复用其 GCIQL/critic。
- **离线数据**：`nas_wenyifan/molmospaces_data/RBY1PickAndPlaceDataGenConfig`、`nas_wenyifan/molmospaces_data/RBY1PickDataGenConfig`。**评测**：`molmospaces_data/assets/benchmarks/molmospaces-bench-v1/procthor-10k/RBY1PickAndPlaceDataGenConfig/RBY1PickAndPlaceDataGenConfig_20260209_json_benchmark`。步骤一已核验当前可访问子集的轨迹字段、成功标签及 house 级 benchmark 重叠；其余 shard 尚未审计。benchmark 元数据标称 2,000 个 episode，不代表首版必须全量运行。
- **步骤二已实现的默认超参**：Top-K=16、有效 batch size=16（microbatch=1）、TTT steps=20、学习率 `1e-5`、`ratio=1.0`、仅更新 action expert。它们只通过工程冒烟，不是最优或数据质量结论；步骤三正式比较前须冻结配置。后续扫描 steps `10/20/50`、学习率 `1e-6/1e-5/3e-5`、Top-K `8/16/32`。
- **已验证接口及剩余风险**：`action_dim=20`、`action_horizon=16`、checkpoint 归一化、原始加载器 batch、flow loss 前后向和单例仿真已验证。在线 `policy_phase` 无效；跨 house 近重复、数据缺失、逐维归一化异常和正式评测功效仍是步骤三的前置限制。
- **执行边界**：本文件保留三步计划；步骤一、二的代码和验收产物已落盘。步骤三的配对 benchmark、消融与数据质量排序尚未实施。
