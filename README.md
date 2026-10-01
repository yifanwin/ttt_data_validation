# RBY1 数据质量验证管线

本目录独立于 MolmoBot、GC-TTT 和 NAS 原始数据。模型、benchmark 与原始归档只读；派生训练缓存仅按需写入临时目录，并在使用后删除。

| 位置 | 用途 |
|---|---|
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | 三步总体计划及当前状态 |
| [`STEP1_AUDIT.md`](STEP1_AUDIT.md) | 步骤一数据契约和隔离结果 |
| [`audit/`](audit/) | 冻结、审计、校验快照的脚本 |
| [`gates/`](gates/) | G1–G4 数据、口径、功效和仿真闸门 |
| `ttt/` | 步骤二检索、按需缓存、模型适应、在线策略与单例入口 |
| `tests/` | 步骤二单元和真实子集测试 |
| [`artifacts/`](artifacts/) | 冻结索引和审计摘要；`step2_smoke/` 存在线单例输出 |

## 推荐顺序

1. 阅读 [`STEP1_AUDIT.md`](STEP1_AUDIT.md)，需要时按 [`audit/README.md`](audit/README.md) 复核快照。
2. 按 [`gates/README.md`](gates/README.md) 复核 G1–G4。
3. 按 `ttt/README.md` 运行步骤二测试和单例冒烟。步骤二**不**给出数据质量统计结论；基线对照留给步骤三。

当前只使用审计通过的 **17,120** 条成功轨迹子集；NAS 其余分片仍不完整。在线 `policy_phase` 不可用，首版只在 episode 首帧触发一次 TTT。
