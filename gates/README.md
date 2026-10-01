# ttt_data_validation/gates —— 四道前置闸门（G1–G4）

这四个脚本回答的是**「这套 TTT 数据验证管线现在能不能评、评出来可不可信」**，
不是 TTT 本身。G1–G3 纯 CPU，G4 需要 GPU。

沿用 `ttt_data_validation/` 的约定：模型、离线数据、benchmark、MolmoBot 与 GC-TTT
源码全部**只读引用**，不复制大文件，不修改它们。所有产物写入本目录 `artifacts/`。

## 为什么需要闸门

`IMPLEMENTATION_PLAN.md` 的三步计划在算法侧是站得住的，但它建立在若干**未验证的资源与
口径前提**上。这些前提一旦不成立，步骤 2/3 写多少代码都不产出可信结论。四道闸门就是
把这些前提单独验出来，成本远低于先实现 TTT 再发现口径错了。

## 四个闸门

| 闸门 | 脚本 | 依赖 | 回答的问题 |
|---|---|---|---|
| **G1** | `g1_data_sufficiency.py` | 无（纯 CPU） | 检索库按 (task, stage) 分桶后每桶有多少？house 级排除之外还有多少语义泄漏？哪些 benchmark episode 的 instruction 畸形需隔离？ |
| **G2** | `g2_alignment.py` | 无（纯 CPU） | 维度/归一化/相机/成功判据/指令来源五组口径是否对齐？ |
| **G3** | `g3_power.py` | 无（纯 CPU） | 要检出多大效应需要多少 episode？按物体类别分层能降多少方差？给定 worker 数的墙钟预算是多少？ |
| **G4** | `g4_smoke.py` + `g4_eval_configs.py` | GPU、场景资产 | 单个 benchmark episode 能否本地走通 `场景实例化 → reset → 前向 → step → success`？ |

## 运行

```bash
cd ttt_data_validation/gates

# G1–G3：秒级，纯 CPU
python3 g1_data_sufficiency.py
python3 g2_alignment.py
python3 g3_power.py

# G4：先静态检查（秒级，不启动仿真）
python3 g4_smoke.py --detect-only

# G4：实际跑一个 episode（约 2–4 分钟）
python3 g4_smoke.py --gpu 6
```

G4 的 `--gpu` 覆盖 `gate_config.G4_GPU`；换目标 house 用 `--house-index`。

## 判定口径

每个闸门输出三档判定，写入 `artifacts/g<N>_*.json` 的 `verdict` 字段：

- `pass`：无阻塞，可进入下一步
- `warn`：可继续，但有一条需要记录在案的偏差
- `fail`：阻塞，必须先解决

## 当前结论（2026-09-30）

- **G1 `warn`**。检索库 17,120 条（Pick 11,909 / PnP 5,211），house 级排除生效（重合 0）。
  但 **stage 作为过滤条件基本失效**：17,059/17,120 条的阶段集合都覆盖到 `done`，
  `(task, stage)` 分桶退化成 `task/done`。计划第 4 节「轨迹经过当前 manipulation
  stage → 截取之后 clip」在轨迹粒度上无法区分轨迹，区分度只能来自逐帧
  `policy_phase` 序列。
  另有 **1,998/2,000 个 benchmark episode 的物体类别在检索库中出现过**——这是
  house 级排除挡不住的语义覆盖，需补近重复核验。
- **G2 `warn`**。维度与归一化往返全通过。三处偏差：action 归一化的第 **10、18 维**
  量纲异常（q01/q99 = ±100，与正常维的梯度权重比约 **5e-4**）；训练 chunk 16 步
  vs rollout 每 chunk 执行 **8 步**；benchmark 相机多出 `camera_follower`，且
  benchmark 记录的是 640×480 而训练用 1024×576。**`end_on_success` 默认 False，
  必须显式置 True**（见 `g4_eval_configs.py`）。
- **G3 `warn`**。历史基线 226 episode 的 95% CI 宽度约 **8.1 个百分点**，只够检出
  翻倍级效应。配对设计下：1.5× 需 529 ep、**2.0× 需 133 ep**、2.5× 需 59 ep。
  推荐首版 **150 episode**。626 个 episode 的物体类别无历史基线。
- **G4 `pass`**（模式 `executed`）。house_107 / idx 24（"pick up the brown egg and place
  it on the bowl"）单 episode 走通：场景重建 → 4 相机 → 401 步 rollout → h5 + 4 路 mp4
  落盘，耗时 **494 s**。`end_on_success` 已被 `g4_eval_configs.py` 强制为 `True`
  并在运行配置里确认生效。
  该 episode 判 **success=False**，`task_info` 给出分量：
  `supported_by_receptacle=false`（蛋没进碗）、`position_error=0.205`（>0.15m 阈值）、
  `robot_contact=false`（已松手）。失败原因是策略没完成任务，**不是判据口径问题**。
  与 E3 报告"house_107 5/5 全成功"的差异属于同一 house 内不同 episode 的正常波动
  （E3 未保存 house_107 的 h5，无法逐条复核）。
  **重要副产品**：这次 rollout 的 `policy_phase` **全程为 0**。核查历史两次运行
  （E3 learned `house_52`、E2 CuRobo `house_29`）**也都恒为 0**。
  即 `policy_phase` 字段在 schema 里存在但两条评测路径都没填，
  **在线阶段信号实际上不存在**——计划第 8 节「stage 变化时触发 TTT」不可实现。

## 与历史基线的关系

G3 的功效估算锚定在 `MolmoBotRBY1PickPnPEvalConfig` 那次运行上
（`MolmoBot/MolmoBot/eval_output/MolmoBotRBY1PickPnPEvalConfig/20260916_220518/`，
**24/226 = 10.62%**，结束原因是 SIGTERM 而非正常完成）。

注意该次运行的 `end_on_success=False`，因此它至少漏掉了「过程中成功、末态失败」的
episode。作为基线参考值可用，但**正式比较必须让基线与 TTT 条件用同一套判据口径**。
