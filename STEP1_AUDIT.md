# 步骤一：MolmoBot × RBY1 数据契约与隔离审计

> 审计范围：模型与数据**只读**；代码和审计产物仅位于 `ttt_data_validation/`。未加载 20 GB 权重、未训练、未运行 benchmark。结果基于本目录的可复查脚本与 `artifacts/`；NAS 数据正在增量出现，因此成功索引只代表扫描时**可访问且成功解码**的部分，不能冒充完整原始数据集。

## 结论

- 两批数据各抽取一个原始归档，并通过 MolmoBot **原始数据加载器、原始多模态预处理器和 collator**：batch 的 `actions=[1,16,20]`、`states=[1,22]`、三路 RGB 相机均为 `480×640×3`；动作及状态的 checkpoint 归一化往返误差见 [`sample_contract.json`](artifacts/sample_contract.json)。这验证了**样本级输入契约**，不是全量数据或模型前向验证。
- 原始数据不是 MolmoBot 加载器直接可读的 `train/house_*/*.h5` 布局，而是 `arrow_table.json → shards/*.tar → house_*.tar.zst → HDF5 + MP4`。抽查 HDF5 的 `obs/sensor_data` 没有相机文件映射；冒烟测试只在**临时副本**中按视频文件名重建映射后通过。因此第二步必须实现可重复的只读流式适配或派生缓存，不能直接指向 NAS 归档让原始加载器读取。
- benchmark 的 2,000 个 episode 涉及 648 个 house，`data_split=val`，`seed` 字段均为 `null`。按 `house_index` 保守隔离，在 PnP 和 Pick 索引中分别发现 **78**、**173** 个与 benchmark 重合的 house，已列入 [`leakage_exclusions.json`](artifacts/leakage_exclusions.json)；成功索引不收录这些 house。
- **快照审计结果**：2026-09-30 12:38（上海时间）冻结 27 个 PnP、43 个 Pick shard；分别扫描 3,248 / 9,911（32.8%）和 5,321 / 30,892（17.2%）个 house 归档条目，共 8,569 条，解码错误 **0**。审计到 5,573 条 PnP、13,299 条 Pick 轨迹；末帧均成功，但有效性 mask 排除 323 / 1,328 条，benchmark house 排除 42 / 68 条（两项有交集），最终成功库收录 **5,211 + 11,909 = 17,120** 条；精确 HDF5+轨迹重复为 0。详见 [`audit_summary.json`](artifacts/audit_summary.json)。
- **完整数据集验收未通过**：NAS 当前只有部分 shard 可访问；缺失分片的成功/失败、有效性和近重复情况无法审计。只有 [`success_index.jsonl`](artifacts/success_index.jsonl) 的已验证子集可供后续工作使用；正式数据质量结论必须等分片补齐并重跑审计。

## 数据契约

| 项目 | 实测 / 源码约定 | 接入要求 |
|---|---|---|
| 存储定位 | `arrow_table.json` 的 `shard_id + offset + size + part` 指向外层 tar 中独立 zstd 包；同名 house/同名 HDF5 可在不同 part 出现且内容不同 | 使用 `(dataset, entry_index, h5_member, traj_key)` 作为稳定定位键，不能仅用 HDF5 文件名去重；原始归档只读 |
| 任务与语言 | `traj_i/obs_scene` JSON 提供 `task_type`、`task_description`、`object_name`、`place_receptacle_name`、`referral_expressions` | 指令取 `task_description`；object/target 内部 ID 不等于语义类别，语义匹配需从文本/引用表达提取；Pick 的 target 可为空 |
| 阶段 | `traj_i/obs/extra/policy_phase` 整数序列，`obs_scene.policy_phases` 通常给出 `pregrasp/grasp/lift/place/postplace/done` 映射 | 可从离线轨迹取阶段；它不是原计划中的 `navigate/transport` 标签。在线 benchmark observation 是否提供对应阶段尚未证实，触发器不能默认可读真值 |
| 成功 | `traj_i/success` 是**逐帧**布尔量；抽样轨迹大部分帧为 False、末帧为 True。部分 HDF5 有 `valid_traj_mask` | 首版仅纳入**末帧成功**、非无效 mask、长度 >2、指令非空、无 benchmark house 重合的轨迹；不能把 `fail.any()` 当作整段失败 |
| 图像 | 外部 MP4 文件：`wrist_camera_r`、`head_camera`、`wrist_camera_l`；抽样帧 `480×640×3` | 顺序与 MolmoBot `RBY1_full_with_head_gopro` preset 一致；派生缓存须重建 HDF5 视频映射或等价地直接喂帧 |
| 状态 | `obs/agent/qpos` 为 JSON 字节；按 `base(3), left_arm(7), left_gripper(1), right_arm(7), right_gripper(1), torso[1,2,3](3)` 拼接，共 22 维 | 与训练 preset/评测策略一致；不能把 6 维原始 torso 或 2 维原始 gripper 直接输入 |
| 动作 | `joint_pos_rel` 提供 base/双臂相对动作；`joint_pos` 提供双 gripper、torso 绝对动作；合计 20 维。首帧 padding，动作从 `step+1` 取，16 步 chunk，末尾不足则 mask | 严格复用 MolmoBot 的动作抽取和 padding；checkpoint 的 `action_dim=20`、`action_horizon=16`、推理 `n_action_steps=8` |
| 归一化 | checkpoint `config.yaml` 内 `synthmanip` 统计：state 22 维 `min_max`，action 20 维 `q01/q99` quantiles；`robot_preprocessor` 与 `robot_postprocessor` 一致 | TTT 目标动作只能使用该 checkpoint 统计，不重算生成数据统计；推理回原动作空间由原 postprocessor 完成 |
| 几何 | HDF5 含 `obj_start`、`obj_end`、`robot_base_pose`、`tcp_pose` 等 7 维位姿；benchmark JSON 含初始物体/机器人位姿 | 物体相对几何可作为后续检索候选，但坐标系和逐帧含义尚未通过仿真核验，不作为步骤一已通过的数值匹配契约 |

## 隔离与可用范围

- 基准原始源路径指向另一个 `/weka/.../val/house_*/*.h5` 数据树；本地提供的是 benchmark JSON，未提供其 HDF5/MP4。因此可以执行**房屋级**重合排除与 JSON 元信息核对，但无法比较评测视频或动作序列的近重复哈希。跨 house 的视觉近重复风险仍然存在，正式评测前需补充核验。
- `success_index.jsonl` 仅含符合上述准入规则的 **17,120** 条轨迹，保留 archive 偏移、part、HDF5 SHA256、task、阶段映射和原始定位信息。`entry_audit.jsonl` 保留每个已扫描归档的全部轨迹审计结果；`audit_summary.json` 给出覆盖率、排除与错误计数。
- 来源指纹见 [`source_snapshot.json`](artifacts/source_snapshot.json)：包含四份小文件 SHA256、模型权重大小/mtime、三个源码仓库 commit 及分片清单。**权重文件未计算 SHA256**；NAS 分片为动态状态，这不是不可变快照。

## 样本验收与剩余门槛

| 检查 | 结果 |
|---|---|
| 原始 PnP 与 Pick 各一份 HDF5 + MP4 经原始数据加载器 | **通过**；三相机、指令、22 维状态、16×20 动作及 padding mask 均返回 |
| 原始多模态预处理器及 collator | **通过**；两个样本均形成 `actions=[1,16,20]`、`states=[1,22]` 的 batch |
| checkpoint 归一化/反归一化往返 | **通过（样本级）**；误差数值见 `sample_contract.json` |
| 失败/无效轨迹及 benchmark house 排除 | **对已扫描归档通过**；全量覆盖取决于缺失 shard 补齐 |
| 模型前向 loss、在线评测 observation 与阶段触发、跨 house 近重复 | **未验证**；属于第二步接入前必须补齐的关口 |

### 复现命令

```bash
MolmoBot/MolmoBot/.venv/bin/python ttt_data_validation/audit/audit_step1.py --snapshot ttt_data_validation/artifacts/source_snapshot.json
python ttt_data_validation/audit/finalize_snapshot.py
python ttt_data_validation/audit/validate_step1.py
PYTHONPATH=MolmoBot/MolmoBot MolmoBot/MolmoBot/.venv/bin/python ttt_data_validation/audit/check_sample_contract.py
```

以上命令复核**当前冻结快照**。`audit_step1.py` 会跳过已审计条目；要扩大覆盖范围，先用 `freeze_sources.py` 生成**新快照**，再审计、定稿和校验。旧快照若分片大小或 mtime 变化，`finalize_snapshot.py` 会拒绝定稿，须清理对应缓存后重扫。
