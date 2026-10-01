"""G1–G4 闸门的统一配置：路径、阈值、常量集中在此处，便于后续修改。

所有路径均为只读引用；本模块不写入、不复制任何源数据。
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- 工作目录
GATE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = GATE_DIR / "artifacts"
ROOT = GATE_DIR.parents[1]  # /data0/wenyifan/MoMaTrajGen

# ---------------------------------------------------------------- 输入路径
# 步骤一已落盘的审计产物（只读消费）
SUCCESS_INDEX = ROOT / "ttt_data_validation/artifacts/success_index.jsonl"
LEAKAGE_EXCLUSIONS = ROOT / "ttt_data_validation/artifacts/leakage_exclusions.json"

BENCHMARK_DIR = (
    ROOT
    / "molmospaces_data/assets/benchmarks/molmospaces-bench-v1/procthor-10k"
    / "RBY1PickAndPlaceDataGenConfig"
    / "RBY1PickAndPlaceDataGenConfig_20260209_json_benchmark"
)
BENCHMARK_JSON = BENCHMARK_DIR / "benchmark.json"
BENCHMARK_METADATA = BENCHMARK_DIR / "benchmark_metadata.json"

CHECKPOINT_DIR = ROOT / "nas_wenyifan/models/MolmoBot-RBY1Multitask"
CHECKPOINT_WEIGHTS = CHECKPOINT_DIR / "model.pt"
CHECKPOINT_CONFIG = CHECKPOINT_DIR / "config.yaml"

# MolmoBot 的虚拟环境解释器。molmo_spaces 只装在这里（site-packages 静态副本，
# 非 editable），因此 G4 必须用这个解释器，不能用跑本脚本的那个。
MOLMOBOT_ROOT = ROOT / "MolmoBot/MolmoBot"
MOLMOBOT_PYTHON = MOLMOBOT_ROOT / ".venv/bin/python"
RUN_EVAL_SCRIPT = MOLMOBOT_ROOT / "launch_scripts/run_eval.py"

# 历史基线：E3（MolmoBot learned）那次运行的输出目录
BASELINE_RUN_DIR = (
    ROOT
    / "MolmoBot/MolmoBot/eval_output/MolmoBotRBY1PickPnPEvalConfig/20260916_220518"
)
BASELINE_RUN_LOG = BASELINE_RUN_DIR / "running_log.log"
BASELINE_REPORT = BASELINE_RUN_DIR / "report.md"

# 场景资产（G4 用）
SCENES_DIR = ROOT / "molmospaces_data/assets/scenes/procthor-10k-val"

# G4 冒烟测试输出目录
SMOKE_OUTPUT_DIR = GATE_DIR / "artifacts/g4_smoke"

# ---------------------------------------------------------------- G1 阈值
# 检索库准入沿用步骤一的规则，这里只做再核对
G1_MIN_BUCKET_SIZE = 30  # 单个 (task, stage) 桶低于此值视为覆盖不足
G1_LEAKAGE_HOUSE_TOLERANCE = 0  # 与 benchmark 重合的 house 数上限

# ---------------------------------------------------------------- G2 阈值
# 归一化往返误差上限（相对量，按维度的 (max-min) 归一）
G2_NORM_RTOL = 1e-4
G2_NORM_ATOL = 1e-5

# 期望的动作/状态维度（来自 checkpoint config）
EXPECT_ACTION_DIM = 20
EXPECT_STATE_DIM = 22
EXPECT_ACTION_HORIZON = 16
EXPECT_EXECUTE_HORIZON = 8

# 评测口径必须对齐的关键开关（G2 检查项）
REQUIRED_END_ON_SUCCESS = True
# 离线数据归一化所用 repo_id
NORM_REPO_ID = "synthmanip"

# ---------------------------------------------------------------- G3 阈值
# 历史基线成功率（E3：24/226）
BASELINE_SUCCESS = 24
BASELINE_TOTAL = 226
BASELINE_RATE = BASELINE_SUCCESS / BASELINE_TOTAL

G3_ALPHA = 0.05  # 显著性水平
G3_POWER = 0.80  # 目标功效
# 目标效应量：相对提升倍数（用于估算所需 episode 数）
G3_EFFECT_MULTIPLIERS = (1.5, 2.0, 2.5)
# 固定抽样的分层方案：按 E3 逐类成功率分组
G3_STRATA_LOW_RATE = 0.05  # 低于此成功率视为低成功层
G3_STRATA_HIGH_RATE = 0.25  # 高于此成功率视为高成功层

# 候选首版规模
G3_CANDIDATE_SIZES = (100, 150, 226, 300, 400)

# ---------------------------------------------------------------- G4 阈值
# 单 episode 端到端冒烟测试的固定 episode
G4_HOUSE_INDEX = 107  # E3 中 5/5 全成功的 house，最可能通过
G4_EPISODE_IDX = 0
G4_EVAL_CONFIG_CLS = "olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPEvalConfig"
G4_TASK_HORIZON = 400
G4_NUM_WORKERS = 1
G4_TIMEOUT_SEC = 900  # 单 episode 参考耗时 ~190s，留 4.7 倍余量
# 冒烟测试使用的 GPU（用 nvidia-smi 挑一张空闲的；None 表示不设置该环境变量）
G4_GPU = "6"

# 缺失场景资产的 house（来自探查，84 个）
G4_KNOWN_MISSING_HOUSES = (
    90, 210, 256, 295, 321, 343, 360, 513, 591, 606, 614, 646, 657, 684, 709, 852,
    891, 893, 895, 896, 897, 898, 899, 900, 901, 903, 904, 905, 906, 907, 908, 909,
    911, 913, 915, 916, 917, 918, 920, 921, 922, 923, 924, 925, 926, 927, 932, 933,
    935, 937, 938, 939, 941, 942, 946, 947, 951, 952, 954, 959, 960, 961, 963, 967,
    968, 970, 971, 973, 974, 975, 978, 979, 982, 983, 985, 986, 988, 991, 993, 995,
    996, 997, 998, 999,
)

# 历史基线逐类成功率（E3 report.md §3.1，约数，用于 G3 分层）
BASELINE_CATEGORY_RATES = {
    "egg": 5 / 7,
    "cup": 4 / 12,
    "candle": 3 / 10,
    "mug": 6 / 22,
    "winebottle": 1 / 7,
    "bottle": 2 / 13,
    "soapdispenser": 1 / 7,
    "peppershaker": 1 / 14,
    "remotecontrol": 1 / 20,
    "potato": 0.0,
    "butterknife": 0.0,
}

# benchmark 侧已知的畸形 instruction 特征（G3/G4 隔离用）
MALFORMED_INSTRUCTION_PATTERNS = (
    "pick up the means",
    "instrumentation",
    "place receptacl",
    "the woodeowl",
)

# ---------------------------------------------------------------- 通用
# 判定结果的三档取值
VERDICT_PASS = "pass"
VERDICT_WARN = "warn"
VERDICT_FAIL = "fail"
