"""G4 专用 eval config：在 MolmoBot 原 config 基础上强制 end_on_success=True。

为什么不直接改 MolmoBot 源码：`ttt_data_validation/` 的约定是只读引用
MolmoBot/GC-TTT 源码，不修改它们。这里用子类覆盖字段，是唯一既不改源码
又能让 `run_eval.py` 的 isinstance(policy_config, SynthVLAPolicyConfig)
校验通过的方式（子类原样继承父类的 policy_config 默认值）。

用法：
    PYTHONPATH=<gates 目录> python launch_scripts/run_eval.py \
        --eval_config_cls g4_eval_configs:G4MolmoBotRBY1PickPnPEvalConfig ...
"""

from __future__ import annotations

from olmo.eval.configure_molmo_spaces import MolmoBotRBY1PickPnPEvalConfig


class G4MolmoBotRBY1PickPnPEvalConfig(MolmoBotRBY1PickPnPEvalConfig):
    """与原 config 完全一致，只把 end_on_success 打开。

    原 config 链路继承自 molmo_spaces 的 MlSpacesExpConfig，其中
    end_on_success 默认 False（molmo_spaces/configs/abstract_exp_config.py:53）。
    为 False 时 rollout 跑满 task_horizon，judge_success() 只在结束后用末态
    判定一次，过程中已成功的会被判为失败——历史同 benchmark 的 E2 run 因此
    低估 2.1 倍（2.67% vs 5.61%）。TTT 最可能的收益形式是「更早成功」，
    所以这个开关必须为 True。
    """

    end_on_success: bool = True
