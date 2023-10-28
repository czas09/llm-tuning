"""（主入口）训练启动脚本"""


from typing import Any, Dict, List, Optional

from transformers import TrainerCallback

from stages import (
    train_pt, 
    train_sft, 
    train_rm, 
    train_ppo, 
    train_dpo, 
)
from argument_parser import get_train_args
from utils.callbacks import LogCallback


def main(
        args: Optional[Dict[str, Any]] = None, 
        callbacks: Optional[List["TrainerCallback"]] = None
): 
    
    # 获取训练参数
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    # 设置回调函数
    callbacks = [LogCallback()] if callbacks is None else callbacks

    # ==========================================================================
    # 训练阶段
    # ==========================================================================

    # Pre-Train 预训练
    # if finetuning_args.stage == "pt":
    #     run_pt(model_args, data_args, training_args, finetuning_args, callbacks)

    # SFT 指令微调 / 对话微调
    if finetuning_args.stage == "sft": 
        train_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)

    # Reward Model 奖励模型
    elif finetuning_args.stage == "rm": 
        train_rm(model_args, data_args, training_args, finetuning_args, callbacks)

    # PPO
    elif finetuning_args.stage == "ppo": 
        train_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)

    # DPO
    elif finetuning_args.stage == "dpo": 
        train_dpo(model_args, data_args, training_args, finetuning_args, callbacks)

    else:
        raise ValueError(f"目前尚未支持 {finetuning_args.stage} 训练阶段！请输入下列关键词之一：[pt, sft, rm, ppo, dpo]")


if __name__ == '__main__': 

    main()