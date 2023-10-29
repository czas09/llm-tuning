# llm-tuning

大型语言模型微调框架

## 支持训练方式

* 模型种类 待整理
  * ChatGLM2-6B
  * Baichuan
  * InternLM
  * Qwen
* 训练方式
  * Full
  * Freeze
  * (Q)LoRA
* 训练阶段 stages
  * Pretrain (WIP)
  * SFT
  * Reward Model (RM)
  * PPO
  * DPO (WIP)

## 代码文件说明

```
configs/              训练参数配置文件
    ...
dataset/              数据处理模块
    load.py
    prep.py
    utils.py
hparams/              参数定义
    data_args.py
    finetuning_args.py
    generating_args.py
    model_args.py
patches/              训练加速、长程建模等 (WIP)
    ...
stages/               训练阶段：SFT、RM、PPO、DPO
    dpo.py                DPO (WIP)
    ppo.py                实现 PPO 训练的完整流程
    pt.py                 预训练 (WIP)
    sft.py                实现对话微调阶段的完整训练流程
    rm.py                 实现奖励模型 Reward Model 的完整训练流程
    sft.py
utils/                工具脚本
    callbacks.py
    constants.py
    ploting.py
    utils.py
argument_parser.py    参数解析模块
model_loader.py       模型加载模块
train.py              （主入口）训练启动脚本
```

## 使用方法

TODO

## 待办事项

* 数据集
* 模型评估