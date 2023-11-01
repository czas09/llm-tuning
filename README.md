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
configs/               训练参数配置文件
    ...
dataset/               数据处理模块
    load.py
    prep.py
    utils.py
hparams/               参数定义
    data_args.py
    finetuning_args.py
    generating_args.py
    model_args.py
patches/               训练加速、长程建模等 (WIP)
    ...
stages/                训练阶段：SFT、RM、PPO、DPO
    pt.py                  （继续）预训练 (WIP)
    sft.py                 实现对话微调阶段的完整训练流程
    rm.py                  实现奖励模型 Reward Model 的完整训练流程
    ppo.py                 实现 PPO 训练的完整流程
    dpo.py                 DPO (WIP)
utils/                 工具脚本
    callbacks.py
    constants.py
    model.py
    ploting.py
    training.py
argument_parser.py     参数解析模块
model_loader.py        模型加载模块
prompt_templates.py    对话模板
train.py               （主入口）训练启动脚本
```

## 使用方法

### 模型微调

填好相关配置文件 (WIP) ，启动 train.py

### 常用参数说明

参见文档：
`./docs/常见参数说明.md` (WIP)

### 分布式部署

分布式训练工具：
* accelerate
* deepspeed
* bitsandbytes

参见文档：
`./docs/分布式训练操作方法.md` (WIP)

### 模型评估 (WIP)

## 待办事项

* 完善微调相关代码
* 参数配置文件
* 数据集
* 模型评估
* 不同模型和训练方法下显卡需求量说明