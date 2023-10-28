# llm-tuning

大型语言模型微调框架

## 代码文件说明

```
data/                 数据处理模块
    load.py
    prep.py
    utils.py
hparams/              参数定义
    data_args.py
    finetuning_args.py
    generating_args.py
    model_args.py
stages/               训练阶段：SFT、RM、PPO、DPO
    dop.py
    ppo.py
    pt.py
    sft.py                实现对话微调阶段的训练流程
    rm.py
    sft.py
utils/                工具脚本
    callbacks.py
    constants.py
    ploting.py
    utils.py
argument_parser.py    参数解析模块
metrics.py            评估指标
model_loader.py       模型加载模块
train.py              （主入口）训练启动脚本
```

## 使用方法

TODO