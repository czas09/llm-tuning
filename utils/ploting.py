import json
import math
import os
import os.path
from typing import List, Optional

from transformers.trainer import TRAINER_STATE_NAME
import matplotlib.pyplot as plt
from loguru import logger


def smooth(scalars: List[float]) -> List[float]: 
    """平滑曲线"""

    last = scalars[0]
    smoothed = list()
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)    # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_loss(
        save_dir: os.PathLike, 
        keys: Optional[List[str]] = ["loss"]
) -> None: 
    """绘制损失曲线"""

    # 读取训练器状态记录文件中的数据
    trainer_state_path = os.path.join(save_dir, TRAINER_STATE_NAME)
    with open(trainer_state_path, mode='r', encoding='utf-8') as f: 
        data = json.load(f)
    
    for key in keys: 
        steps, metrics = [], []
        for i in range(len(data["log_history"])): 
            if key in data["log_history"][i]: 
                steps.append(data["log_history"][i]["step"])
                metrics.append(data["log_history"][i][key])
        
        if not metrics: 
            logger.warning(f"No metric {key} to plot.")
            continue

        plt.figure()
        plt.plot(steps, metrics, alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), label="smoothed")
        plt.title(f"training {key} of {save_dir}")
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        save_img_path = os.path.join(save_dir, f"training_{key}.png")
        plt.savefig(save_img_path, format="png", dpi=100)
        print(f"损失曲线已导出至 {save_img_path}")
