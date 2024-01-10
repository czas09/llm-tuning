# Inspired by:
# https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/train_reward_model_gptj.py


from dataclasses import dataclass
import json
import os.path
from typing import Dict, Sequence, Tuple, Union, Optional, List, Any

import torch
from transformers import (
    Seq2SeqTrainingArguments, 
    DataCollatorWithPadding, 
    Trainer, 
    TrainerCallback, 
)
from transformers.trainer import PredictionOutput
from transformers.modeling_utils import PreTrainedModel
import numpy as np
from loguru import logger

from hparams import ModelArguments, DataArguments, FinetuningArguments
from dataset import load_dataset, prep_dataset, split_dataset
from utils.callbacks import SavePeftModelCallback
from utils.ploting import plot_loss
from model_loader import load_model_and_tokenizer


# TODO(@zyw)
@dataclass
class DataCollatorForPairwiseData(DataCollatorWithPadding):
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        将批次中的数据都填充至本批次中的最大长度

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        批次中的数据有 2n 条，其中前 n 条是选中数据，后 n 条是拒绝数据
        """
        features = [
            {
                "input_ids": feature["prompt_ids"] + feature[key],
                "attention_mask": [1] * (len(feature["prompt_ids"]) + len(feature[key]))
            }
            for key in ("chosen_ids", "rejected_ids") for feature in features
        ]
        return super().__call__(features)


class TrainerForRewardModel(Trainer):
    r"""
    Inherits PeftTrainer to compute pairwise loss.
    主要是改变 transformers.Trainer 的损失值计算方法，这里需要为奖励模型计算 pairwise loss
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_return_loss = True     # override property to return eval_loss

    def compute_loss(
            self,
            model: PreTrainedModel,
            inputs: Dict[str, torch.Tensor],
            return_outputs: Optional[bool] = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        r"""
        Computes pairwise loss. The first n examples are chosen and the last n examples are rejected.
        前 n 条是选中数据，后 n 条是拒绝数据

        Note that the first element will be removed from the output tuple. 
        See: https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/trainer.py#L3509
        """

        # 计算奖励值
        _, _, values = model(**inputs, output_hidden_states=True, return_dict=True)
        unwrapped_model: "PreTrainedModel" = self.accelerator.unwrap_model(self.model)
        if getattr(unwrapped_model.config, "model_type", None) == "chatglm":
            values = torch.transpose(values, 0, 1)

        # 将 input ids 和计算得到的奖励值划分为 chosen 和 rejected 两部分
        batch_size = inputs["input_ids"].size(0) // 2
        chosen_input_ids, rejected_input_ids = inputs["input_ids"][:batch_size], inputs["input_ids"][batch_size:]
        chosen_attn_mask, rejected_attn_mask = (
            inputs["attention_mask"][:batch_size], inputs["attention_mask"][batch_size:]
        )
        chosen_rewards, rejected_rewards = values[:batch_size], values[batch_size:]
        chosen_scores, rejected_scores = [], []

        # Compute pairwise loss. Only backprop on the different tokens before padding
        # 计算 pairwise loss
        # Inspired by: https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
        loss = 0
        for i in range(batch_size):
            chosen_length = chosen_attn_mask[i].nonzero()[-1] + 1
            rejected_length = rejected_attn_mask[i].nonzero()[-1] + 1
            check_divergence = (chosen_input_ids[i] != rejected_input_ids[i]).nonzero()

            if len(check_divergence) == 0:
                end_index = chosen_length
                div_index = end_index - 1
            else:
                end_index = max(chosen_length, rejected_length)
                div_index = check_divergence[0]

            assert div_index > 0
            chosen_trunc_rewards = chosen_rewards[i, div_index:end_index]
            rejected_trunc_rewards = rejected_rewards[i, div_index:end_index]
            if return_outputs:    # use the score on the EOS token for inference
                chosen_scores.append(chosen_rewards[i, chosen_length-1])
                rejected_scores.append(rejected_rewards[i, rejected_length-1])
            # 累积损失值
            loss += -torch.nn.functional.logsigmoid(chosen_trunc_rewards - rejected_trunc_rewards).mean()

        loss = loss / batch_size
        if return_outputs:
            chosen_scores, rejected_scores = torch.stack(chosen_scores), torch.stack(rejected_scores)
            return loss, [loss, chosen_scores, rejected_scores]

        return loss

    def save_predictions(
        self,
        predict_results: "PredictionOutput"
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        chosen_scores, rejected_scores = predict_results.predictions

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for c_score, r_score in zip(chosen_scores, rejected_scores):
                res.append(json.dumps({"chosen": round(float(c_score), 2), "rejected": round(float(r_score), 2)}))
            writer.write("\n".join(res))


def compute_accuracy(eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]: 
    """计算准确率指标"""

    preds, _ = eval_preds
    return dict(
        accuracy=(preds[0] > preds[1]).sum() / len(preds[0])
    )


def run_rm(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: Seq2SeqTrainingArguments,
    finetuning_args: FinetuningArguments,
    callbacks: Optional[List[TrainerCallback]] = None
) -> None: 
    """实现奖励模型 Reward Model 的完整训练流程"""

    # 加载数据集
    dataset = load_dataset(model_args, data_args)
    # 加载模型与 tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="rm")
    # 预处理数据集
    dataset = prep_dataset(dataset, tokenizer, data_args, training_args, stage="rm")
    # 创建 data collator
    data_collator = DataCollatorForPairwiseData(tokenizer, pad_to_multiple_of=8)

    # 配置训练参数
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(remove_unused_columns=False)) # important for pairwise dataset
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # 初始化训练器
    trainer = TrainerForRewardModel(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks + [SavePeftModelCallback()],
        compute_metrics=compute_accuracy,
        **split_dataset(dataset, data_args, training_args)
    )

    # 模型训练
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # 模型评估
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 模型预测
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict")
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)
