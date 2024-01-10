"""指令微调 / 对话微调

SFT: Supervised Fine-Tuning
"""


from __future__ import annotations

from dataclasses import dataclass
import json
import os.path
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union

import torch
import torch.nn as nn
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import TrainerCallback
from transformers import PreTrainedTokenizer
from transformers.trainer import PredictionOutput
import numpy as np
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from loguru import logger

from model_loader import load_model_and_tokenizer
from dataset import load_dataset, prep_dataset, split_dataset
from hparams import (
    ModelArguments, 
    DataArguments, 
    FinetuningArguments, 
    GeneratingArguments, 
)
from utils import get_logits_processor, plot_loss
from utils.constants import IGNORE_INDEX


class TrainerForSFT(Seq2SeqTrainer): 
    """针对对话微调阶段的训练器
    
    TODO(@zyw): 在 transformers.Seq2SeqTrainer 上增加模型预测相关设施
    """

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None
        
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:    # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]
                # inputs["input_ids"] = self._pad_tensors_to_target_len(inputs["input_ids"], inputs["labels"])
                # if "attention_mask" in inputs:
                #     inputs["attention_mask"] = self._pad_tensors_to_target_len(
                #         inputs["attention_mask"], inputs["labels"], pad_token_id=0
                #     )
                # if "position_ids" in inputs:
                #     inputs["position_ids"] = self._pad_tensors_to_target_len(
                #         inputs["position_ids"], inputs["labels"], pad_token_id=0
                #     )

        # ignore the returned labels (may be truncated)
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :max(prompt_len, label_len)] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(
        self,
        src_tensor: torch.Tensor,
        tgt_tensor: torch.Tensor,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        # pad_token_id = pad_token_id if pad_token_id is not None else self.tokenizer.pad_token_id
        # padded_tensor = pad_token_id * torch.ones_like(tgt_tensor)
        # padded_tensor[:, -src_tensor.shape[-1]:] = src_tensor # adopt left-padding
        # return padded_tensor.contiguous() # in contiguous memory
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1]:] = src_tensor # adopt left-padding
        return padded_tensor.contiguous() # in contiguous memory

    def save_predictions(
        self,
        predict_results: PredictionOutput
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)
        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id)
        # labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate((preds[i][pad_len[0]:], preds[i][:pad_len[0]]), axis=-1) # move pad token to last

        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            # for pred, label in zip(decoded_preds, decoded_labels):
            for label, pred in zip(decoded_labels, decoded_preds):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))


@dataclass
class ComputeMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.
    TODO(@zyw)
    """

    tokenizer: PreTrainedTokenizer

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds
        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        return {k: float(np.mean(v)) for k, v in score_dict.items()}


def train_sft(
        model_args: ModelArguments, 
        data_args: DataArguments, 
        training_args: Seq2SeqTrainingArguments, 
        finetuning_args: FinetuningArguments, 
        generating_args: GeneratingArguments, 
        callbacks: Optional[List[TrainerCallback]] = None, 
): 
    """实现对话微调阶段的完整流程
    
    - 加载数据集
    - 加载模型与 tokenizer
    - 预处理数据集
    - 创建 data collator
    - 配置训练参数 training_args
    - 创建训练器 TrainerForSFT
    - 执行模型训练
    - 保存权重以及训练状态
    """
    
    # 加载数据集
    dataset = load_dataset(model_args, data_args)
    # 加载模型与 tokenzier
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage='sft')
    # 对数据集进行预处理
    dataset = prep_dataset(dataset, tokenizer, data_args, training_args, stage='sft')
    # 在文本生成模式下使用左侧填充 left-padding
    if training_args.predict_with_generate: 
        tokenizer.padding_side = "left"
    # 创建 data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        pad_to_multiple_of=8,    # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id, 
    )

    # 更新训练配置参数
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(
        generation_max_length=training_args.generation_max_length or data_args.cutoff_len, 
        generate_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
    ))
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # 初始化训练器
    trainer = TrainerForSFT(
        model=model, 
        args=training_args, 
        data_collator=data_collator, 
        tokenizer=tokenizer, 
        callbacks=callbacks, 
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None, 
        **split_dataset(dataset, data_args, training_args), 
    )

    # 更新文本生成参数
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = [tokenizer.pad_token_id]
    gen_kwargs["logits_processor"] = get_logits_processor()

    # ==========================================================================
    # 模型训练
    # ==========================================================================
    if training_args.do_train: 
        # 执行训练流程
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

        # 保存训练好的模型权重和 tokenizer
        trainer.save_model()
        # 记录评估指标结果
        trainer.log_metrics("train", train_result.metrics)
        # 保存评估指标结果
        trainer.save_metrics("train", train_result.metrics)
        # 保存训练过程产生的记录文件
        trainer.save_state()
        
        if trainer.is_world_process_zero() and model_args.plot_loss: 
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])
    
    # ==========================================================================
    # 模型评估
    # ==========================================================================
    if training_args.do_eval: 
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:    # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # ==========================================================================
    # 模型预测
    # ==========================================================================
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:    # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)
