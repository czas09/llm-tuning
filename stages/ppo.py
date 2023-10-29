# Inspired by: https://github.com/lvwerra/trl/blob/main/examples/research_projects/stack_llama/scripts/rl_training.py


import math
import os.path
from typing import Optional, List, Any, Dict, Tuple, Literal

import torch
from torch.optim import AdamW
from transformers import (
    Seq2SeqTrainingArguments, 
    DataCollatorWithPadding, 
    Trainer, TrainerState, TrainerControl, TrainerCallback, 
    GenerationConfig, PreTrainedModel, 
)
from transformers.optimization import get_scheduler
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import PPODecorators, logprobs_from_logits
from loguru import logger
from tqdm import tqdm

from hparams import DataArguments, ModelArguments, FinetuningArguments, GeneratingArguments
from dataset import load_dataset, prep_dataset
from model_loader import load_model_and_tokenizer
from utils import AverageMeter, count_parameters, get_logits_processor
from utils.callbacks import SavePeftModelCallback, LogCallback
from utils.ploting import plot_loss


def replace_model(
        model: AutoModelForCausalLMWithValueHead, 
        target: Literal["default", "reward"]
) -> None: 
    """替换模型组件，PPO 训练时模型本体还是 valuehead"""
    
    if target == "reward":     # save default head temporarily
        valuehead_state_dict: Dict[str, torch.Tensor] = model.v_head.state_dict()
        setattr(model, "default_head_weight", valuehead_state_dict["summary.weight"].detach().clone())
        setattr(model, "default_head_bias", valuehead_state_dict["summary.bias"].detach().clone())

    model.pretrained_model.set_adapter(target) # set the LoRA adapter to be active
    model.v_head.load_state_dict({
        "summary.weight": model.get_buffer("{}_head_weight".format(target)).detach().clone(),
        "summary.bias": model.get_buffer("{}_head_bias".format(target)).detach().clone()
    })


def dump_layernorm(
        model: PreTrainedModel
) -> Dict[str, torch.Tensor]: 
    """归一化层向上转换成 float32"""
    
    layer_norm_params = {}
    for name, param in model.named_parameters():
        if param.data.dtype == torch.float32:
            layer_norm_params[name] = param.data.detach().clone()
            param.data = param.data.to(model.config.torch_dtype)

    return layer_norm_params


def restore_layernorm(
        model: PreTrainedModel, 
        layernorm_params: Optional[Dict[str, torch.Tensor]] = None
) -> None: 
    
    for name, param in model.named_parameters():
        if name in layernorm_params:
            param.data = layernorm_params[name]



class TrainerForPPO(PPOTrainer, Trainer):
    r"""
    继承 transformers.Trainer 与 trl.PPOTrainer，实现了 PPO 训练功能
    """

    def __init__(
        self,
        model_args: ModelArguments,
        training_args: Seq2SeqTrainingArguments,
        finetuning_args: FinetuningArguments,
        generating_args: GeneratingArguments,
        callbacks: List[TrainerCallback],
        **kwargs
    ):
        
        PPOTrainer.__init__(self, **kwargs)
        if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
            raise ValueError("PPOTrainer is incompatible with DeepSpeed.")

        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict()
        )
        self.state = TrainerState()
        self.control = TrainerControl()
        self.log_callback, self.save_callback = callbacks[0], callbacks[1]
        assert isinstance(self.log_callback, LogCallback) and isinstance(self.save_callback, SavePeftModelCallback)

    def ppo_train(self) -> None:
        r"""
        Implements training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer.
        TODO(@zyw): 这里实现了模型训练的 epoch 循环，看了下 trl.PPOTrainer 源码中只实现了单个步骤 step 的更新？
        """

        total_train_batch_size = (
            self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
        )
        len_dataloader = len(self.dataloader)
        num_examples = len(self.dataset)
        num_train_epochs = self.args.num_train_epochs
        max_steps = math.ceil(num_train_epochs * len_dataloader)

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        if self.is_world_process_zero():
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples}")
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps}")
            logger.info(f"  Number of trainable parameters = {count_parameters(self.model)[0]}")

        unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
        dataiter = iter(self.dataloader)
        steps_trained = 0
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.log_callback.on_train_begin(self.args, self.state, self.control)

        # PPO 训练 steps 循环
        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            batch = next(dataiter)
            steps_trained += 1

            # 将模型转换至推理模式
            unwrapped_model.gradient_checkpointing_disable()
            unwrapped_model.config.use_cache = True
            self.model.eval()

            # Get inputs
            queries, responses = self.get_inputs(batch)
            self.tokenizer.padding_side = "right"    # change padding side
            # 使用给定的奖励模型计算奖励得分
            rewards = self.get_rewards(queries, responses, unwrapped_model)

            # 将模型转换至训练模式
            unwrapped_model.gradient_checkpointing_enable()
            unwrapped_model.config.use_cache = False
            self.model.train()

            # TODO(@zyw): Run PPO step
            stats = self.step(queries, responses, rewards)
            self.tokenizer.padding_side = "left"    # restore padding side
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
                    batch["response"] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, rewards)
                except:
                    logger.warning("Failed to save stats due to unknown errors.")

            self.state.global_step += 1
            self.log_callback.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step+1) % self.args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / len_dataloader, 2)
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.log_callback.on_log(self.args, self.state, self.control)
                loss_meter.reset()
                reward_meter.reset()

            if (step+1) % self.args.save_steps == 0: # save checkpoint
                self.save_model(os.path.join(
                    self.args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, self.state.global_step)
                ))
                self.save_callback.on_save(
                    self.args, self.state, self.control, model=self.accelerator.unwrap_model(self.model)
                )

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

            if steps_trained == len_dataloader:
                dataiter = iter(self.dataloader)
                steps_trained = 0

        self.log_callback.on_train_end(self.args, self.state, self.control)
        self.save_callback.on_train_end(
            self.args, self.state, self.control, model=self.accelerator.unwrap_model(self.model)
        )

    @torch.no_grad()
    def get_inputs(self, batch: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        r"""
        Generates model's responses given queries.
        """
        if self.finetuning_args.upcast_layernorm:
            layernorm_params = dump_layernorm(self.model)

        unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
        response: torch.Tensor = unwrapped_model.generate(
            generation_config=self.generation_config,
            logits_processor=get_logits_processor(),
            **batch
        )

        if self.finetuning_args.upcast_layernorm:
            restore_layernorm(self.model, layernorm_params)

        query, response = batch["input_ids"].detach().cpu(), response[:, batch["input_ids"].size(-1):].detach().cpu()
        queries, responses = [], []
        for i in range(len(query)):
            query_length = (query[i] != self.tokenizer.pad_token_id).nonzero()[0]
            response_index = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_index) == 0:
                response_length = 1                            # allow empty response
            elif self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                response_length = response_index[-1] + 2       # save the EOS token
            else:
                response_length = response_index[-1] + 1

            queries.append(query[i, query_length:])            # remove padding from left
            responses.append(response[i, :response_length])    # remove padding from right

        return queries, responses

    @torch.no_grad()
    def get_rewards(
            self,
            queries: List[torch.Tensor],
            responses: List[torch.Tensor],
            unwrapped_model: "AutoModelForCausalLMWithValueHead"
    ) -> List[torch.Tensor]:
        r"""使用给定的奖励模型计算奖励得分
        Computes scores using given reward model.
        """

        replace_model(unwrapped_model, target="reward")
        batch = self.prepare_model_inputs(queries, responses)

        with torch.cuda.amp.autocast(dtype=self.model_args.compute_dtype):    # support bf16
            _, _, values = self.model(**batch, output_hidden_states=True, return_dict=True)

        if values.size(0) != batch["input_ids"].size(0):                      # adapt to chatglm2
            values = torch.transpose(values, 0, 1)

        rewards = []
        for i in range(values.size(0)):
            end_index = batch["attention_mask"][i].nonzero()[-1]              # use the score on the EOS token
            rewards.append(values[i, end_index].float().detach().cpu())       # use fp32 type

        replace_model(unwrapped_model, target="default")
        return rewards

    @PPODecorators.empty_cuda_cache()
    def batched_forward_pass(
        self,
        model: AutoModelForCausalLMWithValueHead,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: Optional[bool] = False,
        response_masks: Optional[torch.Tensor] = None
    ):
        r"""
        Calculates model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            with torch.cuda.amp.autocast(dtype=self.model_args.compute_dtype): # support bf16
                logits, _, values = model(**input_kwargs)

            if values.size(0) != input_ids.size(0): # adapt to chatglm2
                values = torch.transpose(values, 0, 1)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0: # offset left padding
                    start += attention_mask[j, :].nonzero()[0]
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat(
                        (torch.zeros_like(query_batch[j]), response_masks_batch[j])
                    )[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if self.args.should_save:
            self._save(output_dir)


def run_ppo(
    model_args: ModelArguments, 
    data_args: DataArguments, 
    training_args: Seq2SeqTrainingArguments, 
    finetuning_args: FinetuningArguments, 
    generating_args: GeneratingArguments, 
    callbacks: Optional[List[TrainerCallback]] = None, 
) -> None: 
    """实现 PPO 训练的完整流程
    
    PPO: Proximal Policy Optimization 近端策略优化
    """

    # 加载数据集
    dataset = load_dataset(model_args, data_args)
    # 加载模型与 tokenzier
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="ppo")
    # 对数据集进行预处理
    dataset = prep_dataset(dataset, tokenizer, data_args, training_args, stage="ppo")
    # use left-padding in generation while using right-padding in training
    # TODO(@zyw): 文本生成时采用 left-padding，模型训练时采用 right-padding
    tokenizer.padding_side = "left"
    # 创建 data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # TODO(@zyw): 为 PPO 训练创建配置项
    ppo_config = PPOConfig(
        model_name=model_args.model_name_or_path,
        learning_rate=training_args.learning_rate,
        mini_batch_size=training_args.per_device_train_batch_size,
        batch_size=training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        ppo_epochs=1,
        max_grad_norm=training_args.max_grad_norm,
        seed=training_args.seed,
        optimize_cuda_cache=True,
        target=finetuning_args.ppo_target,
        log_with=finetuning_args.ppo_logger,
        use_score_scaling=finetuning_args.ppo_score_norm,
        use_score_norm=finetuning_args.ppo_score_norm,
        accelerator_kwargs={"step_scheduler_with_optimizer": False}
    )

    # 更新一系列训练参数
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=training_args.learning_rate)
    total_train_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    )
    num_training_steps = training_args.num_train_epochs * math.ceil(len(dataset) / total_train_batch_size)
    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
        num_training_steps=num_training_steps
    )

    # 初始化 PPO 训练器
    ppo_trainer = TrainerForPPO(
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        callbacks=callbacks + [SavePeftModelCallback()],
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=data_collator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )

    # PPO 训练
    if training_args.do_train: 
        # 执行训练
        ppo_trainer.ppo_train()
        # 保存模型权重
        ppo_trainer.save_model()
        # 保存训练状态记录
        ppo_trainer.save_state()    # must be called after save_model to have a folder

        if ppo_trainer.is_world_process_zero() and model_args.plot_loss: 
            # 绘制损失曲线
            plot_loss(training_args.output_dir, keys=["loss", "reward"])
