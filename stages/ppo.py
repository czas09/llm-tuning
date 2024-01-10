# Inspired by: https://github.com/lvwerra/trl/blob/main/examples/research_projects/stack_llama/scripts/rl_training.py


import json
import math
import os.path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

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
import requests
from loguru import logger
from tqdm import tqdm

from hparams import DataArguments, ModelArguments, FinetuningArguments, GeneratingArguments
from dataset import load_dataset, prep_dataset
from model_loader import load_model_and_tokenizer
from utils import AverageMeter, count_parameters, get_logits_processor, load_valuehead_params
from utils.callbacks import SavePeftModelCallback, LogCallback
from utils.ploting import plot_loss


def get_rewards_from_server(server_url: str, messages: List[str]) -> List[torch.Tensor]:
    headers = {"Content-Type": "application/json"}
    payload = {"model": "model", "messages": messages}
    response = requests.post(server_url, json=payload, headers=headers)
    rewards = json.loads(response.text)["scores"]
    return torch.Tensor(rewards)


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
        reward_model: AutoModelForCausalLMWithValueHead,
        **kwargs
    ):
        
        PPOTrainer.__init__(self, **kwargs)
        if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
            raise ValueError("PPOTrainer is incompatible with DeepSpeed.")

        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.reward_model = reward_model

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict()
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = self.accelerator.distributed_type == "DEEPSPEED" and hasattr(
            self.accelerator.state, "deepspeed_plugin"
        )
        self.log_callback, self.save_callback = callbacks[0], callbacks[1]
        assert isinstance(self.log_callback, LogCallback) and isinstance(self.save_callback, SavePeftModelCallback)

        if self.args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        if finetuning_args.reward_model_type == "full":
            if self.is_deepspeed_enabled:
                if not (
                    getattr(reward_model.pretrained_model, "is_loaded_in_8bit", False)
                    or getattr(reward_model.pretrained_model, "is_loaded_in_4bit", False)
                ): # quantized models are already set on the correct device
                    self.reward_model = self._prepare_deepspeed(self.reward_model)
            else:
                self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""
        Implements training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer.
        TODO(@zyw): 这里实现了模型训练的 epoch 循环，看了下 trl.PPOTrainer 源码中只实现了单个步骤 step 的更新？
        """

        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        if self.is_world_process_zero():
            logger.info("***** Running training *****")
            logger.info("  Num examples = {}".format(num_examples))
            logger.info("  Num Epochs = {}".format(num_train_epochs))
            logger.info("  Instantaneous batch size per device = {}".format(self.args.per_device_train_batch_size))
            logger.info("  Total train batch size (w. parallel, buffer, distributed & accumulation) = {}".format(
                total_train_batch_size
            ))
            logger.info("  Gradient Accumulation steps = {}".format(self.args.gradient_accumulation_steps))
            logger.info("  Num optimization epochs per batch = {}".format(self.finetuning_args.ppo_epochs))
            logger.info("  Total training steps = {}".format(max_steps))
            logger.info("  Number of trainable parameters = {}".format(count_parameters(self.model)[0]))

        unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.log_callback.on_train_begin(self.args, self.state, self.control)

        # PPO 训练 steps 循环
        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)

            # 将模型转换至推理模式
            unwrapped_model.gradient_checkpointing_disable()
            unwrapped_model.config.use_cache = True
            self.model.eval()

            # Get inputs
            self.tokenizer.padding_side = "right" # change padding side
            queries, responses, rewards = [], [], []
            for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
                mini_batch_queries, mini_batch_responses = self.get_inputs(batch[idx:idx+self.config.mini_batch_size])
                # 使用给定的奖励模型计算奖励得分
                mini_batch_rewards = self.get_rewards(mini_batch_queries, mini_batch_responses, unwrapped_model)
                queries.extend(mini_batch_queries)
                responses.extend(mini_batch_responses)
                rewards.extend(mini_batch_rewards)

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
                    epoch=round(step / steps_in_epoch, 2)
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

            # if steps_trained == len_dataloader:
            #     dataiter = iter(self.dataloader)
            #     steps_trained = 0

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
        generate_output: torch.Tensor = unwrapped_model.generate(
            generation_config=self.generation_config,
            logits_processor=get_logits_processor(),
            **batch
        )

        if self.finetuning_args.upcast_layernorm:
            restore_layernorm(self.model, layernorm_params)

        query = batch["input_ids"].detach().cpu()
        response = generate_output[:, batch["input_ids"].size(-1):].detach().cpu()
        queries, responses = [], []
        for i in range(len(query)):
            query_length = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            response_index = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_index) == 0:
                response_length = 1 # allow empty response
            else:
                response_length = response_index[-1].item() + 1

            queries.append(query[i, query_length:]) # remove padding from left
            responses.append(response[i, :response_length]) # remove padding from right

        return queries, responses

    @torch.no_grad()
    def get_rewards(
            self,
            queries: List[torch.Tensor],
            responses: List[torch.Tensor],
            unwrapped_model: AutoModelForCausalLMWithValueHead
    ) -> List[torch.Tensor]:
        r"""使用给定的奖励模型计算奖励得分
        Computes scores using given reward model.
        Both inputs and outputs are put on CPU.
        """

        if self.finetuning_args.reward_model_type == "api":
            token_ids = [torch.cat((q, r), dim=-1).tolist() for q, r in zip(queries, responses)]
            messages = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            return get_rewards_from_server(self.reward_model, messages)

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="reward")
            reward_model = self.model
        else:
            reward_model = self.reward_model

        batch = self.prepare_model_inputs(queries, responses)

        with torch.cuda.amp.autocast(dtype=self.model_args.compute_dtype): # support bf16
            _, _, values = reward_model(**batch, output_hidden_states=True, return_dict=True)

        if getattr(unwrapped_model.config, "model_type", None) == "chatglm": # assume same architecture
            values = torch.transpose(values, 0, 1)

        rewards = []
        for i in range(values.size(0)):
            end_indexes = (batch["input_ids"][i] != self.tokenizer.pad_token_id).nonzero()
            end_index = end_indexes[-1].item() if len(end_indexes) else 0
            rewards.append(values[i, end_index].float().detach().cpu()) # use fp32 type

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="default")

        return rewards

    @PPODecorators.empty_device_cache()
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

            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            if getattr(unwrapped_model.config, "model_type", None) == "chatglm":
                values = torch.transpose(values, 0, 1)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0: # offset left padding
                    start += attention_mask[j, :].nonzero()[0].item()
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
            try:
                self._save(output_dir, state_dict=self.accelerator.get_state_dict(self.model))
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                self._save(output_dir, state_dict={})
                for filename in ["pytorch_model.bin", "pytorch_model.bin.index.json"]: # remove dummy checkpoint
                    file = os.path.join(output_dir, filename)
                    if os.path.isfile(file):
                        os.remove(file)

                self.model.save_checkpoint(output_dir) # wrapped model


def create_ref_model(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    add_valuehead: Optional[bool] = False
) -> Union["PreTrainedModel", "AutoModelForCausalLMWithValueHead"]:
    r"""
    Creates reference model for PPO/DPO training. Evaluation mode is not supported.

    The valuehead parameter is randomly initialized since it is useless for PPO training.
    """
    if finetuning_args.ref_model is not None:
        ref_model_args_dict = model_args.to_dict()
        ref_model_args_dict.update(dict(
            model_name_or_path=finetuning_args.ref_model,
            checkpoint_dir=finetuning_args.ref_model_checkpoint,
            quantization_bit=finetuning_args.ref_model_quantization_bit
        ))
        ref_model_args = ModelArguments(**ref_model_args_dict)
        ref_finetuning_args = FinetuningArguments(finetuning_type="lora")
        ref_model, _ = load_model_and_tokenizer(
            ref_model_args, ref_finetuning_args, is_trainable=False, add_valuehead=add_valuehead
        )
        logger.info("Created reference model from {}".format(finetuning_args.ref_model))
    else:
        if finetuning_args.finetuning_type == "lora":
            ref_model = None
        else:
            ref_model, _ = load_model_and_tokenizer(
                model_args, finetuning_args, is_trainable=False, add_valuehead=add_valuehead
            )
            logger.info("Created reference model from the model itself.")

    return ref_model


def create_reward_model(
    model: "AutoModelForCausalLMWithValueHead",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments"
) -> "AutoModelForCausalLMWithValueHead":
    r"""
    Creates reward model for PPO training.
    """
    if finetuning_args.reward_model_type == "api":
        assert finetuning_args.reward_model.startswith("http"), "Please provide full url."
        logger.info("Use reward server {}".format(finetuning_args.reward_model))
        return finetuning_args.reward_model
    elif finetuning_args.reward_model_type == "lora":
        model.pretrained_model.load_adapter(finetuning_args.reward_model, "reward")
        for name, param in model.named_parameters(): # https://github.com/huggingface/peft/issues/1090
            if "default" in name:
                param.data = param.data.to(torch.float32) # trainable params should in fp32
        vhead_params = load_valuehead_params(finetuning_args.reward_model, model_args)
        assert vhead_params is not None, "Reward model is not correctly loaded."
        model.register_buffer("reward_head_weight", vhead_params["v_head.summary.weight"], persistent=False)
        model.register_buffer("reward_head_bias", vhead_params["v_head.summary.bias"], persistent=False)
        model.register_buffer("default_head_weight", torch.zeros_like(vhead_params["v_head.summary.weight"]), persistent=False)
        model.register_buffer("default_head_bias", torch.zeros_like(vhead_params["v_head.summary.bias"]), persistent=False)
        logger.info("Loaded adapter weights of reward model from {}".format(finetuning_args.reward_model))
        return None
    else:
        reward_model_args_dict = model_args.to_dict()
        reward_model_args_dict.update(dict(
            model_name_or_path=finetuning_args.reward_model,
            checkpoint_dir=finetuning_args.reward_model_checkpoint,
            quantization_bit=finetuning_args.reward_model_quantization_bit
        ))
        reward_model_args = ModelArguments(**reward_model_args_dict)
        reward_finetuning_args = FinetuningArguments(finetuning_type="lora")
        reward_model, _ = load_model_and_tokenizer(
            reward_model_args, reward_finetuning_args, is_trainable=False, add_valuehead=True
        )
        logger.info("Loaded full weights of reward model from {}".format(finetuning_args.reward_model))
        logger.warning("Please ensure the ppo model and reward model share SAME tokenizer and vocabulary.")
        return reward_model



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

    # Create reference model and reward model
    ref_model = create_ref_model(model_args, finetuning_args, add_valuehead=True)
    reward_model = create_reward_model(model, model_args, finetuning_args)

    # TODO(@zyw): 为 PPO 训练创建配置项
    backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    ppo_config = PPOConfig(
        model_name=model_args.model_name_or_path,
        learning_rate=training_args.learning_rate,
        mini_batch_size=training_args.per_device_train_batch_size,
        batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        ppo_epochs=finetuning_args.ppo_epochs,
        max_grad_norm=training_args.max_grad_norm,
        seed=training_args.seed,
        optimize_device_cache=True,
        target=finetuning_args.ppo_target,
        log_with=finetuning_args.ppo_logger,
        use_score_scaling=finetuning_args.ppo_score_norm,
        use_score_norm=finetuning_args.ppo_score_norm,
        whiten_rewards=finetuning_args.ppo_whiten_rewards,
        accelerator_kwargs={"step_scheduler_with_optimizer": False}
    )

    # 更新一系列训练参数
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=training_args.learning_rate)
    if training_args.max_steps > 0:
        num_training_steps = training_args.max_steps
    else:
        total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
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
        reward_model=reward_model,
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=data_collator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )

    # PPO 训练
    if training_args.do_train: 
        # 执行训练
        ppo_trainer.ppo_train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        # 保存模型权重
        ppo_trainer.save_model()
        # 保存训练状态记录
        ppo_trainer.save_state()    # must be called after save_model to have a folder

        if ppo_trainer.is_world_process_zero() and model_args.plot_loss: 
            # 绘制损失曲线
            plot_loss(training_args.output_dir, keys=["loss", "reward"])
