"""实现模型加载相关的流程

- init_adapter:                初始化模型适配器，用于 LoRA 训练方法
- load_model_and_tokenizer:    加载模型与 tokenizer
"""


import math
import os
from types import MethodType
from typing import Literal, Optional, Tuple

import torch
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    PretrainedConfig, 
    PreTrainedModel, 
    PreTrainedTokenizerBase, 
    PreTrainedTokenizer, 
)
from transformers.models.llama import modeling_llama as LlamaModule
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from peft import (
    PeftModel,
    TaskType,
    LoraConfig,
    get_peft_model
)
from trl import AutoModelForCausalLMWithValueHead
from loguru import logger

try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError:    # https://github.com/huggingface/transformers/releases/tag/v4.33.1
    from transformers.deepspeed import is_deepspeed_zero3_enabled

from hparams import ModelArguments, FinetuningArguments
from utils import (
    count_parameters, 
    infer_optim_dtype, 
    prepare_model_for_training, 
    find_all_linear_modules, 
    load_valuehead_params, 
)
from patches import llama_patch as LlamaPatches


# 校验依赖库版本
check_min_version("4.31.0")
require_version("datasets>=2.12.0", "To fix: pip install datasets>=2.12.0")
require_version("accelerate>=0.21.0", "To fix: pip install accelerate>=0.21.0")
require_version("peft>=0.4.0", "To fix: pip install peft>=0.4.0")
require_version("trl>=0.7.2", "To fix: pip install trl>=0.7.2")


def init_adapter(
    model: PreTrainedModel,
    model_args: ModelArguments,
    finetuning_args: FinetuningArguments,
    is_trainable: bool,
    is_mergeable: bool
) -> PreTrainedModel:
    r"""初始化模型适配器 model adapters，用于参数高效训练方法

    Support full-parameter, freeze and LoRA training.

    注意这里的可训练参数必须转换到 float32 类型
    """

    if finetuning_args.finetuning_type == "none" and is_trainable:
        raise ValueError("You cannot use finetuning_type=none while training.")

    # ==========================================================================
    # 采用全量参数训练方法
    # ==========================================================================
    elif finetuning_args.finetuning_type == "full" and is_trainable:
        logger.info("采用全量参数训练方法")
        model = model.float()    # TODO(@zyw)

    # ==========================================================================
    # 采用 Freeze 训练方法
    # ==========================================================================
    elif finetuning_args.finetuning_type == "freeze":
        logger.info("采用 Freeze 训练方法")
        num_layers = getattr(model.config, "num_layers")
        if finetuning_args.num_layer_trainable > 0:    # fine-tuning the last n layers if num_layer_trainable > 0
            trainable_layer_ids = [num_layers - k - 1 for k in range(finetuning_args.num_layer_trainable)]
        else:                                          # fine-tuning the first n layers if num_layer_trainable < 0
            trainable_layer_ids = [k for k in range(-finetuning_args.num_layer_trainable)]

        trainable_layers = ["{:d}.{}".format(idx, finetuning_args.name_module_trainable) for idx in trainable_layer_ids]
        for name, param in model.named_parameters():
            if not any(trainable_layer in name for trainable_layer in trainable_layers): 
                # 冻结这些层可训练参数
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32)

    # ==========================================================================
    # 采用 (Q)LoRA 训练方法
    # ==========================================================================
    elif finetuning_args.finetuning_type == "lora": 
        logger.info("采用 LoRA 训练方法")
        latest_checkpoint = None

        if model_args.checkpoint_dir is not None:
            if (is_trainable and finetuning_args.resume_lora_training) or (not is_mergeable):    # 继续微调
                checkpoints_to_merge, latest_checkpoint = model_args.checkpoint_dir[:-1], model_args.checkpoint_dir[-1]
            else: 
                checkpoints_to_merge = model_args.checkpoint_dir

            # TODO(@zyw)
            for checkpoint in checkpoints_to_merge:
                model = PeftModel.from_pretrained(model, checkpoint)
                model = model.merge_and_unload()

            if len(checkpoints_to_merge) > 0: 
                logger.info("Merged {} model checkpoint(s).".format(len(checkpoints_to_merge)))

            if latest_checkpoint is not None:     # resume lora training or quantized inference
                model = PeftModel.from_pretrained(model, latest_checkpoint, is_trainable=is_trainable)

        if is_trainable and latest_checkpoint is None:    # 训练新的 LoRA 权重
            # 确定 LoRA 训练位置
            if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":    # 对所有的 Linear 层训练 LoRA 权重
                target_modules = find_all_linear_modules(model, model_args.quantization_bit)
            else:                                                                                    # 对指定层训练 LoRA 权重
                target_modules = finetuning_args.lora_target

            # 创建 LoRA 配置项
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=target_modules,
                modules_to_save=finetuning_args.additional_target
            )
            # 给模型加上 LoRA 部分
            model = get_peft_model(model, lora_config)
            
            if id(model.peft_config) != id(model.base_model.peft_config): # https://github.com/huggingface/peft/issues/923
                model.base_model.peft_config = model.peft_config

    if model_args.checkpoint_dir is not None:
        logger.info("Loaded fine-tuned model from checkpoint(s): {}".format(",".join(model_args.checkpoint_dir)))

    return model


def load_model_and_tokenizer(
    model_args: ModelArguments,
    finetuning_args: FinetuningArguments,
    is_trainable: Optional[bool] = False,    # is_trainable = training_args.do_train
    stage: Optional[Literal["pt", "sft", "rm", "ppo"]] = "sft"
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]: 
    """
    加载预训练模型权重以及 tokenizer

    Support both training and inference.
    """

    if (not is_trainable) and model_args.checkpoint_dir is None:
        logger.warning("Checkpoint is not found at evaluation, load the original model.")
        finetuning_args = FinetuningArguments(finetuning_type="none")

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="right",    # training with left-padded tensors in fp16 precision may cause overflow
        **config_kwargs
    )

    if finetuning_args.finetuning_type != "lora" and model_args.checkpoint_dir is not None:
        model_to_load = model_args.checkpoint_dir[0]
    else:
        model_to_load = model_args.model_name_or_path

    config = AutoConfig.from_pretrained(model_to_load, **config_kwargs)

    # 修复 ChatGLM2-6B 模型的 tokenizer
    # TODO(@zyw)
    if getattr(config, "model_type", None) == "chatglm":
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    # 设置模型的数据类型
    if model_args.compute_dtype is not None:    # 训练阶段
        setattr(config, "torch_dtype", model_args.compute_dtype)
    else:                                       # 推理阶段，priority: bf16 > fp16 > fp32
        model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    # 修复 Qwen 模型的配置项
    # 修复前：bf16, fp16, fp32 = false, false, false
    # 修复后：bf16, fp16, fp32 = true, true, true
    if getattr(config, "model_type", None) == "qwen":
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, getattr(config, "torch_dtype", None) == dtype)

    # TODO(@zyw): 设置 RoPE scaling，用于超长程文本生成
    if model_args.rope_scaling is not None: 
        if hasattr(config, "use_dynamic_ntk"):    # 针对 Qwen 模型
            if is_trainable: 
                logger.warning("Qwen model does not support RoPE scaling in training.")
            else:
                setattr(config, "use_dynamic_ntk", True)
                setattr(config, "use_logn_attn", True)
                logger.info("Using dynamic NTK scaling.")

        elif hasattr(config, "rope_scaling"):     # 针对 LLaMA and Falcon 模型
            if is_trainable:
                if model_args.rope_scaling == "dynamic":
                    logger.warning(
                        "Dynamic NTK may not work well with fine-tuning. "
                        "See: https://github.com/huggingface/transformers/pull/24653"
                    )
                current_max_length = getattr(config, "max_position_embeddings", None)
                if current_max_length and model_args.model_max_length > current_max_length:
                    scaling_factor = float(math.ceil(model_args.model_max_length / current_max_length))
                else:
                    logger.warning("Input length is smaller than max length. Consider increase input length.")
                    scaling_factor = 1.0
            else:
                scaling_factor = 2.0

            setattr(config, "rope_scaling", {"type": model_args.rope_scaling, "factor": scaling_factor})
            logger.info("Using {} scaling strategy and setting scaling factor to {}".format(
                model_args.rope_scaling, scaling_factor
            ))

        else:
            logger.warning("Current model does not support RoPE scaling.")

    # TODO(@zyw): 设置 FlashAttention-2，用于训练加速与推理加速
    if model_args.flash_attn: 
        if getattr(config, "model_type", None) == "llama": 
            # TODO(@zyw)
            LlamaModule.LlamaAttention = LlamaPatches.LlamaFlashAttention2
            LlamaModule.LlamaModel._prepare_decoder_attention_mask = LlamaPatches._prepare_decoder_attention_mask
            logger.info("Using FlashAttention-2 for faster training and inference.")

        elif getattr(config, "model_type", None) == "qwen": 
            logger.info("Qwen 模型原生支持 FlashAttention，无需额外设置。")

        else: 
            logger.warning("Current model does not support FlashAttention-2.")

    elif is_trainable and model_args.shift_attn and getattr(config, "model_type", None) == "llama":
        LlamaModule.LlamaAttention = LlamaPatches.LlamaShiftShortAttention
        logger.warning("Using `--flash_attn` for faster training in large context length.")

    # TODO(@zyw): 设置 shift short attention (S^2-Attn)
    if is_trainable and model_args.shift_attn:
        if getattr(config, "model_type", None) == "llama":
            setattr(config, "group_size_ratio", 0.25)
            logger.info("Using shift short attention with group_size_ratio=1/4.")
        else:
            logger.warning("Current model does not support shift short attention.")

    # TODO(@zyw): 采用 bitsandbytes 后端来配置量化方法
    if model_args.quantization_bit is not None:
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 与量化当前无法互相兼容！")

        if model_args.quantization_bit == 8: 
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        elif model_args.quantization_bit == 4: 
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_args.compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quantization,
                bnb_4bit_quant_type=model_args.quantization_type
            )

        is_mergeable = False
        config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))} if is_trainable else "auto"
        logger.info("Quantizing model to {} bit.".format(model_args.quantization_bit))

    # 加载预训练模型权重 (without valuehead).
    model = AutoModelForCausalLM.from_pretrained(
        model_to_load,
        config=config,
        torch_dtype=model_args.compute_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        **config_kwargs
    )

    is_mergeable = True
    # 替换 Qwen 和 Baichuan2 模型中自定义的 generate 接口
    if isinstance(model, PreTrainedModel) and "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)

    # 为 ChatGLM2 模型修复 lm_head 属性
    if getattr(config, "model_type", None) == "chatglm":
        setattr(model, "lm_head", model.transformer.output_layer)

    # Register auto class to save the custom code files.
    if isinstance(config, PretrainedConfig) and "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if isinstance(model, PreTrainedModel) and "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if isinstance(tokenizer, PreTrainedTokenizerBase) and "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()

    # TODO(@zyw): 设置模型适配 adapters
    # (1) cast the layernorm in fp32
    # (2) make output embedding layer require grads
    # (3) upcast the lm_head to fp32
    model = prepare_model_for_training(model=model, finetuning_args=finetuning_args) if is_trainable else model
    # TODO(@zyw): 初始化 adapter 模型
    model = init_adapter(model, model_args, finetuning_args, is_trainable, is_mergeable)
    # 设置 train / eval 模式
    model = model.train() if is_trainable else model.eval()

    # TODO(@zyw): RLHF 训练阶段：为模型添加 valuehead
    if stage == "rm" or stage == "ppo":
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        model._keys_to_ignore_on_save = None
        if stage == "rm" and model_args.checkpoint_dir is not None: # 加载 valuehead 权重来对 reward model 进行性能评估
            logger.warning("Only the last checkpoint containing valuehead will be loaded.")
            if load_valuehead_params(model, model_args.checkpoint_dir[-1]):
                model.v_head.load_state_dict({
                    "summary.weight": getattr(model, "reward_head_weight"),
                    "summary.bias": getattr(model, "reward_head_bias")
                })

        if stage == "ppo":    # 加载奖励模型 reward model
            logger.info("Load reward model from {}".format(model_args.reward_model))
            if getattr(model, "is_peft_model", False):
                model.pretrained_model.load_adapter(model_args.reward_model, "reward")
            assert load_valuehead_params(model, model_args.reward_model), "Reward model is not correctly loaded."

    # 模型推理
    if not is_trainable:
        model.requires_grad_(False) # fix all model params
        model = model.to(model_args.compute_dtype) if model_args.quantization_bit is None else model

    # 获取可训练参数的数量
    trainable_params, all_param = count_parameters(model)
    logger.info("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    if not is_trainable:
        logger.info("This IS expected that the trainable params is 0 if you are using model for inference only.")

    return model, tokenizer
