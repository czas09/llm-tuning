import torch
from types import MethodType
from typing import TYPE_CHECKING, List, Optional, Tuple

import os
import torch
from transformers.trainer import WEIGHTS_NAME
from loguru import logger


from utils.constants import LAYERNORM_NAMES

from transformers.modeling_utils import PreTrainedModel
from hparams import FinetuningArguments

try: 
    from transformers.utils import (
        is_torch_bf16_cpu_available, 
        is_torch_bf16_gpu_available, 
        is_torch_cuda_available, 
        is_torch_npu_available, 
    )
    _is_fp16_available = is_torch_npu_available() or is_torch_cuda_available()
    _is_bf16_available = is_torch_bf16_gpu_available() or is_torch_bf16_cpu_available()

except ImportError: 
    _is_fp16_available = torch.cuda.is_available()
    _is_bf16_available = torch.cuda.is_bf16_supported()


def find_all_linear_modules(
    model: PreTrainedModel, 
    quantization_bit: Optional[int] = None, 
    output_layer_name: Optional[str] = "lm_head",    # 最后一层输出层，GPT 类模型中一般是 lm_head
) -> List[str]: 
    """找到当前模型中的线性 Linear 层
    
    用于需要对模型中所有的线性层训练 LoRA 权重的情况
    """

    if quantization_bit is not None:    # QLoRA（NF4）或 8 位量化 LoRA
        import bitsandbytes as bnb
        linear_cls = bnb.nn.Linear4bit if quantization_bit == 4 else bnb.nn.Linear8bitLt
    else:                               # LoRA
        linear_cls = torch.nn.Linear

    module_names = set()
    for name, module in model.named_modules():
        if output_layer_name not in name and isinstance(module, linear_cls):
            module_names.add(name.split(".")[-1])

    if output_layer_name in module_names:
        module_names.pop(output_layer_name)

    return list(module_names)


def prepare_model_for_training(
    model: PreTrainedModel,
    finetuning_args: FinetuningArguments,
    output_layer_name: Optional[str] = "lm_head",
    use_gradient_checkpointing: Optional[bool] = True,
    layernorm_names: Optional[List[str]] = LAYERNORM_NAMES
) -> PreTrainedModel:
    r"""对模型训练进行预处理

    TODO(@zyw)

    Includes:
        (1) cast the layernorm in fp32
        (2) make output embedding layer require grads
        (3) upcast the lm_head to fp32
    Inspired by: https://github.com/huggingface/peft/blob/v0.2.0/src/peft/utils/other.py#L33
    """
    if finetuning_args.upcast_layernorm:
        for name, param in model.named_parameters():
            if param.ndim == 1 and any(ln_name in name for ln_name in layernorm_names):
                param.data = param.data.to(torch.float32)
        logger.info("Upcasting weights in layernorm in float32.")

    if finetuning_args.neft_alpha > 1e-6:
        input_embed = model.get_input_embeddings()
        if isinstance(input_embed, torch.nn.Embedding):
            def noisy_forward(self: torch.nn.Embedding, x: torch.Tensor) -> torch.Tensor:
                embeddings = input_embed.__class__.forward(self, x)
                if self.training:
                    dims = self.num_embeddings * self.embedding_dim
                    mag_norm = finetuning_args.neft_alpha / (dims ** 0.5)
                    embeddings += torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)
                return embeddings

            input_embed.forward = MethodType(noisy_forward, input_embed)
            logger.info("Using noisy embedding with alpha={:.2f}".format(finetuning_args.neft_alpha))
        else:
            logger.warning("Input embeddings are not normal nn.Embedding, cannot transform into noisy embedding.")

    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.gradient_checkpointing_enable()
        model.config.use_cache = False # turn off when gradient checkpointing is enabled
        logger.info("启用梯度检查点（gradient checkpointing）技术。")

    if finetuning_args.finetuning_type != "full" and hasattr(model, output_layer_name):
        output_layer = getattr(model, output_layer_name)
        if isinstance(output_layer, torch.nn.Linear):
            def forward_in_fp32(self, x: torch.Tensor) -> torch.Tensor:
                return output_layer.__class__.forward(self, x.to(output_layer.weight.dtype)).to(torch.float32)

            output_layer.forward = MethodType(forward_in_fp32, output_layer)

    return model


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def infer_optim_dtype(model_dtype: torch.dtype) -> torch.dtype:
    r"""
    Infers the optimal dtype according to the model_dtype and device compatibility.
    """
    if _is_bf16_available and model_dtype == torch.bfloat16:
        return torch.bfloat16
    elif _is_fp16_available:
        return torch.float16
    else:
        return torch.float32


def load_valuehead_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
    vhead_file = os.path.join(checkpoint_dir, WEIGHTS_NAME)
    if not os.path.exists(vhead_file):
        logger.warning("Provided path ({}) does not contain valuehead weights.".format(checkpoint_dir))
        return False
    vhead_params = torch.load(vhead_file, map_location="cpu")
    model.register_buffer("reward_head_weight", vhead_params["v_head.summary.weight"], persistent=False)
    model.register_buffer("reward_head_bias", vhead_params["v_head.summary.bias"], persistent=False)
    model.register_buffer("default_head_weight", torch.zeros_like(vhead_params["v_head.summary.weight"]), persistent=False)
    model.register_buffer("default_head_bias", torch.zeros_like(vhead_params["v_head.summary.bias"]), persistent=False)
    return True
