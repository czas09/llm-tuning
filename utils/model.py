import os
from typing import List, Optional, Tuple

import torch
from transformers.trainer import WEIGHTS_NAME
from transformers.utils import (
    is_torch_bf16_gpu_available, 
    is_torch_cuda_available, 
    is_torch_npu_available, 
)
from loguru import logger

_is_fp16_available = is_torch_cuda_available() or is_torch_npu_available()
try: 
    _is_bf16_available = is_torch_bf16_gpu_available()
except: 
    _is_bf16_available = False

from utils.constants import LAYERNORM_NAMES

from transformers.modeling_utils import PreTrainedModel
from hparams import FinetuningArguments


def find_all_linear_modules(
    model: "PreTrainedModel",
    quantization_bit: Optional[int] = None
) -> List[str]:
    if quantization_bit is not None:    # QLoRA（NF4）或 8 位量化 LoRA
        import bitsandbytes as bnb
        linear_cls = bnb.nn.Linear4bit if quantization_bit == 4 else bnb.nn.Linear8bitLt
    else:                               # LoRA
        linear_cls = torch.nn.Linear

    output_layer_names = ["lm_head"]
    if model.config.model_type == "chatglm":
        output_layer_names.append("output_layer")

    module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, linear_cls)
            and not any([output_layer in name for output_layer in output_layer_names])
        ):
            module_names.add(name.split(".")[-1])

    logger.info("Found linear modules: {}".format(",".join(module_names)))
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
        def neftune_forward_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
            if module.training:
                dims = torch.tensor(output.size(1) * output.size(2))
                mag_norm = finetuning_args.neft_alpha / torch.sqrt(dims)
                output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
            return output

        model.get_input_embeddings().register_forward_hook(neftune_forward_hook)
        logger.info("Using noisy embedding with alpha={:.2f}".format(finetuning_args.neft_alpha))


    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.gradient_checkpointing_enable()
        model.config.use_cache = False # turn off when gradient checkpointing is enabled
        logger.info("Gradient checkpointing enabled.")

    if finetuning_args.finetuning_type != "full" and hasattr(model, output_layer_name):
        output_layer = getattr(model, output_layer_name)
        if isinstance(output_layer, torch.nn.Linear):
            def fp32_forward_pre_hook(module: torch.nn.Module, args: Tuple[torch.Tensor]):
                return args[0].to(output_layer.weight.dtype)
            def fp32_forward_post_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
                return output.to(torch.float32)
            output_layer.register_forward_pre_hook(fp32_forward_pre_hook)
            output_layer.register_forward_hook(fp32_forward_post_hook)

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


def get_current_device() -> torch.device:
    import accelerate
    if accelerate.utils.is_xpu_available():
        device = "xpu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif accelerate.utils.is_npu_available():
        device = "npu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif torch.cuda.is_available():
        device = "cuda:{}".format(os.environ.get("LOCAL_RANK", "0"))
    else:
        device = "cpu"

    return torch.device(device)


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


def dispatch_model(model: "PreTrainedModel") -> "PreTrainedModel":
    r"""
    Dispatches a pre-trained model to GPUs with balanced memory.
    Borrowed from: https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/modeling_utils.py#L2803
    """
    if getattr(model, "quantization_method", None): # already set on current device
        return model

    if torch.cuda.device_count() > 1 and getattr(model.config, "model_type", None) != "chatglm":
        from accelerate import dispatch_model
        from accelerate.utils import infer_auto_device_map, get_balanced_memory

        if model._no_split_modules is None:
            raise ValueError("The model class needs to implement the `_no_split_modules` attribute.")

        kwargs = {"dtype": model.dtype, "no_split_module_classes": model._no_split_modules}
        max_memory = get_balanced_memory(model, **kwargs)
        # Make sure tied weights are tied before creating the device map.
        model.tie_weights()
        device_map = infer_auto_device_map(model, max_memory=max_memory, **kwargs)
        return dispatch_model(model, device_map)
    else:
        return model.cuda()