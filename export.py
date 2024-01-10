import os.path
import sys
from typing import Any, Dict, Optional, Tuple

from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from loguru import logger

# from argument_parser import get_infer_args
from model_loader import load_model_and_tokenizer
from hparams import (
    DataArguments, 
    EvaluationArguments, 
    FinetuningArguments, 
    GeneratingArguments, 
    ModelArguments, 
)

_INFER_ARGS = [
    ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments
]

_INFER_CLS = Tuple[
    ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments
]

def parse_args(parser: "HfArgumentParser", args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()

def _verify_model_args(model_args: "ModelArguments", finetuning_args: "FinetuningArguments") -> None:
    if model_args.quantization_bit is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Quantization is only compatible with the LoRA method.")

    if (
        model_args.checkpoint_dir is not None
        and len(model_args.checkpoint_dir) != 1
        and finetuning_args.finetuning_type != "lora"
    ):
        raise ValueError("Multiple checkpoints are only available for LoRA tuning.")

def parse_infer_args(args: Optional[Dict[str, Any]] = None) -> _INFER_CLS:
    parser = HfArgumentParser(_INFER_ARGS)
    return parse_args(parser, args)


def get_infer_args(args: Optional[Dict[str, Any]] = None) -> _INFER_CLS:
    model_args, data_args, finetuning_args, generating_args = parse_infer_args(args)

    if data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    _verify_model_args(model_args, finetuning_args)

    return model_args, data_args, finetuning_args, generating_args


def export_model(args: Optional[Dict[str, Any]] = None):
    model_args, _, finetuning_args, _ = get_infer_args(args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)

    if getattr(model, "quantization_method", None) in ["gptq", "awq"]:
        raise ValueError("Cannot export a GPTQ or AWQ quantized model.")

    model.config.use_cache = True
    model.save_pretrained(finetuning_args.export_dir, max_shard_size="{}GB".format(finetuning_args.export_size))

    try:
        tokenizer.padding_side = "left" # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(finetuning_args.export_dir)
    except:
        logger.warning("Cannot save tokenizer, please copy the files manually.")
