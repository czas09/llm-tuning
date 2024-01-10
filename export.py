from typing import Any, Dict, Optional

from loguru import logger

from argument_parser import get_infer_args
from model_loader import load_model_and_tokenizer


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
