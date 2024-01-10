MODLE_DIR=/workspace/models/internlm-chat-20b-20230920
OUTPUT_DIR=./temp1/internlm

CUDA_VISIBLE_DEVICES=1 python train.py \
    --stage sft \
    --model_name_or_path $MODLE_DIR \
    --do_train \
    --dataset alpaca_zh \
    --template default \
    --finetuning_type lora \
    --lora_target W_pack,o_proj \
    --output_dir $OUTPUT_DIR \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
