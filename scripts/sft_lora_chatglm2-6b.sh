MODLE_DIR=/workspace/models/chatglm2-6b-20230625
OUTPUT_DIR=./temp/chatglm

CUDA_VISIBLE_DEVICES=1 python train.py \
    --stage sft \
    --model_name_or_path $MODLE_DIR \
    --do_train \
    --dataset alpaca_zh \
    --template default \
    --finetuning_type lora \
    --lora_target query_key_value \
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
