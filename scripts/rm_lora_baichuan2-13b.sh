MODLE_DIR=/workspace/models/baichuan2-13b-chat-20230906
OUTPUT_DIR=./temp/baichuan_reward

CUDA_VISIBLE_DEVICES=1 python train.py \
    --stage rm \
    --model_name_or_path $MODLE_DIR \
    --do_train \
    --dataset comparison_gpt4_zh \
    --template default \
    --finetuning_type lora \
    --lora_target W_pack,o_proj \
    --output_dir $OUTPUT_DIR \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16
