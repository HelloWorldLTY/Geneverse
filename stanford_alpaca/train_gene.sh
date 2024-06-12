# torchrun --nproc_per_node=2 --master_port=1235 train.py \
#     --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
#     --data_path ../genemore_repharse_onlytrain_instruction.json \
#     --bf16 True \
#     --output_dir ./7bllama_only_train_loraplus10epoch \
#     --num_train_epochs 10 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 2000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --tf32 False
# # past 2e-5

torchrun --nproc_per_node=2 --master_port=1235 train.py \
    --model_name_or_path google/gemma-7b-it \
    --data_path ../genemore_repharse_onlytrain_instruction.json \
    --bf16 True \
    --output_dir ./7bgemma_only_train_10epoch \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 False
# past 2e-5