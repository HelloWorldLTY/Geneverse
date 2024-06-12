# python finetune.py \
#     --base_model 'meta-llama/Llama-2-7b-hf' \
#     --data_path '../stanford_alpaca/genedata_instruction.json' \
#     --output_dir './7bllamagene_10epoch' \
#     --batch_size 128 \
#     --micro_batch_size 4 \
#     --num_epochs 10  \
#     --learning_rate 1e-4 \
#     --cutoff_len 512 \
#     --val_set_size 2000 \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
#     --train_on_inputs \
#     --group_by_length

# python finetune.py \
#     --base_model 'meta-llama/Llama-2-7b-chat-hf' \
#     --data_path '../genemore_repharse_onlytrain_instruction.json' \
#     --output_dir './7bllamagene_repharse_10epoch' \
#     --batch_size 128  \
#     --micro_batch_size 4 \
#     --num_epochs 10  \
#     --learning_rate 1e-4 \
#     --cutoff_len 512 \
#     --val_set_size 2000 \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
#     --train_on_inputs \
#     --group_by_length

# python finetune.py \
#     --base_model 'meta-llama/Llama-2-7b-chat-hf' \
#     --data_path '../genemore_repharse_onlytrain_instruction.json' \
#     --output_dir './7bllamarepharse_512_onlytrain_10epoch' \
#     --batch_size 512 \
#     --micro_batch_size 4 \
#     --num_epochs 10  \
#     --learning_rate 1e-4 \
#     --cutoff_len 1024 \
#     --val_set_size 2000 \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
#     --train_on_inputs \
#     --group_by_length

# python finetune.py \
#     --base_model 'mistralai/Mistral-7B-Instruct-v0.2' \
#     --data_path '../final_instruction.json' \
#     --output_dir './7bmistrallora_full_20epoch_1024' \
#     --batch_size 1024 \
#     --micro_batch_size 4 \
#     --num_epochs 20  \
#     --learning_rate 1e-4 \
#     --cutoff_len 2048 \
#     --val_set_size 2000 \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
#     --train_on_inputs \
#     --group_by_length

python finetune.py \
    --base_model 'mistralai/Mistral-7B-Instruct-v0.2' \
    --data_path '../genemore_repharse_onlytrain_instruction.json' \
    --output_dir './7bmistral_onlytrainloraplus_10epoch' \
    --batch_size 1024 \
    --micro_batch_size 4 \
    --num_epochs 10  \
    --learning_rate 2e-4 \
    --cutoff_len 2048 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --train_on_inputs \
    --group_by_length


# python finetune.py \
#     --base_model 'google/gemma-7b-it' \
#     --data_path '../final_instruction.json' \
#     --output_dir './gemma7b_len2048_alldata_20epoch_512' \
#     --batch_size 512 \
#     --micro_batch_size 4 \
#     --num_epochs 20  \
#     --learning_rate 1e-4 \
#     --cutoff_len 1024 \
#     --val_set_size 2000 \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
#     --train_on_inputs \
#     --group_by_length

# python finetune.py \
#     --base_model 'mistralai/Mistral-7B-Instruct-v0.2' \
#     --data_path '../genegpt35_instruction.json' \
#     --output_dir './7bmistralint_lora_gpt35_20epoch' \
#     --batch_size 512 \
#     --micro_batch_size 4 \
#     --num_epochs 20  \
#     --learning_rate 1e-4 \
#     --cutoff_len 1024 \
#     --val_set_size 2000 \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
#     --train_on_inputs \
#     --group_by_length