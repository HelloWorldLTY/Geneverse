# python generate.py \
#     --base_model 'meta-llama/Llama-2-7b-chat-hf' \
#     --lora_weights './7bllamarepharse_rowcombine_10epoch'
# python generate_gene.py \
#     --base_model 'meta-llama/Llama-2-7b-chat-hf' \
#     --lora_weights './7bllamagene_ncbionly_10epoch' 

# python generate_gene.py \
#     --base_model 'mistralai/Mistral-7B-v0.1' \
#     --lora_weights './7bmistralai_2048_20epoch'

# python generate_gene.py \
#     --base_model 'meta-llama/Llama-2-7b-chat-hf' \
#     --lora_weights './7bllamarepharse_512_onlytrain_10epoch'

# python generate_gene.py \
#     --base_model 'meta-llama/Llama-2-13b-chat-hf' \
#     --lora_weights './13bllamarepharse_onlytrain_10epoch'


# python generate_gene.py \
#     --base_model 'mistralai/Mistral-7B-Instruct-v0.2' \
#     --lora_weights './7bmistralaiinst_2048_20epoch_1024'

python generate_gene.py \
    --base_model 'mistralai/Mistral-7B-Instruct-v0.2' \
    --lora_weights './7bmistralint_lora_gpt35_20epoch'