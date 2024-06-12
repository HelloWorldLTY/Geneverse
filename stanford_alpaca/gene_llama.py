from transformers import AutoTokenizer
import transformers
import torch

# model = "meta-llama/Llama-2-7b-hf"
model = "./7bllama_20epoch"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

# sequences = pipeline(
#     'Could you please summarize the major function and information of the gene: CD79A? Use one paragraph.\n',
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=1000,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")
gene_list = [
'HEATR5B',
'ZNF385A',
'ZNF226',
'IGKV2D-36',
'RPL37P24',
'FGF7',
'LRRC7',
'MIR6721',
'SRD5A3',
'MIR608',
'EOGT',
'CASP10',
'CYCSP32',
'HSPA1L',
'SLC30A10',
'GLI1',
'RPS4XP1',
'RPS27AP8',
'CCL23',
'ACSS3']

for i in gene_list:
    query = f"Please summarize the major function of gene: {i}. Use academic language in one paragraph and include pathway information."
    # print(f"Please summarize the major function of gene: {i}. Use academic language in one paragraph and include pathway information.")
    print(i)
    sequences = pipeline(
    query,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=2000,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
