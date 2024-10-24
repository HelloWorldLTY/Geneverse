# Introduction

To run the finetuning steps of LLMs, you need install [stanford-alpaca](https://github.com/tatsu-lab/stanford_alpaca) or [alpaca-lora](https://github.com/tloen/alpaca-lora). The former repo is for full-parameter finetuing, while the later repo is for parameter-efficient finetuning. 

To run the finetuning steps of MLLMs, you need install [fintuned LLaVA](https://github.com/mrseanryan/finetune_LLaVA). The official implementation of LLaVA can be found [here](https://github.com/haotian-liu/LLaVA).

To run the RAG-based LLMs, you need install [llamaindex](https://docs.llamaindex.ai/en/stable/getting_started/installation.html). 

# Tutorial

We have four different folders for four cases:

The folder stanford_alpaca is for full-parameter finetuning LLMs.

The folder alpaca_lora is for parameter-efficient finetuning LLMs.

The folder rag_llm is for RAG-based LLMs.

The folder finetuned_LLaVA is for fintuned LLaVA.

The folder post_processing is for post-processing step.

All of the hyper-parameters used in our experiments can be found in these folders.

# Acknowledgements

We thank the authors from stanford-alpaca, alpaca-lora and LLaVA for offering methods to finetune MLLMs.

# Citation

```
@article{liu2024geneverse,
  title={Geneverse: A collection of Open-source Multimodal Large Language Models for Genomic and Proteomic Research},
  author={Liu, Tianyu and Xiao, Yijia and Luo, Xiao and Xu, Hua and Zheng, W Jim and Zhao, Hongyu},
  journal={The 2024 Conference on Empirical Methods in Natural Language Processing},
  year={2024}
}
```