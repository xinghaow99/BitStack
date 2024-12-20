<p align="center">
  <img src="./assets/icon.png" width="180" height="150"/>
</p>

<h1 align="center">BitStack: Fine-Grained Size Control for Compressed Large Language Models in Variable Memory Environments</h1>

<div align="center">
<a href="https://arxiv.org/abs/2410.23918" target="_blank"><img src=https://img.shields.io/badge/2410.23918-red?style=plastic&logo=arxiv&logoColor=red&logoSize=auto&label=arXiv&labelColor=black&color=b31b1b&link=https%3A%2F%2Farxiv.org%2Fabs%2F2410.23918
></a>
<a href="https://huggingface.co/BitStack" target="_blank"><img src=https://img.shields.io/badge/BitStack-yellow?style=plastic&logo=huggingface&logoColor=ffd21e&logoSize=auto&label=HuggingFace&labelColor=black&color=ffd21e&link=https%3A%2F%2Fhuggingface.co%2FBitStack
></a>
</div>

![BitStack](./assets/bitstack.png)

## ✨ TL;DR
BitStack breaks down large language models into tiny little blocks, which can be sorted and stacked universally, achieving megabyte-level memory-performance tradeoffs while maintaining or surpassing the performance of practical compression methods like GPTQ and AWQ. Check out [our paper](https://arxiv.org/abs/2410.23918) for more details!


## 📰 News
- [2024-11-06] 🚀 We've released Triton kernels optimized for fused inference with BitStack models! These kernels deliver an impressive **3x** to **10x** speedup over the original implementation. Just set the `--fused_level` flag to get started! For more details, check out the speedup visualization [here](./assets/speedup_visualization.png).
- [2024-11-01] 🎈 Try out this [Colab demo](https://colab.research.google.com/drive/1GoXIVyhofOEpGzOUint8LOivlFSDVHle?usp=sharing) and play with BitStack models across various memory budgets using an intuitive slider built with Gradio!
- [2024-11-01] 📄 Check out our paper on [arXiv](https://arxiv.org/abs/2410.23918)!
- [2024-10-31] ✨ Pre-decomposed models are now available on [HuggingFace🤗](https://huggingface.co/BitStack)!
- [2024-10-31] 🚀 Code release! We have some awesome inference kernels for BitStack models coming soon, stay tuned!

## 🚀 Quick Start
### ⚙️ Installation
```
conda create -yn bitstack python=3.10
conda activate bitstack
pip install -e .
```

### 🔄 Decomposition
To run the decomposition of a model, run [this script](./scripts/decompose.sh) or the following command:
```
python -m bitstack.main \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --niter 16 \ # Number of iterations of decomposition, decrease for shorter runtime
    --k 16 \ # Number of singular vectors kept
    --nsamples 256 \ # Number of calibration samples
    --output_dir outputs \
    --do_save \
    --score_importance \ # Run the sorting process
    --generate_compression_configs # Generate compression configs
```

### 📊 Evaluation
To evaluate the decomposed model, run [this script](./scripts/evaluate.sh) or the following command:
```
python -m bitstack.main \
    --model_name_or_path /YOUR/CHECKPOINT/PATH \
    --k 16 \
    --max_memory_MB 5541 \ # Maximum available memory for the model
    --load_bitstack \ # Load the decomposed model
    --do_eval \ # Perplexity evaluation
    --lm_eval \ # Zero-shot evaluation
    --output_dir outputs
```
## 📌 Checkpoints
We provide pre-decomposed models and compression configs. Currently, the following models are available, with more to come—stay tuned!
| Model  | Download |
| :---: | :---: |
| Llama-2 | [🤗7B](https://huggingface.co/BitStack/BitStack-Llama-2-7B) / [🤗13B](https://huggingface.co/BitStack/BitStack-Llama-2-13B) / [🤗70B](https://huggingface.co/BitStack/BitStack-Llama-2-70B) |
| Llama-3 | [🤗8B](https://huggingface.co/BitStack/BitStack-Llama-3-8B) / [🤗70B](https://huggingface.co/BitStack/BitStack-Llama-3-70B) |
| Llama-3.1 | [🤗8B](https://huggingface.co/BitStack/BitStack-Llama-3.1-8B) / [🤗70B](https://huggingface.co/BitStack/BitStack-Llama-3.1-70B) |
| Llama-3.1-Instruct | [🤗8B](https://huggingface.co/BitStack/BitStack-Llama-3.1-8B-Instruct) / [🤗70B](https://huggingface.co/BitStack/BitStack-Llama-3.1-70B-Instruct) |
| Llama-3.2 | [🤗1B](https://huggingface.co/BitStack/BitStack-Llama-3.2-1B) / [🤗3B](https://huggingface.co/BitStack/BitStack-Llama-3.2-3B) |
| Mistral-7B-v0.3 | [🤗7B](https://huggingface.co/BitStack/BitStack-Mistral-7B-v0.3)|

You can download them via the following commands:
```
# (Optional) enable hf_transfer for faster download
# pip install hf_transfer
# export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli download BitStack/BitStack-Llama-3.1-8B --local-dir ./models/BitStack-Llama-3.1-8B
```
Or just download the compression config for your already decomposed model:
```
huggingface-cli download BitStack/BitStack-Llama-3.1-8B --local-dir /YOUR/CHECKPOINT/PATH/ --include "compression_config.json"
```

## 📖 Citation
```
@misc{wang2024bitstackfinegrainedsizecontrol,
      title={BitStack: Fine-Grained Size Control for Compressed Large Language Models in Variable Memory Environments}, 
      author={Xinghao Wang and Pengyu Wang and Bo Wang and Dong Zhang and Yunhua Zhou and Xipeng Qiu},
      year={2024},
      eprint={2410.23918},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.23918}, 
}
```