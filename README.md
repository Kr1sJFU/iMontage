<h1 align="center">
  iMontage: Unified, Versatile, Highly Dynamic Many-to-many Image Generation
</h1>

<p align="center">
  <!-- 可以放 arXiv / project page / demo / hf 等链接的徽章 -->
  <a href="https://arxiv.org/abs/2511.20635"><img src="https://img.shields.io/badge/arXiv-2511.20635-b31b1b.svg" alt="arXiv"></a>
  <a href="https://kr1sjfu.github.io/iMontage-web/"><img src="https://img.shields.io/badge/Project-Page-4b9e5f.svg" alt="Project Page"></a>
  <a href="assets/demo/demo.mp4">
    <img src="https://img.shields.io/badge/Online-Demo-blue.svg" alt="Demo">
  </a>
  <a href="https://huggingface.co/Kr1sJ/iMontage"><img src="https://img.shields.io/badge/Model-HuggingFace-orange.svg" alt="HuggingFace"></a>
</p>


What if an image model could turn multi images into a coherent, dynamic visual universe? 🤯 iMontage brings video-like motion priors to image generation, enabling rich transitions and consistent multi-image outputs—all from your own inputs.
Try it out below and explore your imagination!


## 📦 Features

- ⚡ High-dynamic, high-consistency image generation from flexible inputs
- 🎛️ Robust instruction following across heterogeneous tasks
- 🌀 Video-like temporal coherence, even for non-video image sets
- 🏆 SOTA results across different tasks


## 📰 News

+ **2025.11.26** – Arxiv version paper of iMontage is released. 
+ **2025.11.26** – Inference code and model weights of iMontage are released. 



## 🛠 Installation

### 1. Create virtual environment

```bash
conda create -n iMontage python=3.10
conda activate iMontage

# NOTE Choose torch version compatible with your CUDA
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126

# Install Flash Attention 2
# NOTE Also choose the correct version compatible with installed torch
pip install "flash-attn==2.7.4.post1" --no-build-isolation

```

(Alternative) We train and evaluate our model with FlashAttention-3.  
If you are working on NVIDIA H100/H800 GPUs, you can follow the official guidance [here](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#flashattention-3-beta-release).
But you have to replace code in [fastvideo/models/flash_attn_no_pad.py](https://github.com/Kr1sJFU/iMontage/blob/main/fastvideo/models/flash_attn_no_pad.py)


After install torch and flash attention, you can install all other dependencies following this command:
```bash
pip install -e .
```

### 2. Download model weights

```bash
mkdir ckpts/hyvideo_ckpts

# Downloading hunyuan-video-i2v-720p, may takes 10 minutes to 1 hour depending on network conditions.
huggingface-cli download tencent/HunyuanVideo-I2V --local-dir ./ckpts/hyvideo_ckpts

# Downloading text_encoder from HunyuanVideo-T2V
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./ckpts/llava-llama-3-8b-v1_1-transformers
python fastvideo/models/hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py --input_dir ckpts/llava-llama-3-8b-v1_1-transformers --output_dir ckpts/hyvideo_ckpts/text_encoder

# Downloading text_encoder_2 from HunyuanVideo-I2V
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./ckpts/hyvideo_ckpts/text_encoder_2

mkdir ckpts/iMontage_ckpts
# Downloading iMontage dit weights, also might takes some time.
huggingface-cli download Kr1sJ/iMontage --local-dir ./ckpts/iMontage_ckpts
```

The final ckpt file structure should be formed as:
```code
iMontage
  ├──ckpts
  │  ├──hyvideo_ckpts
  │  │  ├──hunyuan-video-i2v-720p
  │  │  │  ├──transformers
  │  │  │  │  ├──mp_rank_00_model_states.pt
  ├  │  │  ├──vae
  │  │  ├──text_encoder_i2v
  │  │  ├──text_encoder_2
  │  ├──iMontage_ckpts
  │  │  ├──diffusion_pytorch_model.safetensors
  │ ...
```


## 🚀 Inference

After installing the environment and downloading the pretrained weights, let's start with our infer example.

---

### 🔹 Example

```bash
bash scripts/inference.sh
```