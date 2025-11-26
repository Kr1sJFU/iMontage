<h1 align="center">
  iMontage: Unified, Versatile, Highly Dynamic Many-to-many Image Generation
</h1>

<p align="center">
  <!-- 可以放 arXiv / project page / demo / hf 等链接的徽章 -->
  <a href="https://arxiv.org/abs/2511.20635"><img src="https://img.shields.io/badge/arXiv-2511.20635-b31b1b.svg" alt="arXiv"></a>
  <a href="https://kr1sjfu.github.io/iMontage-web/"><img src="https://img.shields.io/badge/Project-Page-4b9e5f.svg" alt="Project Page"></a>
  <a href="assets/demo/iMontage_demo.mp4">
    <img src="https://img.shields.io/badge/Online-Demo-blue.svg" alt="Demo">
  </a>
  <a href="https://huggingface.co/Kr1sJ/iMontage"><img src="https://img.shields.io/badge/Model-HuggingFace-orange.svg" alt="HuggingFace"></a>
</p>


<p align="center">
  <img src="assets/demo/teaser.png" alt="iMontage Teaser" width="30%">
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

(**Note**) We train and evaluate our model with FlashAttention-3, so the inference quality might be suboptimal with flash-attn-2. 

If you are working on NVIDIA H100/H800 GPUs and want to get the best performance of our model, you can follow the official guidance of FlashAttention-3 [here](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#flashattention-3-beta-release).
And you have to replace code in [fastvideo/models/flash_attn_no_pad.py](https://github.com/Kr1sJFU/iMontage/blob/main/fastvideo/models/flash_attn_no_pad.py)


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


### 🔹 Example

Run the following command:

```bash
bash scripts/inference.sh
```

In this example, we run inference with:

```bash
--prompt assets/prompt.json
```

The JSON file contains six representative tasks, including:

+ Image editing

+ Character reference generation (CRef)

+ CRef + Vision signal

+ Style reference generation (SRef)

+ Multi-view generation

+ Storyboard generation

Each entry specifies the task type, instruction prompt, input reference images, output resolution, and desired number of generated frames.
Running the script will automatically process all tasks in the JSON and save the results under the output directory.

The expected results should be:

| **Task Type**        | **Input**                                                   | **Prompt**                                                    | **Output**                                                  |
|----------------------|-------------------------------------------------------------|----------------------------------------------------------------|-------------------------------------------------------------|
| **image_editing**    | <img src="assets/images/llava.png" width="120">             | *Change the material of the lava to silver.*                  | <img src="assets/results/llava_0.png" width="120">    |
| **cref**             | <img src="assets/images/Confucius.png" width="80"> <img src="assets/images/Moses.png" width="80"> <img src="assets/images/Solon.png" width="80"> | *Confucius from the first image, Moses from the second…*      | <img src="assets/results/Confucius_0.png" width="120">             |
| **conditioned_cref**    | <img src="assets/images/depth.png" width="80">  <img src="assets/images/girl.png" width="80">             | *depth*                  | <img src="assets/results/depth_0.png" width="120">    |
| **sref**             | <img src="assets/images/woman.png" width="80"> <img src="assets/images/joker.png" width="80">      | *(empty)*                                                      | <img src="assets/results/woman_0.png" width="120">             |
| **multiview**        | <img src="assets/images/city.png" width="120">              | *1. Shift left; 2. Look up; 3. Zoom out.*                     | <img src="assets/results/city_0.png" width="80">  <img src="assets/results/city_1.png" width="80"> <img src="assets/results/city_2.png" width="80">       |
| **storyboard**       | <img src="assets/images/Hepburn.png" width="80"> <img src="assets/images/yellow_bag.png" width="80"> | *Vintage film: 1. Hepburn carrying the yellow bag…*           | <img src="assets/results/Hepburn_0.png" width="80"> <img src="assets/results/Hepburn_1.png" width="80"> <img src="assets/results/Hepburn_2.png" width="80">     |


### 🔹 Run your own job

To inference with your own images, you should create a JSON file and create an entry like this:

```code
"0" :
    {
        "task_type": "image_editing",
        "prompts" : "Change the material of the lava to silver.",
        "images" : [
            "assets/images/llava.png"
        ],
        "height" : 416,
        "width" : 640,
        "output_num" : 1
    }
```

And instruction of all tasks can be concluded as:

| **Task Type**                  | **Description**                                                                           | **Inputs**       | **Notes / Tips**                                                                                                                                |
| ------------------------------ | ----------------------------------------------------------------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **image_editing**              | Edit the input image according to the instruction (material, style, object change, etc.). | 1 image          | Prompt should clearly describe *what to change*. Best to align output size  with input image size.                                                |
| **cref**                       | Generate an output using multiple character reference images.     | ≥ 1 images       | Order of reference images matters. Prompt should specify *who from which image*. Best results with 2–4 reference images.                        |
| **conditioned_cref**              | Generate an output using multi images and a vision signal control map (depth, canny, openpose). | ≥ 1 image          | Only support depth, canny, openpose, prompt should be one of these three word. Put control map image in the first image.
| **sref**                       | Apply the style/features of the reference images to generate a new image.                 | 2 images | Leave `prompts` empty if only using style; model will infer style from input images. Put style reference image in the second place.                                                        |
| **multiview**                  | Generate multiple viewpoints of the same scene.                                           | 1 image          | Prompt should contain step-by-step view changes (e.g., “move left”, “look up”, “zoom out”). `output_num` must match number of described views. **NOTE** Might generate unsatisfying results, please try with different prompts and seed.|
| **storyboard**                 | Generate a sequence of frames forming a short story based on references.                  | ≥ 1 images       | Prompts should be enumerated (1, 2, 3…), and start with the story style word (Vintage file, Japanese anime, etc.). Use reference images to anchor characters or props. Output resolution often wider for cinematic style. |


## 💖 Acknowledgment

We sincerely thank the open-source community for providing strong foundations that enabled this work.  
In particular, we acknowledge the following projects for their models, datasets, and valuable insights:

- **HunyuanVideo-T2V**, **HunyuanVideo-I2V** – Provided base generative model designs and code. 
- **FastVideo** – Contributed key components and open-source utilities that supported our development.


These contributions have greatly influenced our research and helped shape the design of **iMontage**.

---

## 📝 Citation

If you find **iMontage** useful for your research or applications, please consider starring ⭐ the repo and citing our paper:

