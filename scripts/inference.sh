export MODEL_BASE="./ckpts/hyvideo_ckpts"

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=1112 \
    fastvideo/sample/sample_imontage.py \
    --height 1024 \
    --width 1024 \
    --num_frames 1 \
    --num_inference_steps 50 \
    --guidance_scale 6.0 \
    --embedded_cfg_scale 1.0 \
    --flow_shift 7 \
    --flow-reverse \
    --dit-weight "./ckpts/iMontage_ckpts/diffusion_pytorch_model.safetensors" \
    --model_path "$MODEL_BASE" \
    --prompt "assets/prompt.json" \
    --output_path "./outputs" \
    --seed 42 