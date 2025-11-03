#!/bin/bash

set -exo pipefail

__conda_setup="$('/mnt/shared-storage-user/yangdingdong/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then                                                           
    eval "$__conda_setup"                                                       
else                                                                            
    if [ -f "/mnt/shared-storage-user/yangdingdong/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/mnt/shared-storage-user/yangdingdong/miniconda3/etc/profile.d/conda.sh"
    else                                                                        
        export PATH="/mnt/shared-storage-user/yangdingdong/miniconda3/bin:$PATH"
    fi                                                                          
fi                                                                              
unset __conda_setup

conda activate wan

cd /mnt/shared-storage-user/yangdingdong/code/DiffSynth-Studio

MODEL_PATHS_JSON='[
    [
        "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors",
        "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors",
        "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors",
        "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors",
        "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors",
        "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors",
        "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors"
    ],
    "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
    "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth"
]'

OUTPUT_PATH="./out_models/train/Wan2.1-I2V-14B-480P_lora_merged_data_1007_filtered_continue"

# Set this to the checkpoint you want to resume from, or leave empty to train from the base weights.
LORA_CHECKPOINT="${LORA_CHECKPOINT:-}"

ACCELERATE_CMD=(
  accelerate launch
  --config_file ./accelerate_config.yaml
  examples/wanvideo/model_training/train.py
  --dataset_base_path data/merged_data_1007
  --dataset_metadata_path data/merged_data_1007/merged_1007_filtered_simplified.csv
  --dataset_repeat 100
  --height 512
  --width 512
  --model_paths "$MODEL_PATHS_JSON"
  --learning_rate 1e-4
  --num_epochs 5
  --remove_prefix_in_ckpt pipe.dit.
  --output_path "$OUTPUT_PATH"
  --lora_base_model dit
  --lora_target_modules q,k,v,o,ffn.0,ffn.2
  --lora_rank 32
  --extra_inputs input_image,ref_mask
  --data_file_keys video,ref_mask
  --save_steps 4000
)

if [[ -n "$LORA_CHECKPOINT" ]]; then
  echo "Using LoRA checkpoint: $LORA_CHECKPOINT"
  ACCELERATE_CMD+=(--lora_checkpoint "$LORA_CHECKPOINT")
fi

"${ACCELERATE_CMD[@]}"
