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

#  --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
#  --height 480 \
#  --width 832 \

accelerate launch --config_file ./accelerate_config.yaml \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path data/data_0910_color_direct \
  --dataset_metadata_path data/data_0910_color_direct/metadata.csv \
  --dataset_repeat 100 \
  --height 512 \
  --width 512 \
  --model_paths '[
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
  ]' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./out_models/train/Wan2.1-I2V-14B-480P_lora_data_0910_512x512_direct_color" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image,ref_mask" \
  --data_file_keys "video,ref_mask" \
  --save_steps 4000

