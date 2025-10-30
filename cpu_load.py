import torch
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import launch_training_task

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cpu",
    model_configs=[
        ModelConfig(path=[
            "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ]),
        ModelConfig(path="/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth"),
    ]
)
