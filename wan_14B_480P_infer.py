import argparse
import glob
import gc
import os
import re
from PIL import Image
import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

parser = argparse.ArgumentParser()
parser.add_argument("--lora_folder", type=str, default=None)
parser.add_argument("--img_path", type=str, required=True)
parser.add_argument("--mask_path", type=str, required=True)
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--size", type=lambda y: [int(x) for x in y.split(',')], default=None)
parser.add_argument("--step_threshold", type=int, default=None)
parser.add_argument("--stride", type=int, default=1, help="Stride length to filter checkpoints (e.g., stride=5 means every 5th checkpoint)")

args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)

step_threshold = args.step_threshold
stride = args.stride
lora_folder = args.lora_folder
lora_ckpts = []
step_n_list = []

if lora_folder is not None:
    # Get all LoRA checkpoints
    all_lora_ckpts = glob.glob(os.path.join(lora_folder, "*.safetensors"))
    
    # Extract step numbers and create (step_number, checkpoint_path) pairs
    checkpoint_pairs = []
    for ckpt in all_lora_ckpts:
        basename = os.path.basename(ckpt).split('.')[0]
        # Extract step number using regex (handles formats like "step-1000", "checkpoint-1000", etc.)
        match = re.search(r'(\d+)', basename)
        if match:
            step_num = int(match.group(1))
            checkpoint_pairs.append((step_num, ckpt, basename))
    
    # Sort by step number
    checkpoint_pairs.sort(key=lambda x: x[0])
    
    # Apply stride filtering and step threshold
    filtered_pairs = []
    for i, (step_num, ckpt, basename) in enumerate(checkpoint_pairs):
        # Apply step threshold filter
        if step_threshold is not None and step_num < step_threshold:
            continue
        
        # Apply stride filter (take every stride-th checkpoint)
        if i % stride == 0:
            filtered_pairs.append((step_num, ckpt, basename))
    
    # Extract the filtered checkpoints and step names
    lora_ckpts = [pair[1] for pair in filtered_pairs]
    step_n_list = [pair[2] for pair in filtered_pairs]


# Always include the original checkpoint (no LoRA)
lora_ckpts.append(None)
step_n_list.append('original_ckpt')

MODEL_ROOT = "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P"

image = Image.open(args.img_path).convert("RGB")
mask = Image.open(args.mask_path)
if args.size is not None:
    resize = image.size
    image = image.resize((args.size[0], args.size[1]), resample=Image.Resampling.BILINEAR)
    mask = mask.resize((args.size[0], args.size[1]), resample=Image.Resampling.BILINEAR)
    width = args.size[0]
    height = args.size[1]
else:
    width = 832
    height = 480
    resize = image.size

print(f"Processing {len(lora_ckpts)} checkpoints (including original)...")
if lora_folder is not None:
    print(f"LoRA checkpoints to process: {[s for s in step_n_list if s != 'original_ckpt']}")

for i, (lora_ckpt, step_n) in enumerate(zip(lora_ckpts, step_n_list)):
    print(f"Processing checkpoint {i+1}/{len(lora_ckpts)}: {step_n}")
    
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=[
                f"{MODEL_ROOT}/diffusion_pytorch_model-00001-of-00007.safetensors",
                f"{MODEL_ROOT}/diffusion_pytorch_model-00002-of-00007.safetensors",
                f"{MODEL_ROOT}/diffusion_pytorch_model-00003-of-00007.safetensors",
                f"{MODEL_ROOT}/diffusion_pytorch_model-00004-of-00007.safetensors",
                f"{MODEL_ROOT}/diffusion_pytorch_model-00005-of-00007.safetensors",
                f"{MODEL_ROOT}/diffusion_pytorch_model-00006-of-00007.safetensors",
                f"{MODEL_ROOT}/diffusion_pytorch_model-00007-of-00007.safetensors",
            ]),
            ModelConfig(path=f"{MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path=f"{MODEL_ROOT}/Wan2.1_VAE.pth"),
        ]
    )
    
    if lora_ckpt is not None:
        pipe.load_lora(pipe.dit, lora_ckpt, alpha=1)
    
    pipe.enable_vram_management()
    
    # Image-to-video
    video = pipe(
        prompt=args.prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=image,
        ref_mask=mask,
        seed=0,
        tiled=False,
        height=height,
        width=width
    )
    
    out_path = os.path.join(args.output_path, f"{step_n}.mp4")
    save_video(video, out_path, fps=15, quality=5, resize=resize)
    
    print(f"Saved video: {out_path}")
    
    # Clean up memory
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

print("All checkpoints processed successfully!")