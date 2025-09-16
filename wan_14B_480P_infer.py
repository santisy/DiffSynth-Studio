import argparse
import glob
import gc
import os

from PIL import Image
import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

parser = argparse.ArgumentParser()
parser.add_argument("--lora_folder", type=str, default=None)
parser.add_argument("--img_path", type=str, required=True)
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--size", type=lambda y: [int(x) for x in y.split(',')], default=None)

args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)

lora_folder = args.lora_folder
if lora_folder is not None:
    lora_ckpts = glob.glob(os.path.join(lora_folder, "*.safetensors"))
    step_n_list = [os.path.basename(ckpt).split('.')[0] for ckpt in lora_ckpts]
else:
    lora_ckpts = [None]
    step_n_list = ['original_ckpt']

MODEL_ROOT = "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P"



image = Image.open(args.img_path)
if args.size is not None:
    image.resize((args.size[0], args.size[1]), resample=Image.Resampling.BILINEAR)
    width = args.size[0]
    height = args.size[1]
    resize = None
else:
    width = 832
    height = 480
    resize = image.size[::-1]
    #image.resize((width, height), resample=Image.Resampling.BILINEAR)


for lora_ckpt, step_n in zip(lora_ckpts, step_n_list):

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
        seed=0,
        tiled=False,
        height=height,
        width=width
    )

    out_path = os.path.join(args.output_path, f"{step_n}.mp4")
    save_video(video, out_path, fps=15, quality=5, resize=resize)


    gc.collect()
    torch.cuda.empty_cache()
