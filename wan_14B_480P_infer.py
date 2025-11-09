import argparse
import glob
import gc
import math
import os
import re
import numpy as np
from PIL import Image
import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

parser = argparse.ArgumentParser()
parser.add_argument("--lora_folder", type=str, default=None)
parser.add_argument("--img_path", type=str, required=True)
parser.add_argument("--mask_path", type=str, default=None)
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--size", type=lambda y: [int(x) for x in y.split(',')], default=None)
parser.add_argument(
    "--crop_mode",
    type=str,
    default="mask-square",
    choices=["mask-square", "full-resize"],
    help="mask-square: crop square around mask; full-resize: resize entire image as before."
)
parser.add_argument("--step_threshold", type=int, default=None)
parser.add_argument("--stride", type=int, default=1, help="Stride length to filter checkpoints (e.g., stride=5 means every 5th checkpoint)")
parser.add_argument(
    "--checkpoint_step",
    type=int,
    default=None,
    help="Only run the LoRA checkpoint whose filename contains this step number."
)
parser.add_argument(
    "--baseline",
    action="store_true",
    help="Run the base WAN 14B 480P model without LoRA, mask guidance, or cropping."
)

args = parser.parse_args()

baseline_mode = args.baseline

if baseline_mode:
    if args.crop_mode != "full-resize":
        print("Baseline mode: forcing --crop_mode full-resize.")
    args.crop_mode = "full-resize"

    if args.size != [832, 480]:
        if args.size is not None:
            print("Baseline mode: overriding --size to 832,480.")
    args.size = [832, 480]

    if args.mask_path is not None:
        print("Baseline mode: ignoring provided --mask_path.")
    args.mask_path = None

    if args.lora_folder is not None:
        print("Baseline mode: ignoring --lora_folder.")
    args.lora_folder = None

    if args.checkpoint_step is not None:
        print("Baseline mode: ignoring --checkpoint_step.")
    args.checkpoint_step = None

    if args.step_threshold is not None:
        print("Baseline mode: ignoring --step_threshold.")
    args.step_threshold = None

    if args.stride != 1:
        print("Baseline mode: forcing --stride to 1.")
    args.stride = 1

if args.mask_path is None and not baseline_mode:
    print("--mask_path is required unless --baseline is specified.")
    raise SystemExit(1)

os.makedirs(args.output_path, exist_ok=True)

step_threshold = args.step_threshold
stride = args.stride
lora_folder = args.lora_folder
checkpoint_step = args.checkpoint_step
lora_ckpts = []
step_n_list = []

if checkpoint_step is not None and lora_folder is None:
    print("--checkpoint_step requires --lora_folder to be set.")
    raise SystemExit(1)

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
    
    # Apply step threshold, optional checkpoint filter, then stride
    filtered_pairs = []
    for step_num, ckpt, basename in checkpoint_pairs:
        if step_threshold is not None and step_num < step_threshold:
            continue
        if checkpoint_step is not None and step_num != checkpoint_step:
            continue
        filtered_pairs.append((step_num, ckpt, basename))

    if checkpoint_step is not None and not filtered_pairs:
        print(f"Checkpoint step {checkpoint_step} not found in {lora_folder}.")
        raise SystemExit(1)

    stride_filtered = []
    for idx, (step_num, ckpt, basename) in enumerate(filtered_pairs):
        if idx % stride == 0:
            stride_filtered.append((step_num, ckpt, basename))
    filtered_pairs = stride_filtered

    # Skip checkpoints whose outputs already exist
    for _, ckpt, basename in filtered_pairs:
        out_file = os.path.join(args.output_path, f"{basename}.mp4")
        if os.path.exists(out_file):
            print(f"Skipping checkpoint '{basename}' because '{out_file}' already exists.")
            continue
        lora_ckpts.append(ckpt)
        step_n_list.append(basename)


include_original = checkpoint_step is None

# Always include the original checkpoint (no LoRA) if not already rendered
if include_original:
    original_out = os.path.join(args.output_path, "original_ckpt.mp4")
    if not os.path.exists(original_out):
        lora_ckpts.append(None)
        step_n_list.append('original_ckpt')
    else:
        print(f"Skipping original checkpoint because '{original_out}' already exists.")

if not lora_ckpts:
    print("No checkpoints to process; all outputs already exist.")
    raise SystemExit(0)

MODEL_ROOT = "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P"

original_image = Image.open(args.img_path).convert("RGB")
original_mask = None
if args.mask_path is not None:
    original_mask = Image.open(args.mask_path).convert("L")
original_size = original_image.size
img_w, img_h = original_size
output_resize = None
crop_metadata = None

if args.crop_mode == "full-resize":
    image = original_image
    mask = None if baseline_mode else original_mask
    output_resize = original_size  # Always stretch back to source canvas for legacy behavior.
    if args.size is not None:
        if len(args.size) != 2:
            raise ValueError("--size must include width and height, e.g., 512,512")
        width, height = args.size[0], args.size[1]
        inference_size = (width, height)
        image = image.resize(inference_size, resample=Image.Resampling.BILINEAR)
        if mask is not None:
            mask = mask.resize(inference_size, resample=Image.Resampling.NEAREST)
    else:
        width = 832
        height = 480
        inference_size = (width, height)
        if mask is not None:
            mask = mask.resize(inference_size, resample=Image.Resampling.NEAREST)
    print("Full-resize mode: using entire image without mask-driven cropping.")
else:
    if original_mask is None:
        raise ValueError("mask-square mode requires --mask_path.")
    mask_array = np.array(original_mask)
    mask_binary = mask_array > 0
    min_square_limit = min(img_w, img_h)

    if mask_binary.any():
        y_indices, x_indices = np.where(mask_binary)
        x_min, x_max = int(x_indices.min()), int(x_indices.max())
        y_min, y_max = int(y_indices.min()), int(y_indices.max())
        center_x = (x_min + x_max + 1) / 2.0
        center_y = (y_min + y_max + 1) / 2.0

        bbox_w = x_max - x_min + 1
        bbox_h = y_max - y_min + 1
        required_side = max(bbox_w, bbox_h)
        square_side = min_square_limit
        if required_side > square_side:
            print("Warning: mask bounding box is larger than the largest possible square crop; some content may be truncated.")

        allowed_left_min = max(0, x_max + 1 - square_side)
        allowed_left_max = min(img_w - square_side, x_min)
        allowed_top_min = max(0, y_max + 1 - square_side)
        allowed_top_max = min(img_h - square_side, y_min)

        if allowed_left_min > allowed_left_max:
            allowed_left_min = allowed_left_max = max(0, min(img_w - square_side, x_min))
        if allowed_top_min > allowed_top_max:
            allowed_top_min = allowed_top_max = max(0, min(img_h - square_side, y_min))

        left_float = center_x - square_side / 2.0
        top_float = center_y - square_side / 2.0
        left = int(round(left_float))
        top = int(round(top_float))
        left = max(allowed_left_min, min(left, allowed_left_max))
        top = max(allowed_top_min, min(top, allowed_top_max))
    else:
        print("Warning: mask has no active pixels; using centered crop.")
        square_side = min_square_limit
        left = (img_w - square_side) // 2
        top = (img_h - square_side) // 2

    right = left + square_side
    bottom = top + square_side
    crop_box = (left, top, right, bottom)

    cropped_image = original_image.crop(crop_box)
    cropped_mask_array = (np.array(original_mask.crop(crop_box)) > 0).astype(np.uint8) * 255
    cropped_mask = Image.fromarray(cropped_mask_array, mode="L")

    if args.size is not None:
        if len(args.size) != 2:
            raise ValueError("--size must include width and height, e.g., 512,512")
        requested_side = min(args.size[0], args.size[1])
        if args.size[0] != args.size[1]:
            print(f"Provided --size {args.size} is not square; using {requested_side} to preserve aspect ratio.")
    else:
        requested_side = square_side

    target_side = max(16, int(math.ceil(requested_side / 16.0) * 16))

    image = cropped_image.resize((target_side, target_side), resample=Image.Resampling.LANCZOS)
    mask = cropped_mask.resize((target_side, target_side), resample=Image.Resampling.NEAREST)

    width = target_side
    height = target_side
    crop_metadata = {
        "crop_box": crop_box,
        "crop_size": square_side,
        "paste_offset": (left, top)
    }

    print(f"Cropping square region {crop_box} -> pipeline input {target_side}x{target_side}")

if include_original:
    print(f"Processing {len(lora_ckpts)} checkpoints (including original)...")
else:
    print(f"Processing {len(lora_ckpts)} checkpoint(s)...")
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
    if crop_metadata is not None:
        composited_frames = []
        crop_size = crop_metadata["crop_size"]
        paste_offset = crop_metadata["paste_offset"]

        for frame in video:
            if isinstance(frame, Image.Image):
                pil_frame = frame.convert("RGB")
            elif isinstance(frame, torch.Tensor):
                frame_np = frame.detach().cpu().permute(1, 2, 0).numpy()
                pil_frame = Image.fromarray(((frame_np.clip(0, 1)) * 255).astype(np.uint8)).convert("RGB")
            else:
                pil_frame = Image.fromarray(np.array(frame)).convert("RGB")

            if pil_frame.size != (crop_size, crop_size):
                pil_frame = pil_frame.resize((crop_size, crop_size), resample=Image.Resampling.LANCZOS)

            canvas = original_image.copy()
            canvas.paste(pil_frame, paste_offset)
            composited_frames.append(canvas)

        save_video(composited_frames, out_path, fps=15, quality=5)
    else:
        save_video(video, out_path, fps=15, quality=5, resize=output_resize)
    
    print(f"Saved video: {out_path}")
    
    # Clean up memory
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

print("All checkpoints processed successfully!")
