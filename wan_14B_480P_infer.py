import argparse
import glob
import gc
import json
import math
import os
import re
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

parser = argparse.ArgumentParser()
parser.add_argument("--lora_folder", type=str, default=None)
parser.add_argument("--img_path", type=str, default=None)
parser.add_argument("--mask_path", type=str, default=None)
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--tasks_file", type=str, default=None, help="JSON file containing a list of tasks to process in batch.")
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


def _load_tasks_payload(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        tasks = payload.get("tasks")
        if tasks is None:
            raise ValueError("Tasks file must contain a 'tasks' array.")
        return tasks
    if isinstance(payload, list):
        return payload
    raise ValueError("Tasks file must contain either a list of tasks or an object with a 'tasks' array.")


def load_tasks(args, baseline_mode):
    if args.tasks_file:
        raw_tasks = _load_tasks_payload(args.tasks_file)
    else:
        raw_tasks = [{
            "image_path": args.img_path,
            "mask_path": None if baseline_mode else args.mask_path,
            "prompt": args.prompt,
            "output_path": args.output_path,
        }]

    normalized = []
    for idx, entry in enumerate(raw_tasks):
        if not isinstance(entry, dict):
            raise ValueError(f"Task entry at index {idx} is not an object.")

        image_path_val = entry.get("image_path")
        prompt_val = entry.get("prompt")
        output_path_val = entry.get("output_path")
        mask_path_val = entry.get("mask_path")

        if image_path_val is None:
            raise ValueError(f"Task entry at index {idx} missing 'image_path'.")
        if prompt_val is None:
            raise ValueError(f"Task entry at index {idx} missing 'prompt'.")
        if output_path_val is None:
            raise ValueError(f"Task entry at index {idx} missing 'output_path'.")
        if not baseline_mode and mask_path_val is None:
            raise ValueError(f"Task entry at index {idx} missing 'mask_path' for non-baseline run.")

        image_path = Path(image_path_val).expanduser().resolve()
        output_path = Path(output_path_val).expanduser().resolve()
        mask_path = None
        if not baseline_mode and mask_path_val is not None:
            mask_path = Path(mask_path_val).expanduser().resolve()

        prompt = str(prompt_val)
        image_name = entry.get("image_name") or image_path.stem
        mask_id = entry.get("mask_id")
        hash_part = entry.get("hash_part")
        annotation_path = entry.get("annotation_path")

        normalized.append({
            "image_path": str(image_path),
            "mask_path": str(mask_path) if mask_path is not None else None,
            "prompt": prompt,
            "output_path": str(output_path),
            "image_name": image_name,
            "mask_id": mask_id,
            "hash_part": hash_part,
            "annotation_path": annotation_path,
            "index": idx,
        })
    return normalized


def format_task_label(task):
    image_name = task.get("image_name") or Path(task["image_path"]).stem
    mask_id = task.get("mask_id")
    if mask_id is not None:
        try:
            return f"{image_name} mask {int(mask_id):03d}"
        except (TypeError, ValueError):
            return f"{image_name} mask {mask_id}"
    return image_name


def prepare_task_inputs(task, args, baseline_mode):
    original_image = Image.open(task["image_path"]).convert("RGB")
    original_mask = None
    if not baseline_mode:
        mask_path = task.get("mask_path")
        if mask_path is None:
            raise ValueError(f"Task '{format_task_label(task)}' is missing a mask path.")
        original_mask = Image.open(mask_path).convert("L")

    original_size = original_image.size
    img_w, img_h = original_size
    output_resize = None
    crop_metadata = None
    mask = None if baseline_mode else original_mask

    if args.crop_mode == "full-resize":
        image = original_image
        output_resize = original_size  # Stretch back to source canvas to preserve layout.
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
        return {
            "original_image": original_image,
            "image": image,
            "mask": mask,
            "width": width,
            "height": height,
            "crop_metadata": None,
            "output_resize": output_resize,
        }

    if original_mask is None:
        raise ValueError("mask-square mode requires a mask path per task.")

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

    crop_metadata = {
        "crop_box": crop_box,
        "crop_size": square_side,
        "paste_offset": (left, top),
    }

    return {
        "original_image": original_image,
        "image": image,
        "mask": mask,
        "width": target_side,
        "height": target_side,
        "crop_metadata": crop_metadata,
        "output_resize": None,
    }


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

step_threshold = args.step_threshold
stride = args.stride
lora_folder = args.lora_folder
checkpoint_step = args.checkpoint_step
lora_ckpts = []
step_n_list = []

if checkpoint_step is not None and lora_folder is None:
    print("--checkpoint_step requires --lora_folder to be set.")
    raise SystemExit(1)

if args.tasks_file is None:
    required_fields = ["img_path", "prompt", "output_path"]
    if not baseline_mode:
        required_fields.append("mask_path")
    missing = [field for field in required_fields if getattr(args, field) is None]
    if missing:
        parser.error(f"Missing required argument(s): {', '.join(missing)} (or provide --tasks_file).")

try:
    tasks = load_tasks(args, baseline_mode)
except (OSError, ValueError) as exc:
    print(f"Failed to load tasks: {exc}")
    raise SystemExit(1)

if not tasks:
    print("No tasks provided. Exiting.")
    raise SystemExit(0)

if args.tasks_file:
    print(f"Loaded {len(tasks)} task(s) from {args.tasks_file}.")
else:
    print(f"Loaded {len(tasks)} task(s) from command-line arguments.")

if lora_folder is not None:
    all_lora_ckpts = glob.glob(os.path.join(lora_folder, "*.safetensors"))

    checkpoint_pairs = []
    for ckpt in all_lora_ckpts:
        basename = os.path.basename(ckpt).split('.')[0]
        match = re.search(r'(\d+)', basename)
        if match:
            step_num = int(match.group(1))
            checkpoint_pairs.append((step_num, ckpt, basename))

    checkpoint_pairs.sort(key=lambda x: x[0])

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
    for idx_pair, pair in enumerate(filtered_pairs):
        if idx_pair % stride == 0:
            stride_filtered.append(pair)

    for _, ckpt, basename in stride_filtered:
        lora_ckpts.append(ckpt)
        step_n_list.append(basename)

include_original = checkpoint_step is None
if include_original and "original_ckpt" not in step_n_list:
    lora_ckpts.append(None)
    step_n_list.append("original_ckpt")

if not lora_ckpts:
    print("No checkpoints to process.")
    raise SystemExit(0)

MODEL_ROOT = "/mnt/shared-storage-user/yangdingdong/models/Wan2.1-I2V-14B-480P"

if args.crop_mode == "full-resize":
    print("Full-resize mode: using entire image without mask-driven cropping.")

total_checkpoints = len(lora_ckpts)
if include_original:
    print(f"Processing {total_checkpoints} checkpoint(s) (including original).")
else:
    print(f"Processing {total_checkpoints} checkpoint(s).")
if lora_folder is not None:
    lora_names = [name for name in step_n_list if name != "original_ckpt"]
    if lora_names:
        print(f"LoRA checkpoints to process: {lora_names}")

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

processed_any = False

for idx, (lora_ckpt, step_n) in enumerate(zip(lora_ckpts, step_n_list), start=1):
    pending = []
    for task in tasks:
        out_dir = Path(task["output_path"])
        out_path = out_dir / f"{step_n}.mp4"
        if not out_path.exists():
            pending.append((task, out_dir, out_path))

    if not pending:
        print(f"Skipping checkpoint {step_n}: outputs already exist for all tasks.")
        continue

    print(f"Processing checkpoint {idx}/{total_checkpoints}: {step_n}")

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

    for task, out_dir, out_path in pending:
        task_label = format_task_label(task)
        print(f"  -> {task_label}")

        prepared = prepare_task_inputs(task, args, baseline_mode)
        out_dir.mkdir(parents=True, exist_ok=True)

        video = pipe(
            prompt=task["prompt"],
            negative_prompt=NEGATIVE_PROMPT,
            input_image=prepared["image"],
            ref_mask=prepared["mask"],
            seed=0,
            tiled=False,
            height=prepared["height"],
            width=prepared["width"]
        )

        if prepared["crop_metadata"] is not None:
            composited_frames = []
            crop_size = prepared["crop_metadata"]["crop_size"]
            paste_offset = prepared["crop_metadata"]["paste_offset"]

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

                canvas = prepared["original_image"].copy()
                canvas.paste(pil_frame, paste_offset)
                composited_frames.append(canvas)

            save_video(composited_frames, str(out_path), fps=15, quality=5)
        else:
            save_video(video, str(out_path), fps=15, quality=5, resize=prepared["output_resize"])

        print(f"Saved video: {out_path}")
        processed_any = True

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

if processed_any:
    print("All checkpoints processed successfully!")
else:
    print("Nothing to process; all requested outputs already exist.")
