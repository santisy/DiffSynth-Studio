import os
import argparse
import subprocess
import sys
import json
import shlex
from pathlib import Path
from collections import deque
import re


def extract_image_name(img_path):
    """Extract image name without extension from path."""
    return Path(img_path).stem


def parse_mask_filename(mask_filename, image_name, fallback_hash=None):
    """
    Parse mask filename to extract hash part and mask id.
    Returns (hash_part, mask_id).
    """
    mask_pattern = "^{}-(?P<hash>[A-Za-z0-9]+)-mask-(?P<mask_id>\\d+)\\.png$".format(
        re.escape(image_name)
    )
    mask_regex = re.compile(mask_pattern)
    filename = Path(mask_filename).name
    match = mask_regex.match(filename)
    if match:
        return match.group("hash"), int(match.group("mask_id"))

    mask_id_match = re.search(r"mask-(\d+)", filename)
    if fallback_hash is not None and mask_id_match:
        return fallback_hash, int(mask_id_match.group(1))

    raise ValueError(
        f"Mask filename '{filename}' does not follow expected pattern and no fallback hash provided."
    )


def generate_command(img_path, lora_folder, prompt, output_root, mask_path,
                     hash_part, mask_id, stride, size="512,512"):
    """Generate inference command for a single mask."""
    image_name = extract_image_name(img_path)
    output_path = os.path.join(
        output_root, f"{image_name}-{hash_part}-results_mask_{mask_id:03d}"
    )

    cmd = ["python", "wan_14B_480P_infer.py"]
    cmd.extend(["--img_path", img_path])
    if lora_folder:
        cmd.extend(["--lora_folder", lora_folder])
    cmd.extend([
        "--prompt", prompt,
        "--output_path", output_path,
    ])
    if size:
        cmd.extend(["--size", size])
    cmd.extend([
        "--mask_path", mask_path,
        "--stride", str(stride),
    ])

    return cmd


def parse_mask_id_range(range_str):
    """
    Parse a mask id range specification.
    Supported forms:
        "5"   -> only id 5
        "2-6" -> ids 2 through 6 inclusive
        "3-"  -> ids >= 3
        "-4"  -> ids <= 4
    """
    if not range_str:
        return None

    parts = range_str.split("-")
    if len(parts) > 2:
        raise argparse.ArgumentTypeError(
            "Mask id range must be formatted as 'start-end', 'start-', '-end', or a single id."
        )

    if len(parts) == 1:
        try:
            value = int(parts[0])
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Mask id must be an integer.") from exc
        return (value, value)

    start_str, end_str = parts
    start = None
    end = None

    if start_str:
        try:
            start = int(start_str)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Mask id range start must be an integer.") from exc
    if end_str:
        try:
            end = int(end_str)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Mask id range end must be an integer.") from exc

    if start is not None and end is not None and start > end:
        raise argparse.ArgumentTypeError("Mask id range start cannot be greater than end.")

    return (start, end)


def clear_lines(n):
    """Clear n lines above cursor."""
    for _ in range(n):
        sys.stdout.write('\033[F')  # Move cursor up one line
        sys.stdout.write('\033[K')  # Clear line


def run_command_with_live_output(cmd, command_num, total_commands):
    """
    Run command and display live output, maintaining only the newest 4 lines.
    If an error occurs, display the full output for debugging.
    """
    print(f"Executing command {command_num}/{total_commands}...")

    # Buffer to store all output (for error case)
    full_output = []

    # Use deque to maintain last 4 lines (for normal case)
    last_lines = deque(maxlen=4)
    current_line_buffer = ""
    lines_displayed = 0

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    for line in process.stdout:
        # Store all output for potential error display
        full_output.append(line.rstrip('\n'))

        # Check if this is a progress bar update (contains \r)
        if '\r' in line:
            # Split by \r and take the last part (most recent update)
            parts = line.split('\r')
            for part in parts[:-1]:
                if part.strip():
                    current_line_buffer = part
            current_line_buffer = parts[-1].rstrip('\n')

            # Clear previous display
            if lines_displayed > 0:
                clear_lines(lines_displayed)

            # Update the last line in deque if it exists, otherwise add
            if last_lines and not last_lines[-1].endswith('\n'):
                last_lines[-1] = current_line_buffer
            else:
                last_lines.append(current_line_buffer)

            # Display the last 4 lines
            lines_displayed = 0
            for display_line in last_lines:
                print(display_line)
                lines_displayed += 1
            sys.stdout.flush()
        else:
            # Regular line with newline
            line = line.rstrip('\n')
            if line.strip():  # Only add non-empty lines
                # Clear previous display
                if lines_displayed > 0:
                    clear_lines(lines_displayed)

                # Add new line
                last_lines.append(line)

                # Display the last 4 lines
                lines_displayed = 0
                for display_line in last_lines:
                    print(display_line)
                    lines_displayed += 1
                sys.stdout.flush()

    process.wait()
    returncode = process.returncode

    # If there was an error, clear the 4-line display and show full output
    if returncode != 0:
        # Clear the current display
        if lines_displayed > 0:
            clear_lines(lines_displayed)

        print("\n" + "="*80)
        print("ERROR OCCURRED - Full output below:")
        print("="*80)
        for output_line in full_output:
            print(output_line)
        print("="*80)
        print(f"Command exited with code: {returncode}")
        print("="*80 + "\n")
    else:
        # Ensure we end with a newline for successful runs
        if lines_displayed > 0:
            print()

    return returncode


def collect_dataset_tasks(data_root, annotations_root, mask_id_range=None):
    """
    Collect all inference tasks from dataset root.
    Returns a tuple (tasks, warnings).
    Each task is a dict with keys:
        image_name, image_path, mask_path, mask_id, hash_part, prompt
    """
    data_root = Path(data_root)
    annotations_root = Path(annotations_root)

    tasks = []
    warnings = []

    json_paths = sorted(annotations_root.glob("*.json"))
    for json_path in json_paths:
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                annotation = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"Failed to read {json_path}: {exc}")
            continue

        image_filename = annotation.get("image_filename")
        if not image_filename:
            warnings.append(f"{json_path} missing 'image_filename'; skipping.")
            continue

        image_path = data_root / image_filename
        if not image_path.is_file():
            warnings.append(f"Image file not found: {image_path}")
            continue

        masks = annotation.get("masks", [])
        if not masks:
            warnings.append(f"No masks listed in {json_path}; skipping.")
            continue

        image_name = Path(image_filename).stem
        image_hash = annotation.get("image_hash")

        for mask_entry in masks:
            mask_filename = mask_entry.get("mask_filename")
            prompt = mask_entry.get("prompt")
            if not mask_filename:
                warnings.append(f"{json_path} contains mask entry without 'mask_filename'.")
                continue
            if prompt is None:
                warnings.append(f"{json_path} mask '{mask_filename}' missing prompt.")
                continue

            mask_path = annotations_root / mask_filename
            if not mask_path.is_file():
                warnings.append(f"Mask file not found: {mask_path}")
                continue

            try:
                hash_part, mask_id = parse_mask_filename(mask_filename, image_name, image_hash)
            except ValueError as exc:
                warnings.append(str(exc))
                continue

            if mask_id_range:
                start, end = mask_id_range
                if start is not None and mask_id < start:
                    continue
                if end is not None and mask_id > end:
                    continue

            tasks.append({
                "image_name": image_name,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "mask_id": mask_id,
                "hash_part": hash_part,
                "prompt": prompt.strip(),
            })

    tasks.sort(key=lambda item: (item["image_name"], item["mask_id"]))
    return tasks, warnings


def main():
    parser = argparse.ArgumentParser(
        description="Run DiffSynth inference across a dataset root."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing input images and an annotations subdirectory."
    )
    parser.add_argument(
        "--annotations_subdir",
        type=str,
        default="annotations",
        help="Subdirectory (relative to data_root) containing JSON files and masks."
    )
    parser.add_argument(
        "--lora_folder",
        type=str,
        default=None,
        help="Path to LoRA model folder (optional)."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory for generated outputs."
    )
    parser.add_argument("--stride", type=int, default=4, help="Stride parameter.")
    parser.add_argument(
        "--size",
        type=str,
        default="512,512",
        help="Output size (width,height). Use blank string to keep original size."
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--mask_id_range",
        type=parse_mask_id_range,
        help="Limit masks by id using 'start-end', 'start-', '-end', or a single id (inclusive)."
    )

    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()

    annotations_root = Path(args.annotations_subdir)
    if not annotations_root.is_absolute():
        annotations_root = data_root / annotations_root
    annotations_root = annotations_root.expanduser().resolve()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not data_root.is_dir():
        print(f"Data root not found: {data_root}")
        return

    if not annotations_root.is_dir():
        print(f"Annotations directory not found: {annotations_root}")
        return

    lora_folder = Path(args.lora_folder).expanduser().resolve() if args.lora_folder else None

    if args.mask_id_range:
        start, end = args.mask_id_range
        start_str = f"{start:03d}" if start is not None else "min"
        end_str = f"{end:03d}" if end is not None else "max"
        print(f"Mask id filter applied: {start_str} to {end_str}")

    tasks, warnings = collect_dataset_tasks(data_root, annotations_root, args.mask_id_range)
    for warning in warnings:
        print(f"[WARN] {warning}")

    if not tasks:
        print("No inference tasks discovered. Nothing to run.")
        return

    image_names = []
    last_name = None
    for task in tasks:
        if task["image_name"] != last_name:
            image_names.append(task["image_name"])
            last_name = task["image_name"]

    print(f"Discovered {len(image_names)} image(s) with {len(tasks)} mask(s) in {annotations_root}.")
    for image_name in image_names:
        mask_ids = [task["mask_id"] for task in tasks if task["image_name"] == image_name]
        mask_ids_str = ", ".join(f"{mask_id:03d}" for mask_id in mask_ids)
        print(f"  - {image_name}: masks {mask_ids_str}")

    print("\nGenerating commands...\n")

    size_arg = args.size if args.size else None
    total_commands = len(tasks)
    successful = 0
    failed = 0

    for idx, task in enumerate(tasks, 1):
        cmd = generate_command(
            task["image_path"],
            str(lora_folder) if lora_folder else None,
            task["prompt"],
            str(output_root),
            task["mask_path"],
            task["hash_part"],
            task["mask_id"],
            args.stride,
            size_arg
        )

        print(f"Command {idx}/{total_commands} for {task['image_name']} mask {task['mask_id']:03d}:")
        print(shlex.join(cmd))
        print()

        if args.dry_run:
            print(f"[DRY RUN] Would execute command {idx}/{total_commands}")
            print("-" * 80)
            continue

        returncode = run_command_with_live_output(cmd, idx, total_commands)
        if returncode == 0:
            print(f"✓ Successfully completed command {idx}/{total_commands}")
            successful += 1
        else:
            print(f"✗ Failed command {idx}/{total_commands} (exit code: {returncode})")
            failed += 1
        print("-" * 80)

    print(f"\n{'='*80}")
    print(f"Processing complete for data root '{data_root}'")
    if not args.dry_run:
        print(f"Total commands: {total_commands} | Successful: {successful} | Failed: {failed}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
