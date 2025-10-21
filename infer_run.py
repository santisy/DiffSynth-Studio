import os
import glob
import argparse
import subprocess
import sys
from pathlib import Path
from collections import deque
import re


def extract_image_name(img_path):
    """Extract image name without extension from path."""
    return Path(img_path).stem


def find_masks(mask_root, image_name, mask_id_range=None):
    """
    Find all masks matching the pattern: {image_name}-{hash}-mask-{id:03d}.png
    Returns list of tuples: (mask_path, hash, mask_id)
    mask_id_range is a tuple(start, end) inclusive where bounds can be None.
    """
    pattern = os.path.join(mask_root, f"{image_name}-*-mask-*.png")
    masks = glob.glob(pattern)

    mask_info = []
    mask_regex = re.compile(
        rf"^{re.escape(image_name)}-(?P<hash>[A-Za-z0-9]+)-mask-(?P<mask_id>\d+)\.png$"
    )
    for mask_path in masks:
        mask_filename = Path(mask_path).name
        match = mask_regex.match(mask_filename)
        if not match:
            continue

        hash_part = match.group("hash")
        mask_id = int(match.group("mask_id"))

        if mask_id_range:
            start, end = mask_id_range
            if start is not None and mask_id < start:
                continue
            if end is not None and mask_id > end:
                continue

        mask_info.append((mask_path, hash_part, mask_id))

    mask_info.sort(key=lambda item: item[2])

    return mask_info


def generate_command(img_path, lora_folder, prompt, output_root, mask_path, 
                     hash_part, mask_id, stride, size="512,512"):
    """Generate inference command for a single mask."""
    image_name = extract_image_name(img_path)
    output_path = os.path.join(
        output_root, f"{image_name}-{hash_part}-results_mask_{mask_id:03d}"
    )
    
    cmd = [
        "python", "wan_14B_480P_infer.py",
        "--img_path", img_path,
        "--lora_folder", lora_folder,
        "--prompt", prompt,
        "--output_path", output_path,
        "--size", size,
        "--mask_path", mask_path,
        "--stride", str(stride)
    ]
    
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


def main():
    parser = argparse.ArgumentParser(description="Generate and run inference scripts for multiple masks")
    parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--lora_folder", type=str, required=True, help="Path to LoRA model folder")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory for outputs")
    parser.add_argument("--mask_root", type=str, required=True, help="Root directory containing masks")
    parser.add_argument("--stride", type=int, default=4, help="Stride parameter")
    parser.add_argument("--size", type=str, default="512,512", help="Output size (width,height)")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--mask_id_range",
        type=parse_mask_id_range,
        help="Limit masks by id using 'start-end', 'start-', '-end', or a single id (inclusive)."
    )
    
    args = parser.parse_args()
    
    # Extract image name
    image_name = extract_image_name(args.img_path)
    print(f"Processing image: {image_name}")

    if args.mask_id_range:
        start, end = args.mask_id_range
        start_str = f"{start:03d}" if start is not None else "min"
        end_str = f"{end:03d}" if end is not None else "max"
        print(f"Mask id filter: {start_str} to {end_str}")
    
    # Find all matching masks
    masks = find_masks(args.mask_root, image_name, args.mask_id_range)
    
    if not masks:
        print(f"No masks found for image '{image_name}' in '{args.mask_root}'")
        print(f"Looking for pattern: {image_name}-*-mask-*.png")
        return
    
    print(f"Found {len(masks)} mask(s):")
    for mask_path, hash_part, mask_id in masks:
        print(f"  - {Path(mask_path).name} (hash: {hash_part}, mask_id: {mask_id:03d})")
    
    print("\nGenerating commands...\n")
    
    # Track success/failure
    successful = 0
    failed = 0
    
    # Generate and run commands for each mask
    for i, (mask_path, hash_part, mask_id) in enumerate(masks, 1):
        cmd = generate_command(
            args.img_path,
            args.lora_folder,
            args.prompt,
            args.output_root,
            mask_path,
            hash_part,
            mask_id,
            args.stride,
            args.size
        )
        
        # Print command
        cmd_str = " \\\n    ".join([cmd[0] + " " + cmd[1]] + 
                                   [f"{cmd[j]} {cmd[j+1]}" for j in range(2, len(cmd), 2)])
        print(f"Command {i}/{len(masks)}:")
        print(cmd_str)
        print()
        
        # Execute command
        if not args.dry_run:
            returncode = run_command_with_live_output(cmd, i, len(masks))
            
            if returncode == 0:
                print(f"✓ Successfully completed command {i}/{len(masks)}")
                successful += 1
            else:
                print(f"✗ Failed command {i}/{len(masks)} (exit code: {returncode})")
                failed += 1
            print("-" * 80)
        else:
            print(f"[DRY RUN] Would execute command {i}/{len(masks)}")
            print("-" * 80)
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"Processing complete for image '{image_name}'")
    if not args.dry_run:
        print(f"Total: {len(masks)} | Successful: {successful} | Failed: {failed}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
