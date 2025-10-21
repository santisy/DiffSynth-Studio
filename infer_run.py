import os
import glob
import argparse
import subprocess
import sys
from pathlib import Path
from collections import deque


def extract_image_name(img_path):
    """Extract image name without extension from path."""
    return Path(img_path).stem


def find_masks(mask_root, image_name):
    """
    Find all masks matching the pattern: {image_name}-{hash}-*-mask.png
    Returns list of tuples: (mask_path, hash, location)
    """
    pattern = os.path.join(mask_root, f"{image_name}-*-mask.png")
    masks = glob.glob(pattern)
    
    mask_info = []
    for mask_path in masks:
        mask_filename = Path(mask_path).name
        # Parse: {image_name}-{hash}-{location}-mask.png
        parts = mask_filename.replace(f"{image_name}-", "").replace("-mask.png", "").split("-")
        
        if len(parts) >= 2:
            hash_part = parts[0]  # 6-digit MD5 hash
            location = "-".join(parts[1:])  # Everything between hash and "-mask.png"
            mask_info.append((mask_path, hash_part, location))
    
    return mask_info


def generate_command(img_path, lora_folder, prompt, output_root, mask_path, 
                     hash_part, location, stride, size="512,512"):
    """Generate inference command for a single mask."""
    image_name = extract_image_name(img_path)
    output_path = os.path.join(output_root, f"{image_name}-{hash_part}-results_mask_{location}")
    
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
    
    args = parser.parse_args()
    
    # Extract image name
    image_name = extract_image_name(args.img_path)
    print(f"Processing image: {image_name}")
    
    # Find all matching masks
    masks = find_masks(args.mask_root, image_name)
    
    if not masks:
        print(f"No masks found for image '{image_name}' in '{args.mask_root}'")
        print(f"Looking for pattern: {image_name}-*-mask.png")
        return
    
    print(f"Found {len(masks)} mask(s):")
    for mask_path, hash_part, location in masks:
        print(f"  - {Path(mask_path).name} (hash: {hash_part}, location: {location})")
    
    print("\nGenerating commands...\n")
    
    # Track success/failure
    successful = 0
    failed = 0
    
    # Generate and run commands for each mask
    for i, (mask_path, hash_part, location) in enumerate(masks, 1):
        cmd = generate_command(
            args.img_path,
            args.lora_folder,
            args.prompt,
            args.output_root,
            mask_path,
            hash_part,
            location,
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