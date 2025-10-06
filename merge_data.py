#!/usr/bin/env python3

import argparse
import glob
import os
import shutil
import pandas as pd


def find_matching_folders(input_root, patterns):
    matching_folders = []
    for pattern in patterns:
        full_pattern = os.path.join(input_root, pattern)
        matches = glob.glob(full_pattern)
        folders = [m for m in matches if os.path.isdir(m)]
        matching_folders.extend(folders)
    return sorted(set(matching_folders))


def merge_csv_files(csv_files, output_path):
    dfs = [pd.read_csv(f) for f in csv_files]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_path, index=False)


def copy_other_files(input_folders, output_folder):
    for folder in input_folders:
        folder_name = os.path.basename(folder)
        for item in os.listdir(folder):
            if item.lower().endswith('.csv'):
                continue
            item_path = os.path.join(folder, item)
            dest_path = os.path.join(output_folder, f"{folder_name}_{item}")
            if os.path.isfile(item_path):
                shutil.copy2(item_path, dest_path)
            elif os.path.isdir(item_path):
                shutil.copytree(item_path, dest_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--input_names", nargs="+", required=True)
    parser.add_argument("--output_folder", required=True)
    args = parser.parse_args()
    
    input_folders = find_matching_folders(args.input_root, args.input_names)
    output_path = os.path.join(args.input_root, args.output_folder)
    os.makedirs(output_path, exist_ok=True)
    
    csv_files = [os.path.join(f, "metadata.csv") for f in input_folders 
                 if os.path.isfile(os.path.join(f, "metadata.csv"))]
    
    if csv_files:
        merge_csv_files(csv_files, os.path.join(output_path, "metadata.csv"))
    
    copy_other_files(input_folders, output_path)


if __name__ == "__main__":
    main()