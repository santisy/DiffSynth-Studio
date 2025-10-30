import pandas as pd
import argparse
from pathlib import Path

def combine_csvs_with_paths(csv_files, output_file='combined.csv'):
    """
    Combine multiple CSV files and prepend the parent directory name to video and input_image columns.
    
    Args:
        csv_files: List of CSV file paths
        output_file: Output combined CSV file path
    """
    combined_data = []
    
    for csv_path in csv_files:
        # Get the parent directory name (B_1, B_2, etc.)
        parent_dir = Path(csv_path).parent.name
        
        # Read the CSV
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        required_columns = ['video', 'prompt', 'input_image']
        if not all(col in df.columns for col in required_columns):
            print(f"Warning: {csv_path} missing required columns. Skipping.")
            continue
        
        # Prepend parent directory to video and input_image only
        df['video'] = df['video'].apply(lambda x: f"{parent_dir}/{x}" if pd.notna(x) else x)
        df['input_image'] = df['input_image'].apply(lambda x: f"{parent_dir}/{x}" if pd.notna(x) else x)
        # prompt stays as-is (text string)
        
        combined_data.append(df)
        print(f"Processed: {csv_path} (parent dir: {parent_dir})")
    
    # Combine all dataframes
    if combined_data:
        result_df = pd.concat(combined_data, ignore_index=True)
        result_df.to_csv(output_file, index=False)
        print(f"\nCombined {len(combined_data)} files into {output_file}")
        print(f"Total records: {len(result_df)}")
    else:
        print("No valid CSV files to combine.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine CSV files with path prefixing')
    parser.add_argument('csv_files', nargs='+', help='CSV files to combine')
    parser.add_argument('-o', '--output', default='combined.csv', help='Output file name')
    
    args = parser.parse_args()
    
    combine_csvs_with_paths(args.csv_files, args.output)