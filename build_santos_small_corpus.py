"""
Script for building a small corpus from the Santos dataset by deduplicating and filtering tables.
This script creates a smaller version of the Santos dataset by removing duplicate tables and
their corresponding entries in the alignment file.
"""

import argparse
import os
import pickle
import shutil
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd

import utilities as utl


def deduplicate_and_copy(
    input_dict: Dict[str, List[str]],
    query_dir: str,
    datalake_dir: str,
    output_base_dir: str
) -> str:
    """
    Deduplicates tables and copies them to a new directory structure.
    
    Args:
        input_dict: Dictionary mapping query tables to their corresponding datalake tables
        query_dir: Directory containing query tables
        datalake_dir: Directory containing datalake tables
        output_base_dir: Base directory for output files
        
    Returns:
        str: Path to the saved filtered dictionary
        
    Raises:
        FileNotFoundError: If required directories don't exist
    """
    # Convert all paths to absolute paths
    query_dir = os.path.abspath(query_dir)
    datalake_dir = os.path.abspath(datalake_dir)
    output_base_dir = os.path.abspath(output_base_dir)
    query_out_dir = os.path.join(output_base_dir, "query")
    datalake_out_dir = os.path.join(output_base_dir, "datalake")
    
    # Step 1: Deduplicate keys with overlapping values
    total_keys = len(input_dict)
    keys = list(input_dict.keys())
    to_remove: Set[str] = set()

    for i in range(total_keys):
        for j in range(i + 1, total_keys):
            key1, key2 = keys[i], keys[j]
            set1, set2 = set(input_dict[key1]), set(input_dict[key2])
            if set1 & set2 and key2 not in to_remove and key1 not in to_remove:
                to_remove.add(key2)

    # Create filtered dict
    filtered_dict = {k: v for k, v in input_dict.items() if k not in to_remove}
    kept = len(filtered_dict)
    print(f"Kept {kept} out of {total_keys} keys: {list(filtered_dict.keys())}")

    # Step 2: Ensure output directories exist
    os.makedirs(query_out_dir, exist_ok=True)
    os.makedirs(datalake_out_dir, exist_ok=True)

    # Step 3: Copy files based on filtered dictionary
    for key, values in filtered_dict.items():
        # Copy query file
        src_query = os.path.join(query_dir, key)
        dst_query = os.path.join(query_out_dir, key)
        if os.path.exists(src_query):
            shutil.copy(src_query, dst_query)
        else:
            print(f"Warning: Query file not found for key {key}: {src_query}")

        # Copy each datalake file
        for value in values:
            src_data = os.path.join(datalake_dir, value)
            dst_data = os.path.join(datalake_out_dir, value)
            if os.path.exists(src_data):
                shutil.copy(src_data, dst_data)
            else:
                print(f"Warning: Datalake file not found for value {value}: {src_data}")

    # Step 4: Save filtered dictionary
    filtered_dict_path = os.path.join(output_base_dir, "santos_small_union_groundtruth.pkl")
    with open(filtered_dict_path, "wb") as f:
        pickle.dump(filtered_dict, f)

    print(f"Filtered dictionary saved to: {filtered_dict_path}")
    return filtered_dict_path


def filter_alignment_csv_by_dict(
    csv_path: str,
    filtered_dict_path: str,
    output_csv_path: str
) -> pd.DataFrame:
    """
    Filters an alignment CSV file based on the filtered dictionary.
    
    Args:
        csv_path: Path to the input alignment CSV file
        filtered_dict_path: Path to the filtered dictionary pickle file
        output_csv_path: Path to save the filtered CSV file
        
    Returns:
        pd.DataFrame: The filtered DataFrame
        
    Raises:
        FileNotFoundError: If input files don't exist
    """
    # Load filtered dictionary
    filtered_dict = utl.loadDictionaryFromPickleFile(filtered_dict_path)
    
    # Load and filter CSV
    df = pd.read_csv(os.path.abspath(csv_path))
    kept_query_tables = set(filtered_dict.keys())
    filtered_df = df[df['query_table_name'].isin(kept_query_tables)]

    # Save filtered CSV
    output_csv_path = os.path.abspath(output_csv_path)
    filtered_df.to_csv(output_csv_path, index=False)
    
    print(f"Filtered CSV saved to: {output_csv_path}")
    print(f"Kept {len(filtered_df)} rows out of {len(df)}")

    return filtered_df


def main():
    """Main function to process command line arguments and run the script."""
    parser = argparse.ArgumentParser(
        description='Build a small corpus from the Santos dataset by deduplicating and filtering tables.'
    )
    
    # Required arguments
    parser.add_argument('--input_dict', type=str, required=True,
                      help='Path to the input dictionary pickle file')
    parser.add_argument('--query_dir', type=str, required=True,
                      help='Directory containing query tables')
    parser.add_argument('--datalake_dir', type=str, required=True,
                      help='Directory containing datalake tables')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Base directory for output files')
    parser.add_argument('--alignment_csv', type=str, required=True,
                      help='Path to the input alignment CSV file')
    
    args = parser.parse_args()
    
    # Load input dictionary
    try:
        input_dict = utl.loadDictionaryFromPickleFile(args.input_dict)
    except FileNotFoundError:
        print(f"Error: Input dictionary file not found: {args.input_dict}")
        return
    except Exception as e:
        print(f"Error loading input dictionary: {e}")
        return
    
    # Create small corpus
    try:
        filtered_path = deduplicate_and_copy(
            input_dict,
            args.query_dir,
            args.datalake_dir,
            args.output_dir
        )
    except Exception as e:
        print(f"Error creating small corpus: {e}")
        return
    
    # Filter alignment CSV
    try:
        output_csv = os.path.join(args.output_dir, "Manual_Alignment_4gtruth_santos_small.csv")
        filter_alignment_csv_by_dict(
            args.alignment_csv,
            filtered_path,
            output_csv
        )
    except Exception as e:
        print(f"Error filtering alignment CSV: {e}")
        return


if __name__ == "__main__":
    main()


    