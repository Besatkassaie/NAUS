"""
Script for building a corpus from the Table Union Search (TUS) benchmark dataset.
This script creates a corpus ensuring no overlap between query and datalake tables.
"""

import argparse
import os
import random
import shutil
from typing import List, Set, Tuple

import pandas as pd


def keep_noverlap_copy(
    input_csv: str = "input.csv",
    benchmark_dir: str = "benchmark",
    query_out_dir: str = "query",
    datalake_out_dir: str = "datalake",
    output_csv: str = "sampled_groundtruth.csv",
    sample_size: int = 50
) -> None:
    """
    Creates a corpus ensuring no overlap between query and datalake tables.
    
    Args:
        input_csv: Path to input CSV file
        benchmark_dir: Directory containing the benchmark tables
        query_out_dir: Directory to store query tables
        datalake_out_dir: Directory to store datalake tables
        output_csv: Path to save the filtered ground truth CSV
        sample_size: Number of unique query tables to keep
        
    Raises:
        FileNotFoundError: If input files or directories don't exist
    """
    # Read and filter the main CSV
    df = pd.read_csv(input_csv)
    df_filtered_with_duplicates = df[['query_table', 'data_lake_table']].copy()
    df_filtered = df_filtered_with_duplicates.drop_duplicates()
    
    # Process tables to ensure no overlap
    seen: Set[str] = set()  # Track seen datalake tables
    kept_rows: List[pd.Series] = []  # Store kept rows
    
    # First pass: Keep self-matches
    for _, row in df_filtered.iterrows():
        dt_value = row['data_lake_table']
        q_value = row['query_table']
        if dt_value == q_value:
            kept_rows.append(row)
            seen.add(dt_value)
    
    # Second pass: Keep non-overlapping tables
    queries = df_filtered['query_table'].unique()
    for query in queries:
        df_query = df_filtered[df_filtered['query_table'] == query]
        kept_rows_q = []
        for _, row in df_query.iterrows():
            dt_value = row['data_lake_table']
            if (dt_value not in seen) and (len(kept_rows_q) < 11):
                kept_rows_q.append(row)
                seen.add(dt_value)
        kept_rows.extend(kept_rows_q)
    
    # Combine and filter results
    df_combined = pd.DataFrame(kept_rows)
    group_counts = df_combined.groupby('query_table').size()
    valid_groups = group_counts[group_counts > 10].sort_values(ascending=False)
    top_queries = valid_groups.head(sample_size).index
    df_top = df_combined[df_combined['query_table'].isin(top_queries)].copy()
    
    # Ensure output folders exist
    os.makedirs(query_out_dir, exist_ok=True)
    os.makedirs(datalake_out_dir, exist_ok=True)

    # Copy query table files
    for qtbl in df_top["query_table"].unique():
        src = os.path.join(benchmark_dir, f"{qtbl}")
        dst = os.path.join(query_out_dir, f"{qtbl}")
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} not found. Skipped copying.")

    # Copy datalake table files
    for dtbl in df_top["data_lake_table"].unique():
        src = os.path.join(benchmark_dir, f"{dtbl}")
        dst = os.path.join(datalake_out_dir, f"{dtbl}")
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} not found. Skipped copying.")

    # Clean and save the filtered DataFrame
    df_top['query_table'] = df_top['query_table'].str.rstrip('\r')
    df_top['data_lake_table'] = df_top['data_lake_table'].str.rstrip('\r')
    df_top.to_csv(output_csv, index=False)

    # Print statistics
    print("\n=== Ground Truth Statistics ===")
    print(f"Total number of queries: {len(df_top['query_table'].unique())}")
    print("\nNumber of data lake tables per query:")
    query_counts = df_top.groupby('query_table')['data_lake_table'].nunique()
    for query, count in query_counts.items():
        print(f"- {query}: {count} data lake tables")
    print("\nTotal number of unique data lake tables:", len(df_top['data_lake_table'].unique()))
    print("=============================")


def main():
    """Main function to process command line arguments and run the script."""
    parser = argparse.ArgumentParser(
        description='Build a corpus from the Table Union Search (TUS) benchmark dataset.'
    )
    
    # Required arguments
    parser.add_argument('--input_csv', type=str, required=True,
                      help='Path to the input ground truth CSV file')
    parser.add_argument('--benchmark_dir', type=str, required=True,
                      help='Directory containing the benchmark tables')
    parser.add_argument('--query_dir', type=str, required=True,
                      help='Directory to store query tables')
    parser.add_argument('--datalake_dir', type=str, required=True,
                      help='Directory to store datalake tables')
    parser.add_argument('--output_csv', type=str, required=True,
                      help='Path to save the filtered ground truth CSV')
    
    # Optional arguments
    parser.add_argument('--sample_size', type=int, default=50,
                      help='Number of unique query tables to keep (default: 50)')
    
    args = parser.parse_args()
    
    try:
        keep_noverlap_copy(
            input_csv=args.input_csv,
            benchmark_dir=args.benchmark_dir,
            query_out_dir=args.query_dir,
            datalake_out_dir=args.datalake_dir,
            output_csv=args.output_csv,
            sample_size=args.sample_size
        )
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()