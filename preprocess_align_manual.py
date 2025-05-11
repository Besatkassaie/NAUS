"""
Table Alignment and Union Processing Module

This module provides functionality for processing and aligning tables from query and data lake sources.
It supports multiple benchmark datasets and exports alignments to CSV format.

Key features:
- Processes query tables and data lake tables
- Supports multiple benchmark datasets (santos, ugen_v2, tus_benchmark)
- Exports alignments to CSV format
- Handles ground truth validation
"""

# Standard library imports
import os
import glob
import csv
import argparse
import time
import random

# Third-party imports
import pandas as pd
import numpy as np
import utilities as utl

# %%

def export_alignment_to_csv(final_alignment, track_columns_reverse, output_file):
    """
    Export the final alignment to a CSV file.

    Args:
        final_alignment (set): A set of tuples representing aligned column pairs
        track_columns_reverse (dict): Reverse mapping from indices to (table_name, column_name)
        output_file (str): Path to the output CSV file

    Returns:
        bool: True if export was successful, False otherwise
    """
    try:
        file_exists = os.path.exists(output_file)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow([
                    "query_table_name", 
                    "query_column", 
                    "query_column#", 
                    "dl_table_name", 
                    "dl_column#",
                    "dl_column"
                ])
            
            # Write alignment data
            for col_pair in final_alignment:
                query_col = track_columns_reverse[col_pair[0]]
                dl_col = track_columns_reverse[col_pair[1]]
                
                # Strip trailing '\r' from column names if present
                query_col_name = query_col[1].rstrip('\r')
                dl_col_name = dl_col[1].rstrip('\r')
                
                writer.writerow([
                    query_col[0],  # query table name
                    query_col_name,  # query column name (stripped)
                    query_col[2],  # query column index
                    dl_col[0],     # data lake table name
                    dl_col_name,     # data lake column index
                    dl_col[2]    # data lake column name (stripped)
                ])
        
        print(f"Successfully exported {len(final_alignment)} alignments to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error exporting alignments to {output_file}: {str(e)}")
        return False

def get_table_columns(table_path_list: list) -> dict:
    """
    Get column information from a list of table files.
    
    Args:
        table_path_list (list): List of paths to table files
        
    Returns:
        dict: Dictionary mapping (table_name, column_name, column_index) to 0
    """
    table_columns = {}
    
    for file in table_path_list:
        try:
            df = utl.read_csv_file(file)
            table_name = os.path.basename(file)
            
            for idx, column in enumerate(df.columns):
                # Keep original column name in the key
                table_columns[(table_name, column, idx)] = 0
                
        except Exception as e:
            print(f"Error processing table {file}: {str(e)}")
            continue
            
    return table_columns

def manual_alignmnet_export(
    output_file: str,
    dl_table_folder: str,
    query_table_folder: str,
    benchmark_name: str,
    groundtruth_file: str,
    excluded_queries: set
) -> None:
    """
    Export manual alignments for tables based on ground truth data.
    
    Args:
        output_file (str): Path to the output CSV file
        dl_table_folder (str): Path to the data lake tables folder
        query_table_folder (str): Path to the query tables folder
        benchmark_name (str): Name of the benchmark dataset
        groundtruth_file (str): Path to the ground truth file
        excluded_queries (set): Set of query names to exclude from processing
        
    Raises:
        FileNotFoundError: If required files or directories are not found
        ValueError: If ground truth data is invalid
    """
    # Validate input paths
    if not os.path.exists(dl_table_folder):
        raise FileNotFoundError(f"Data lake folder not found: {dl_table_folder}")
    if not os.path.exists(query_table_folder):
        raise FileNotFoundError(f"Query folder not found: {query_table_folder}")
    if not os.path.exists(groundtruth_file):
        raise FileNotFoundError(f"Ground truth file not found: {groundtruth_file}")

    # Load ground truth based on benchmark type
    try:
        if benchmark_name == 'santos':
            groundtruth = utl.loadDictionaryFromPickleFile(groundtruth_file)
        elif benchmark_name == 'ugen_v2':
            groundtruth = utl.loadDictionaryFromCSV_ugenv2(groundtruth_file)
        elif benchmark_name == 'tus_benchmark':
            groundtruth = utl.loadDictionaryFromCsvFile(groundtruth_file)
        else:
            raise ValueError(f"Unsupported benchmark name: {benchmark_name}")
    except Exception as e:
        raise ValueError(f"Error loading ground truth: {str(e)}")

    if not groundtruth:
        raise ValueError("Ground truth data is empty")

    # Get query tables
    query_tables = glob.glob(os.path.join(query_table_folder, "*.csv"))
    if not query_tables:
        raise FileNotFoundError(f"No query tables found in {query_table_folder}")

    start_time = time.time()
    used_queries = 0
    queries_with_problematic_alignments = {}
    
    # Initialize statistics tracking
    alignment_stats = {}  # Will store stats for each query table

    # Process each query table
    for query_table in query_tables:
        query_table_name = os.path.basename(query_table)
        print(f"Processing query: {query_table_name}")

        # Skip excluded queries (case-insensitive comparison)
        if any(query_table_name.lower() == q.lower() for q in excluded_queries):
            print(f"Skipping excluded query: {query_table_name}")
            continue

        # Skip if query not in ground truth (case-insensitive comparison)
        if not any(query_table_name.lower() == k.lower() for k in groundtruth.keys()):
            print(f"Query not in ground truth: {query_table_name}")
            continue

        try:
            # Initialize tracking dictionaries
            track_tables = {}
            track_columns = {}
            track_columns_reverse = {}
            record_same_cluster = {}
            query_column_ids = set()
            i = 0

            # Get query table columns
            query_embeddings = get_table_columns([query_table])
            used_queries += 1
            print(f"Processing query {used_queries}")

            # Get unionable tables (case-insensitive comparison)
            unionable_tables = groundtruth[query_table_name]
            unionable_table_path = [
                os.path.join(dl_table_folder, tab)
                for tab in unionable_tables
                if tab != query_table_name and os.path.exists(os.path.join(dl_table_folder, tab))
            ]

            # Limit to 10 tables for tus_benchmark


            # Get data lake table columns
            dl_embeddings = get_table_columns(unionable_table_path)
            if not dl_embeddings:
                print("Not enough rows in any data lake tables. Skipping this cluster.")
                continue

            # Process query columns
            for column in query_embeddings:
                track_columns[column] = i
                track_columns_reverse[i] = column
                # Use case-insensitive and whitespace-insensitive lookup for table name
                table_name_lower = column[0].lower().strip()
                if table_name_lower not in track_tables:
                    track_tables[table_name_lower] = set()
                track_tables[table_name_lower].add(i)
                
                # Use case-insensitive and whitespace-insensitive lookup for column name
                col_name_lower = column[1].lower().strip()
                if col_name_lower not in record_same_cluster:
                    record_same_cluster[col_name_lower] = set()
                record_same_cluster[col_name_lower].add(i)
                
                query_column_ids.add(i)
                i += 1

            # Process data lake columns
            for column in dl_embeddings:
                track_columns[column] = i
                track_columns_reverse[i] = column
                # Use case-insensitive and whitespace-insensitive lookup for table name
                table_name_lower = column[0].lower().strip()
                if table_name_lower not in track_tables:
                    track_tables[table_name_lower] = set()
                track_tables[table_name_lower].add(i)
                
                # Use case-insensitive and whitespace-insensitive lookup for column name
                col_name_lower = column[1].lower().strip()
                if col_name_lower not in record_same_cluster:
                    record_same_cluster[col_name_lower] = set()
                record_same_cluster[col_name_lower].add(i)
                
                i += 1

            # Generate true edges with case-insensitive comparison
            all_true_edges = set()
            all_true_query_edges = set()

            for col_index_set in record_same_cluster:
                set1 = record_same_cluster[col_index_set]
                set2 = record_same_cluster[col_index_set]
                
                for s1 in set1:
                    for s2 in set2:
                        edge = tuple(sorted((s1, s2)))
                        all_true_edges.add(edge)
                        if s1 in query_column_ids or s2 in query_column_ids:
                            all_true_query_edges.add(edge)
                            
            # Check if all_true_query_edges is empty and log it
            if not all_true_query_edges:
                print(f"WARNING: No true query edges found for {query_table_name}")
                # Add this query to the problematic alignments with a specific message
                queries_with_problematic_alignments[query_table_name] = "No true query edges found"
            else:
                print(f"Found {len(all_true_query_edges)} true query edges for {query_table_name}")

            # Export alignments
            export_alignment_to_csv(all_true_query_edges, track_columns_reverse, output_file)
            
            # Track statistics for this query
            alignment_stats[query_table_name] = {
                'total_unionable_tables': len(unionable_tables),
                'tables_with_alignments': len(set(track_columns_reverse[edge[1]][0] for edge in all_true_query_edges)),
                'unionable_tables': set(unionable_tables),
                'tables_with_alignments_set': set(track_columns_reverse[edge[1]][0] for edge in all_true_query_edges)
            }
            
            print("-" * 50)

        except Exception as e:
            print(f"Error processing query {query_table_name}: {str(e)}")
            queries_with_problematic_alignments[query_table_name] = str(e)
            continue

    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
    print(f"Processed {used_queries} queries")
    
    # Print alignment statistics
    print("\n=== Alignment Coverage Statistics ===")
    print("Query Table | Total Unionable Tables | Tables With Alignments | Coverage %")
    print("-" * 80)
    for query, stats in alignment_stats.items():
        coverage = (stats['tables_with_alignments'] / stats['total_unionable_tables']) * 100 if stats['total_unionable_tables'] > 0 else 0
        print(f"{query:<30} | {stats['total_unionable_tables']:^20} | {stats['tables_with_alignments']:^21} | {coverage:>8.1f}%")
        
        # If coverage is not 100%, show missing alignments
        if coverage < 100:
            missing_tables = stats['unionable_tables'] - stats['tables_with_alignments_set']
            if missing_tables:
                print(f"\nMissing alignments for {query}:")
                for table in sorted(missing_tables):
                    print(f"  - {table}")
                print()  # Add blank line for readability
    
    if queries_with_problematic_alignments:
        print(f"\nFailed queries: {len(queries_with_problematic_alignments)}")
        for query, error in queries_with_problematic_alignments.items():
            print(f"  - {query}: {error}")

def main():
    """
    Main function to run the table alignment process.
    Supports multiple benchmark datasets and configurations.
    """
    parser = argparse.ArgumentParser(description='Process table alignments for different benchmarks')
    parser.add_argument('--dataset', type=str, default='santos',
                      choices=['santos', 'ugen_v2', 'tus_benchmark'],
                      help='Dataset to process')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Directory for output files (defaults to ground truth file directory)')
    args = parser.parse_args()

    # Configuration for different datasets
    dataset_configs = {
        "santos": {
            "excluded_queries": set(),
            "benchmark_name": "santos",
            "dl_table_folder": os.path.join("data", "santos", "datalake"),
            "query_table_folder": os.path.join("data", "santos", "query"),
            "groundtruth_file": "/u6/bkassaie/NAUS/data/santos/santos_union_groundtruth.pickle"
        },
        "ugen_v2": {
            "excluded_queries": set(),
            "benchmark_name": "ugen_v2",
            "dl_table_folder": os.path.join("data", "ugen_v2", "datalake"),
            "query_table_folder": os.path.join("data", "ugen_v2", "query"),
            "groundtruth_file": os.path.join("data", "ugen_v2", "manual_benchmark_validation_results", "ugen_v2_eval_groundtruth.csv")
        },
        "tus_benchmark": {
            "excluded_queries": set(),
            "benchmark_name": "tus_benchmark",
            "dl_table_folder": os.path.join("data", "table-union-search-benchmark", "small", "datalake"),
            "query_table_folder": os.path.join("data", "table-union-search-benchmark", "small", "query"),
            "groundtruth_file": os.path.join("data", "table-union-search-benchmark", "small", "tus_small_noverlap_groundtruth_not_dlt.csv")
        }
    }
  
    # Get configuration for selected dataset
    config = dataset_configs[args.dataset]
    
    # Set output directory to ground truth file directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.dirname(config['groundtruth_file'])
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set output file path
    output_file = os.path.join(args.output_dir, f"Manual_Alignment_4gtruth_{config['benchmark_name']}.csv")
    
    # Run alignment export
    manual_alignmnet_export(
        output_file,
        config['dl_table_folder'],
        config['query_table_folder'],
        config['benchmark_name'],
        config['groundtruth_file'],
        config['excluded_queries']
    )
    
    print(f"Manual Alignment Export for {args.dataset} completed.")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()