"""
Script for diluting benchmark datasets and their ground truth files.
This script handles the process of diluting datasets by adding rows from query tables
to datalake tables based on alignment information.
"""

import argparse
import csv
import glob
import os
from typing import Dict, List, Optional, Set, Union

import pandas as pd
import pickle

import utilities as utl


def dilute_alignmentfile(input_file: str, output_file: str) -> None:
    """
    Dilutes an alignment file by creating new rows with '_dlt' suffix for datalake tables.
    
    Args:
        input_file: Path to the input alignment CSV file
        output_file: Path to save the diluted alignment file
    """
    df = pd.read_csv(input_file)
    
    if 'dl_table_name' not in df.columns:
        raise ValueError("The CSV file must contain a column named 'dl_table_name'")
    
    new_rows = df.copy()
    new_rows['dl_table_name'] = new_rows['dl_table_name'].apply(
        lambda x: x[:-4] + '_dlt.csv' if isinstance(x, str) and x.endswith('.csv') else x
    )
    
    result_df = pd.concat([df, new_rows], ignore_index=True)
    result_df.to_csv(output_file, index=False)


def dilute_datalake_by_alignment(
    dilation_degree: float,
    query_directory: str,
    datalake_directory: str,
    diluted_datalake_directory: str,
    ground_truth_path: str,
    alignment_file: str,
    notdiluted_file: str,
    dataset: str,
    missingfiles: str
) -> None:
    """
    Dilutes datalake tables by adding rows from query tables based on alignment information.
    
    Args:
        dilation_degree: Degree of dilution (e.g., 0.4 means 40% of rows will be added)
        query_directory: Directory containing query tables
        datalake_directory: Directory containing datalake tables
        diluted_datalake_directory: Directory to save diluted tables
        ground_truth_path: Path to ground truth file
        alignment_file: Path to alignment file
        notdiluted_file: Path to save list of tables that couldn't be diluted
        dataset: Name of the dataset being processed
        missingfiles: Path to save list of missing files
    """
    # Load ground truth and alignment data
    groundtruth_dict = (utl.loadDictionaryFromCsvFile(ground_truth_path) 
                       if 'csv' in ground_truth_path 
                       else utl.loadDictionaryFromPickleFile(ground_truth_path))
    alignment = utl.load_alignment(alignment_file)
    
    # Initialize tracking variables
    no_common_column: Dict[str, List[str]] = {}
    missing_files_in_datalake: List[str] = []
    missing_files_in_query: List[str] = []
    delim = ';' if dataset in ['ugen-v2', 'ugen-v2_small'] else ','
    
    # Process each query table
    for query_file, datalake_files in groundtruth_dict.items():
        try:
            df_query = pd.read_csv(os.path.join(query_directory, query_file), sep=delim)
            df_query.columns = [col.lower() for col in df_query.columns]
        except FileNotFoundError:
            print(f"Error: Query file '{query_file}' not found")
            missing_files_in_query.append(query_file)
            continue
        except Exception as e:
            print(f"Error processing query file '{query_file}': {e}")
            continue
            
        num_rows_query = df_query.shape[0]
        
        # Process each datalake table
        for dl_file in datalake_files:
            try:
                df_dl = pd.read_csv(os.path.join(datalake_directory, dl_file), sep=delim)
            except FileNotFoundError:
                print(f"Error: Datalake file '{dl_file}' not found")
                missing_files_in_datalake.append(dl_file)
                continue
            except Exception as e:
                print(f"Error processing datalake file '{dl_file}': {e}")
                continue
                
            # Calculate sample size and get random sample
            num_rows_dl = df_dl.shape[0]
            sample_size = min(int(num_rows_dl * dilation_degree), num_rows_query)
            random_sample = df_query.sample(n=sample_size, random_state=42)
            
            # Get alignment information
            filtered_alignment = alignment[
                (alignment['query_table_name'] == query_file) &
                (alignment['dl_table_name'] == dl_file)
            ]
            
            if len(filtered_alignment) == 0:
                if query_file in no_common_column:
                    no_common_column[query_file].append(dl_file)
                else:
                    no_common_column[query_file] = [dl_file]
                continue
                
            # Create mapping for column alignment
            mapping = {row['dl_column']: row['query_column#'] 
                      for _, row in filtered_alignment.iterrows()}
            
            # Add diluted rows
            for _, row in random_sample.iterrows():
                new_row = {col: None for col in df_dl.columns}
                
                for col in df_dl.columns:
                    dl_col_index = df_dl.columns.get_loc(col)
                    if dl_col_index in mapping:
                        try:
                            new_row[col] = row.iloc[mapping[dl_col_index]]
                        except IndexError:
                            new_row[col] = None
                            
                df_dl = pd.concat([df_dl, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save diluted table
            output_file = os.path.join(diluted_datalake_directory, dl_file)
            df_dl.to_csv(output_file, sep=delim, index=False)
    
    # Save tracking information
    with open(notdiluted_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        for key, values in no_common_column.items():
            writer.writerow((key, values))
            
    with open(missingfiles, mode="w") as f:
        f.write("Missing files in data lake which exist in ground truth:\n")
        for item in missing_files_in_datalake:
            f.write(f"{item}\n")
        f.write("Missing files in query folder which exist in ground truth:\n")
        for item in missing_files_in_query:
            f.write(f"{item}\n")


def dilute_groundtruth(
    ground_truth_path: str,
    ground_truth_path_diluted: str,
    notdiluted_tnames_file: str
) -> None:
    """
    Dilutes ground truth file by adding entries for diluted tables.
    
    Args:
        ground_truth_path: Path to original ground truth file
        ground_truth_path_diluted: Path to save diluted ground truth file
        notdiluted_tnames_file: Path to file containing tables that couldn't be diluted
    """
    # Load ground truth and not diluted tables
    groundtruth_dict = (utl.loadDictionaryFromCsvFile(ground_truth_path)
                       if 'csv' in ground_truth_path
                       else utl.loadDictionaryFromPickleFile(ground_truth_path))
    notdiluted_tnames = utl.loadDictionaryFromCsvFile(notdiluted_tnames_file)
    
    new_groundtruth_dict = {}
    
    # Process each query table
    for query_file, datalake_files in groundtruth_dict.items():
        # Get not diluted tables for this query
        notdiluted = set()
        if query_file in notdiluted_tnames:
            dlstring = notdiluted_tnames[query_file]
            if isinstance(dlstring, list):
                dlstring = dlstring[0]
            dlstring = dlstring.replace('[\'', '').replace('\']', '').replace('\'', '')
            notdiluted = {item.strip() for item in dlstring.split(",")}
        
        # Remove not diluted tables and add diluted versions
        values = list(set(datalake_files) - notdiluted)
        updated_files = [f"{name[:-4]}_dlt.csv" if name.endswith(".csv") else name
                        for name in values]
        
        new_groundtruth_dict[query_file] = values + updated_files
    
    # Save diluted ground truth
    if 'csv' in ground_truth_path:
        with open(ground_truth_path_diluted, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["query_table", "data_lake_table"])
            for key, values in new_groundtruth_dict.items():
                for tbl in values:
                    writer.writerow([key, tbl])
    else:
        with open(ground_truth_path_diluted, "wb") as f:
            pickle.dump(new_groundtruth_dict, f)


def add_dlt_to_csv_filenames(folder_path: str) -> None:
    """
    Adds '_dlt' suffix to all CSV files in the specified folder.
    
    Args:
        folder_path: Path to the folder containing CSV files
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            base, ext = os.path.splitext(filename)
            new_filename = f"{base}_dlt{ext}"
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} --> {new_filename}")


def all_query_in_datalake(query_folder: str, datalake_folder: str) -> bool:
    """
    Verifies that all query files exist in the datalake folder.
    
    Args:
        query_folder: Path to query folder
        datalake_folder: Path to datalake folder
        
    Returns:
        bool: True if all query files exist in datalake, False otherwise
    """
    query_files = {os.path.basename(f) for f in glob.glob(os.path.join(query_folder, "*"))
                  if os.path.isfile(f)}
    datalake_files = {os.path.basename(f) for f in glob.glob(os.path.join(datalake_folder, "*"))
                     if os.path.isfile(f)}
    
    missing_files = query_files - datalake_files
    
    if missing_files:
        print("The following files in the query folder are missing in the data lake folder:")
        for file in sorted(missing_files):
            print(file)
        return False
    
    print("All files in the query folder are present in the data lake folder.")
    return True


def verify_groungtruth(groundtruth_file: str, query_folder: str) -> bool:
    """
    Verifies that ground truth file contains all necessary mappings.
    
    Args:
        groundtruth_file: Path to ground truth file
        query_folder: Path to query folder
        
    Returns:
        bool: True if ground truth is valid, False otherwise
    """
    # Load ground truth
    _, ext = os.path.splitext(groundtruth_file)
    ext = ext.lower()
    
    query_to_datalake_dict = (utl.loadDictionaryFromCsvFile(groundtruth_file)
                             if ext == ".csv"
                             else utl.loadDictionaryFromPickleFile(groundtruth_file))
    
    # Get query files
    query_files = {os.path.basename(f) for f in glob.glob(os.path.join(query_folder, "*"))
                  if os.path.isfile(f)}
    
    # Check if all query files are in ground truth
    missing_keys = [fname for fname in query_files if fname not in query_to_datalake_dict]
    if missing_keys:
        print("The following query files are missing as keys in the dictionary:")
        for key in missing_keys:
            print(f"  - {key}")
        return False
    
    print("All query files are present as keys in the dictionary.")
    
    # Check if each key appears in its own list
    missing_self = [key for key, table_list in query_to_datalake_dict.items()
                   if key not in table_list and key != 'query_table']
    
    if missing_self:
        print("\nThe following dictionary keys do not appear in their associated list:")
        for key in missing_self:
            print(f"  - {key} -> {query_to_datalake_dict[key]}")
        return False
    
    print("All dictionary keys are mapped to lists that contain themselves.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dilute benchmark datasets and their ground truth files.')
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['santos', 'santos_small', 'ugen-v2', 'ugen-v2_small', 'TUS_small'],
                      help='Dataset to process')
    parser.add_argument('--dilation_degree', type=float, required=True,
                      help='Dilation degree for the dataset (e.g., 0.4)')
    
    # Optional arguments with defaults
    parser.add_argument('--base_dir', type=str, default='/u6/bkassaie/NAUS',
                      help='Base directory for all paths')
    
    # Optional path arguments
    parser.add_argument('--ground_truth_path', type=str,
                      help='Path to ground truth file')
    parser.add_argument('--ground_truth_path_diluted', type=str,
                      help='Path to diluted ground truth file')
    parser.add_argument('--query_directory', type=str,
                      help='Path to query directory')
    parser.add_argument('--datalake_directory', type=str,
                      help='Path to datalake directory')
    parser.add_argument('--diluted_datalake_directory', type=str,
                      help='Path to diluted datalake directory')
    parser.add_argument('--alignmnet_file', type=str,
                      help='Path to alignment file')
    parser.add_argument('--alignmnet_file_diluted', type=str,
                      help='Path to diluted alignment file')
    parser.add_argument('--notdiluted_file', type=str,
                      help='Path to not diluted file')
    parser.add_argument('--missingfiles', type=str,
                      help='Path to missing files output')
    
    args = parser.parse_args()
    
    # Set up default paths based on dataset
    default_paths = {
        'santos_small': {
            'ground_truth_path': f"{args.base_dir}/data/santos/small/santos_small_union_groundtruth.pkl",
            'ground_truth_path_diluted': f"{args.base_dir}/data/santos/small/santos_small_union_groundtruth_diluted.pickle",
            'query_directory': f"{args.base_dir}/data/santos/small/query",
            'datalake_directory': f"{args.base_dir}/data/santos/small/datalake",
            'diluted_datalake_directory': f"{args.base_dir}/data/santos/small/datalake_diluteonly_dltdeg{args.dilation_degree}",
            'alignmnet_file': f"{args.base_dir}/data/santos/small/Manual_Alignment_4gtruth_santos_small.csv",
            'alignmnet_file_diluted': f"{args.base_dir}/data/santos/small/Manual_Alignment_4gtruth_santos_small_all.csv",
            'notdiluted04_file': f"{args.base_dir}/data/santos/small/notdiluted04_file.csv",
            'missingfiles': f"{args.base_dir}/data/santos/small/missing_files_dltdegree{args.dilation_degree}_{args.dataset}.csv"
        },
        'ugen-v2_small': {
            'ground_truth_path': f"{args.base_dir}/data/ugen_v2/ugenv2_small/ugenv2_small_unionable_groundtruth.pickle",
            'ground_truth_path_diluted': f"{args.base_dir}/data/ugen_v2/ugenv2_small/ugenv2_small_unionable_groundtruth_diluted.pickle",
            'query_directory': f"{args.base_dir}/data/ugen_v2/ugenv2_small/query",
            'datalake_directory': f"{args.base_dir}/data/ugen_v2/ugenv2_small/datalake",
            'diluted_datalake_directory': f"{args.base_dir}/data/ugen_v2/ugenv2_small/datalake_dilutedonly_dg{args.dilation_degree}",
            'alignmnet_file': f"{args.base_dir}/data/ugen_v2/ugenv2_small/ugenv2_small_manual_alignment.csv",
            'notdiluted04_file': f"{args.base_dir}/data/ugen_v2/ugenv2_small/notdiluted04_file.csv",
            'missingfiles': f"{args.base_dir}/data/ugen_v2/ugenv2_small/missing_files_dltdegree{args.dilation_degree}_{args.dataset}.csv"
        },
        'TUS_small': {
            'ground_truth_path': f"{args.base_dir}/data/table-union-search-benchmark/small/tus_small_noverlap_groundtruth_not_dlt.csv",
            'ground_truth_path_diluted': f"{args.base_dir}/data/table-union-search-benchmark/small/tus_small_noverlap_groundtruth_dlt_{args.dilation_degree}.csv",
            'query_directory': f"{args.base_dir}/data/table-union-search-benchmark/small/query",
            'datalake_directory': f"{args.base_dir}/data/table-union-search-benchmark/small/datalake",
            'diluted_datalake_directory': f"{args.base_dir}/data/table-union-search-benchmark/small/datalake_diluted{args.dilation_degree}_only",
            'alignmnet_file': f"{args.base_dir}/data/table-union-search-benchmark/small/Manual_Alignment_4gtruth_tus_benchmark.csv",
            'alignmnet_file_diluted': f"{args.base_dir}/data/table-union-search-benchmark/small/manual_alignment_tus_benchmark_all.csv",
            'notdiluted04_file': f"{args.base_dir}/data/table-union-search-benchmark/small/notdiluted04_file.csv",
            'missingfiles': f"{args.base_dir}/data/table-union-search-benchmark/small/missing_files_dltdegree{args.dilation_degree}_{args.dataset}.csv"
        }
    }
    
    if args.dataset not in default_paths:
        raise ValueError(f"Dataset {args.dataset} not implemented")
    
    # Start with default paths
    paths = default_paths[args.dataset].copy()
    
    # Override with any custom paths provided
    if args.ground_truth_path:
        paths['ground_truth_path'] = args.ground_truth_path
    if args.ground_truth_path_diluted:
        paths['ground_truth_path_diluted'] = args.ground_truth_path_diluted
    if args.query_directory:
        paths['query_directory'] = args.query_directory
    if args.datalake_directory:
        paths['datalake_directory'] = args.datalake_directory
    if args.diluted_datalake_directory:
        paths['diluted_datalake_directory'] = args.diluted_datalake_directory
    if args.alignmnet_file:
        paths['alignmnet_file'] = args.alignmnet_file
    if args.alignmnet_file_diluted:
        paths['alignmnet_file_diluted'] = args.alignmnet_file_diluted
    if args.notdiluted_file:
        paths['notdiluted04_file'] = args.notdiluted_file
    if args.missingfiles:
        paths['missingfiles'] = args.missingfiles

    # Verify data structure
    queries_are_duplicated = all_query_in_datalake(paths['query_directory'], 
                                                 datalake_folder=paths['datalake_directory'])
    if queries_are_duplicated:
        grthrut_has_query = verify_groungtruth(paths['ground_truth_path'], 
                                             paths['query_directory'])
        if grthrut_has_query:
            print("There is mapping in the groundtruth for blatant duplicates")
        else:
            raise RuntimeError("There is missing mapping in the groundtruth for blatant duplicates")
    else:
        raise RuntimeError("Blatant duplicate not in the folders")
    print("Dataset and ground truth file passed the test")

    # Process the dataset
    dilute_datalake_by_alignment(
        args.dilation_degree,
        paths['query_directory'],
        paths['datalake_directory'],
        paths['diluted_datalake_directory'],
        paths['ground_truth_path'],
        paths['alignmnet_file'],
        paths['notdiluted04_file'],
        args.dataset,
        paths['missingfiles']
    )
    
    add_dlt_to_csv_filenames(paths['diluted_datalake_directory'])
    dilute_groundtruth(paths['ground_truth_path'], 
                      paths['ground_truth_path_diluted'], 
                      paths['notdiluted04_file'])
    
    if 'alignmnet_file_diluted' in paths:
        dilute_alignmentfile(paths['alignmnet_file'], 
                           paths['alignmnet_file_diluted'])
     