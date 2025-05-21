"""
Script for evaluating novelty metrics across different search methods.
This script computes various novelty metrics (SNM, SSNM) and execution times
for different search methods including GMC, Penalized, Starmie, and others.
"""

import os
import argparse
from evaluate_novelty import (
    query_duplicate_returned,
    query_duplicate_returned_exclude,
    compute_syntactic_novelty_measure,
    compute_syntactic_novelty_measure_simplified,
    Avg_executiontime_by_k
)


def get_dataset_paths(dataset_name: str) -> tuple[str, str]:
    """
    Returns the appropriate datafolder and groundtruth filename based on the dataset name.
    
    Args:
        dataset_name: Name of the dataset ('santos', 'tus', 'ugen_v2', or 'ugen_v2_small')
        
    Returns:
        tuple: (datafolder, gtruthfilename)
    """
    dataset_paths = {
        'santos': {
            'datafolder': "/u6/bkassaie/NAUS/data/santos/small/",
            'gtruthfilename': "santos_small_union_groundtruth_diluted.pickle"
        },
        'tus': {
            'datafolder': "/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/",
            'gtruthfilename': "tus_small_noverlap_groundtruth_all.csv"
        },
        'ugen_v2': {
            'datafolder': "/u6/bkassaie/NAUS/data/ugen_v2/",
            'gtruthfilename': "ugenv2_unionable_groundtruth_diluted.pickle"
        },
        'ugen_v2_small': {
            'datafolder': "/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/",
            'gtruthfilename': "ugenv2_small_unionable_groundtruth_diluted.pickle"
        }
    }
    
    if dataset_name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of {list(dataset_paths.keys())}")
        
    return dataset_paths[dataset_name]['datafolder'], dataset_paths[dataset_name]['gtruthfilename']


def get_result_paths(datafolder: str) -> dict:
    """
    Get paths for all result files and diversity data directories.
    
    Args:
        datafolder: Base data folder path
        
    Returns:
        dict: Dictionary containing all result file paths and diversity data paths
    """
    return {
        'result_files': {
            'gmc': f"{datafolder}diveristy_data/search_results/GMC/gmc_results_diluted04_restricted.csv",
            'penalize': f"{datafolder}diveristy_data/search_results/Penalized/search_result_penalize_04diluted_restricted_pdeg1.csv",
            'starmie': f"{datafolder}diveristy_data/search_results/Starmie/starmie_results_04diluted_restricted.csv",
            'semanticNovelty': f"{datafolder}diveristy_data/search_results/semanticNovelty/search_result_semNovelty_04diluted_restricted_pdeg1.csv",
            'starmie0': f"{datafolder}diveristy_data/search_results/starmie0/search_result_starmie0_04diluted_restricted_pdeg1.csv",
            'starmie1': f"{datafolder}diveristy_data/search_results/starmie1/search_result_starmie1_04diluted_restricted_pdeg1.csv"
        },
        'diversity_paths': {
            'gmc': f"{datafolder}diveristy_data/search_results/GMC/",
            'penalized': f"{datafolder}diveristy_data/search_results/Penalized/",
            'starmie': f"{datafolder}diveristy_data/search_results/Starmie/",
            'semanticNovelty': f"{datafolder}diveristy_data/search_results/semanticNovelty/",
            'starmie0': f"{datafolder}diveristy_data/search_results/starmie0/",
            'starmie1': f"{datafolder}diveristy_data/search_results/starmie1/"
        }
    }


def process_duplicates(paths: dict) -> None:
    """
    Process duplicate results for all methods.
    
    Args:
        paths: Dictionary containing result file paths and diversity data paths
    """
    # Process regular duplicates
    for method in ['penalize', 'gmc', 'starmie', 'semanticNovelty', 'starmie0', 'starmie1']:
        dup_file = f"{paths['diversity_paths'][method]}search_result_{method}_diluted_restricted_duplicate.csv"
        if not os.path.exists(dup_file):
            query_duplicate_returned(paths['result_files'][method], dup_file)
        else:
            print(f"This file exists: {dup_file}")

    # Process excluded duplicates for ugen_v2_small
    exclude_set = {
        'Art-History_YZMEPGTH_ugen_v2.csv', 'Veterinary-Medicine_V4B1K1KD_ugen_v2.csv',
        'Criminology_JHT51TIW_ugen_v2.csv', 'Cooking_ZUIB5SQ0_ugen_v2.csv',
        'Architecture_EPZHPCF0_ugen_v2.csv', 'Horticulture_FAALFS04_ugen_v2.csv',
        'Math_XSK28T7A_ugen_v2.csv', 'Economics_XSEF6Y39_ugen_v2.csv',
        'Philosophy_SK28T7A9_ugen_v2.csv', 'International-Relations_YFSPEFRJ_ugen_v2.csv',
        'Religion_8F1CBFNO_ugen_v2.csv', 'Fashion_80QMAA7H_ugen_v2.csv',
        'Medicine_CVFJJ50Y_ugen_v2.csv', 'Culture_4QOD4K7Y_ugen_v2.csv',
        'Sociology_GIQHG9JR_ugen_v2.csv'
    }

    for method in ['penalize', 'gmc', 'starmie', 'semanticNovelty', 'starmie0', 'starmie1']:
        dup_file = f"{paths['diversity_paths'][method]}search_result_{method}_diluted_restricted_duplicate_excluded.csv"
        if not os.path.exists(dup_file):
            query_duplicate_returned_exclude(paths['result_files'][method], dup_file, exclude_set)
        else:
            print(f"This file exists: {dup_file}")


def compute_novelty_metrics(paths: dict, groundtruth: str) -> None:
    """
    Compute SNM and SSNM metrics for all methods.
    
    Args:
        paths: Dictionary containing result file paths and diversity data paths
        groundtruth: Path to ground truth file
    """
    # Compute SNM
    for method in ['starmie', 'penalize', 'semanticNovelty', 'starmie0', 'starmie1']:
        avg_file = f"{paths['diversity_paths'][method]}{method}_snm_diluted_restricted_avg_nodup.csv"
        whole_file = f"{paths['diversity_paths'][method]}{method}_snm_diluted_restricted_whole_nodup.csv"
        
        if not (os.path.exists(avg_file) or os.path.exists(whole_file)):
            compute_syntactic_novelty_measure(
                groundtruth,
                paths['result_files'][method],
                avg_file,
                whole_file,
                remove_duplicate=1
            )
        else:
            print(f"This file exists: {avg_file} or {whole_file}")

    # Compute SSNM
    for method in ['gmc', 'penalize', 'starmie', 'semanticNovelty', 'starmie0', 'starmie1']:
        avg_file = f"{paths['diversity_paths'][method]}{method}_ssnm_diluted_restricted_avg_nodup.csv"
        whole_file = f"{paths['diversity_paths'][method]}{method}_ssnm_diluted_restricted_whole_nodup.csv"
        
        if not (os.path.exists(avg_file) or os.path.exists(whole_file)):
            compute_syntactic_novelty_measure_simplified(
                groundtruth,
                paths['result_files'][method],
                avg_file,
                whole_file,
                remove_dup=1
            )
        else:
            print(f"This file exists: {avg_file} or {whole_file}")


def compute_execution_times(paths: dict) -> None:
    """
    Compute execution times for all methods.
    
    Args:
        paths: Dictionary containing result file paths and diversity data paths
    """
    for method in ['penalize', 'gmc', 'semanticNovelty', 'starmie0', 'starmie1', 'starmie']:
        time_file = f"{paths['diversity_paths'][method]}time_{method}_diluted_restricted.csv"
        if not os.path.exists(time_file):
            Avg_executiontime_by_k(paths['result_files'][method], time_file)
        else:
            print(f"This file exists: {time_file}")


def main():
    """Main function to run the evaluation pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate novelty metrics for different search methods')
    parser.add_argument("--dataset", type=str, default="tus",
                      choices=['santos', 'tus', 'ugen_v2', 'ugen_v2_small'],
                      help="Dataset to use for evaluation")
    args = parser.parse_args()

    try:
        # Get dataset paths
        datafolder, gtruthfilename = get_dataset_paths(args.dataset)
        groundtruth = os.path.join(datafolder, gtruthfilename)

        # Get all result paths
        paths = get_result_paths(datafolder)

        # Process duplicates
        print("Processing duplicates...")
        process_duplicates(paths)

        # Compute novelty metrics
        print("Computing novelty metrics...")
        compute_novelty_metrics(paths, groundtruth)

        # Compute execution times
        print("Computing execution times...")
        compute_execution_times(paths)

        print("Evaluation completed successfully!")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == '__main__':
    main()




 