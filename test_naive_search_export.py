"""
Script for testing and exporting naive search results with both scored and unscored outputs.
This script performs table search using either exact or bounds matching and exports results
in two formats: with and without scores.
"""

import numpy as np
import pickle
import argparse
import mlflow
import time
import os

from naive_search_export import NaiveSearcher


def main(args2=None):
    """
    Main function to run the naive search test and export results.
    Exports results in two formats:
    1. With scores: (table_id, score) pairs
    2. Without scores: just table_ids
    
    Args:
        args2: Optional command line arguments
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    
    # Model configuration
    parser.add_argument("--encoder", type=str, default="sato", 
                      choices=['sherlock', 'sato', 'cl', 'tapex'],
                      help='Encoder type to use')
    parser.add_argument("--benchmark", type=str, default='santos',
                      help='Benchmark dataset to use')
    parser.add_argument("--augment_op", type=str, default="drop_col",
                      help='Augmentation operation')
    parser.add_argument("--sample_meth", type=str, default="tfidf_entity",
                      help='Sampling method')
    parser.add_argument("--matching", type=str, default='exact',
                      choices=['exact', 'bounds'],
                      help='Matching type: exact or bounds')
    parser.add_argument("--table_order", type=str, default="column",
                      help='Table ordering method')
    
    # Experiment configuration
    parser.add_argument("--run_id", type=int, default=0,
                      help='Run identifier')
    parser.add_argument("--single_column", dest="single_column", action="store_true",
                      help='Use single column mode')
    parser.add_argument("--K", type=int, default=10,
                      help='Number of top results to return')
    parser.add_argument("--threshold", type=float, default=0.6,
                      help='Similarity threshold')
    parser.add_argument("--scal", type=float, default=1.00,
                      help='Scaling factor')
    parser.add_argument("--restrict", type=int, default=0,
                      help='Restrict datalake to unionables based on ground truth')
    
    # Error analysis
    parser.add_argument("--bucket", type=int, default=0,
                      help='Error analysis bucket (0-4)')
    parser.add_argument("--analysis", type=str, default='col',
                      choices=['col', 'row', 'numeric'],
                      help='Analysis type')
    
    # MLFlow configuration
    parser.add_argument("--mlflow_tag", type=str, default=None,
                      help='MLFlow tag')

    hp = parser.parse_args(args=args2)

    # Log parameters to MLFlow
    for variable in ["encoder", "benchmark", "augment_op", "sample_meth", "matching", 
                    "table_order", "run_id", "single_column", "K", "threshold", 
                    "scal", "restrict"]:
        mlflow.log_param(variable, getattr(hp, variable))

    if hp.mlflow_tag:
        mlflow.set_tag("tag", hp.mlflow_tag)

    # Set up file paths based on encoder type
    dataFolder = hp.benchmark
    if hp.encoder == 'cl':
        # For column-level encoder, use more specific path structure
        base_path = f"data/{dataFolder}/vectors/{hp.encoder}"
        suffix = f"_{hp.augment_op}_{hp.sample_meth}_{hp.table_order}_{hp.run_id}"
        if hp.single_column:
            suffix += "_singleCol"
        query_path = f"{base_path}_query{suffix}.pkl"
        table_path = f"{base_path}_datalake{suffix}.pkl"
    else:
        # For other encoders, use simpler path structure
        query_path = f"data/{dataFolder}/{hp.encoder}_query.pkl"
        table_path = f"data/{dataFolder}/{hp.encoder}_datalake.pkl"

    # Load query data
    with open(query_path, "rb") as qfile:
        queries = pickle.load(qfile)
    print(f"Number of queries: {len(queries)}")

    # Initialize searcher and results containers
    searcher = NaiveSearcher(table_path, hp.scal)
    returnedResults = {}
    returnedResults_noscore = {}
    
    # Process queries
    start_time = time.time()
    query_times = []
    queries.sort(key=lambda x: x[0])
    qCount = 0

    for query in queries:
        qCount += 1
        if qCount % 10 == 0:
            print(f"Processing query {qCount} of {len(queries)} total queries.")

        query_start_time = time.time()
        
        # Perform search based on matching type
        if hp.matching == 'exact':
            if hp.restrict == 0:
                qres = searcher.topk(hp.encoder, query, hp.K, threshold=hp.threshold)
            else:
                print("Working on restricted data")
                gt_path = f"data/{dataFolder}/santos_small_union_groundtruth_diluted.pickle"
                qres = searcher.topk(hp.encoder, query, hp.K, threshold=hp.threshold,
                                   restrict=1, gth=gt_path)
        else:  # Bounds matching
            qres = searcher.topk_bounds(hp.encoder, query, hp.K, threshold=hp.threshold)

        # Process and store results
        res = [(tpl[0], tpl[1]) for tpl in qres]
        returnedResults[query[0]] = [(r[1], r[0]) for r in res]
        returnedResults_noscore[query[0]] = [r[1] for r in res]
        query_times.append(time.time() - query_start_time)

    # Create output directory if it doesn't exist
    output_dir = f"data/{dataFolder}/diveristy_data/search_results/Starmie"
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    with open(f"{output_dir}/top_20_Starmie_output_04diluted_restricted_withscore.pkl", 'wb') as file:
        pickle.dump(returnedResults, file)
    with open(f"{output_dir}/top_20_Starmie_output_04diluted_restricted_noscore.pkl", 'wb') as file:
        pickle.dump(returnedResults_noscore, file)

    # Print performance statistics
    print(f"10th percentile: {np.percentile(query_times, 10):.2f}s")
    print(f"90th percentile: {np.percentile(query_times, 90):.2f}s")
    print(f"Total Query Time: {time.time() - start_time:.2f}s")


if __name__ == '__main__':
    main()     