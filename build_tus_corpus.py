import os
import random
import shutil
import pandas as pd

def sample_and_copy(
    input_csv="input.csv",
    benchmark_dir="benchmark",
    query_out_dir="query",
    datalake_out_dir="datalake",
    output_csv="sampled_groundtruth.csv",
    sample_size=50,
    random_seed=42
):
    """
    1. Reads 'input_csv' which has columns:
         serial_num, query_table, intent_col_index,
         data_lake_table, intent_col_name, tree_level
    2. Randomly samples up to 'sample_size' unique query_table values.
    3. Keeps rows whose 'query_table' is in that sample.
    4. Copies CSV files from 'benchmark_dir/<table_name>.csv' to
       'query_out_dir' (for query_table) or 'datalake_out_dir'
       (for data_lake_table).
    5. Saves the filtered DataFrame to 'output_csv'.
    """

    # For reproducibility:
    random.seed(random_seed)
    
    # Read the main CSV
    df = pd.read_csv(input_csv)

    # 1) Collect unique 'query_table' values
    unique_query_tables = df["query_table"].unique()

    # 2) Sample 50 (or fewer, if there aren't that many) unique query_table values
    actual_sample_size = min(sample_size, len(unique_query_tables))
    sampled_query_tables = random.sample(list(unique_query_tables), actual_sample_size)

    # 3) Filter df to keep only rows whose query_table is in the sample
    df_subset = df[df["query_table"].isin(sampled_query_tables)]

    # Ensure output folders exist
    os.makedirs(query_out_dir, exist_ok=True)
    os.makedirs(datalake_out_dir, exist_ok=True)

    # 4) Copy relevant query_table files into 'query_out_dir'
    #    We assume the file is named "<table_name>.csv" in 'benchmark_dir'.
    for qtbl in df_subset["query_table"].unique():
        src = os.path.join(benchmark_dir, f"{qtbl}")
        dst = os.path.join(query_out_dir, f"{qtbl}")
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} not found. Skipped copying.")

    # Copy relevant data_lake_table files into 'datalake_out_dir'
    for dtbl in df_subset["data_lake_table"].unique():
        src = os.path.join(benchmark_dir, f"{dtbl}")
        dst = os.path.join(datalake_out_dir, f"{dtbl}")
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} not found. Skipped copying.")

    # 5) Write the filtered rows as a new ground truth CSV
    df_subset.to_csv(output_csv, index=False)
    
    
def keep_noverlap_copy(   
    input_csv="input.csv",
    benchmark_dir="benchmark",
    query_out_dir="query",
    datalake_out_dir="datalake",
    output_csv="sampled_groundtruth.csv",
    sample_size=50):
    
    # Read the main CSV
    df = pd.read_csv(input_csv)
        # Step 1: Keep only the desired columns
        # Step 1: Keep only the desired columns
    df_filtered_with_duplicates = df[['query_table', 'data_lake_table']].copy()
    
    # Step 2: Remove duplicate rows
    df_filtered = df_filtered_with_duplicates.drop_duplicates()
    
    # Step 3 and 4: Process each unique query_table
    seen = set()  # This will accumulate data lake table values that have already been kept.
    filtered_results = []  # List to hold DataFrames for each query_table
  
    # We'll accumulate rows for this query in a list.
    kept_rows = []
    for _, row in df_filtered.iterrows():
                dt_value = row['data_lake_table']
                q_value = row['query_table']
                # Condition (a): Always keep if data_lake_table equals the query.
                if dt_value == q_value:
                    kept_rows.append(row)
                    seen.add(dt_value)
                    
    queries=df_filtered['query_table'].unique()
    for query in queries:
            df_query = df_filtered[df_filtered['query_table'] == query]
            # Iterate row-by-row over the current query table's rows.
            kept_rows_q=[]       
            for _, row in df_query.iterrows():
                dt_value = row['data_lake_table']
            # Condition (b): Keep if the value is not in seen AND
            # the count of already kept rows is less than 11.
                if (dt_value not in seen) and (len(kept_rows_q) < 11):
                    kept_rows_q.append(row)
                    seen.add(dt_value) 
            kept_rows.extend(kept_rows_q)
            
            # Create a DataFrame from the kept rows for this query.
            # df_query_filtered = pd.DataFrame(kept_rows)
            # filtered_results.append(df_query_filtered)

    
    # Combine all filtered results.
    #df_combined = pd.concat(filtered_results, ignore_index=True)
    df_combined= pd.DataFrame(kept_rows)
    # Step 5: Group by query_table and sort by group count in descending order.
    group_counts = df_combined.groupby('query_table').size()
    # Retain only groups with more than 5 rows.
    valid_groups = group_counts[group_counts > 10].sort_values(ascending=False)
    # Select the top 50 query_table values.
    top_queries = valid_groups.head(50).index
    df_top = df_combined[df_combined['query_table'].isin(top_queries)].copy()
       # Ensure output folders exist
    os.makedirs(query_out_dir, exist_ok=True)
    os.makedirs(datalake_out_dir, exist_ok=True)

    # 4) Copy relevant query_table files into 'query_out_dir'
    #    We assume the file is named "<table_name>.csv" in 'benchmark_dir'.
    for qtbl in df_top["query_table"].unique():
        src = os.path.join(benchmark_dir, f"{qtbl}")
        dst = os.path.join(query_out_dir, f"{qtbl}")
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} not found. Skipped copying.")

    # Copy relevant data_lake_table files into 'datalake_out_dir'
    for dtbl in df_top["data_lake_table"].unique():
        src = os.path.join(benchmark_dir, f"{dtbl}")
        dst = os.path.join(datalake_out_dir, f"{dtbl}")
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} not found. Skipped copying.")

    # 5) Write the filtered rows as a new ground truth CSV
    df_top.to_csv(output_csv, index=False)

if __name__ == "__main__":

    
    keep_noverlap_copy(
        input_csv="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/table-union-search-benchmark/TUS_benchmark_relabeled_groundtruth.csv",        # CSV with the required columns
        benchmark_dir="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/table-union-search-benchmark/small/benchmark",           # Folder where all CSVs originally live
        query_out_dir="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/table-union-search-benchmark/small/query",               # Where to copy query tables
        datalake_out_dir="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/table-union-search-benchmark/small/datalake",         # Where to copy data lake tables
        output_csv="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/table-union-search-benchmark/small/tus_small_noverlap_groundtruth.csv",
        sample_size=50
    )