from collections import defaultdict
from itertools import combinations
import utilities as utl 
from itertools import combinations
import pandas as pd
import pickle
import os
import shutil
import pickle
from itertools import combinations

def deduplicate_and_copy(input_dict, query_dir, datalake_dir, output_base_dir):
    # Convert all paths to absolute paths
    query_dir = os.path.abspath(query_dir)
    datalake_dir = os.path.abspath(datalake_dir)
    output_base_dir = os.path.abspath(output_base_dir)
    query_out_dir = os.path.join(output_base_dir, "query")
    datalake_out_dir = os.path.join(output_base_dir, "datalake")
    
    total_keys = len(input_dict)
    keys = list(input_dict.keys())
    to_remove = set()

    # Step 1: Deduplicate keys with overlapping values
    for i in range(total_keys):
        for j in range(i + 1, total_keys):
            key1, key2 = keys[i], keys[j]
            set1, set2 = set(input_dict[key1]), set(input_dict[key2])
            if set1 & set2:
                if key2 not in to_remove and key1 not in to_remove:
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


def filter_alignment_csv_by_dict(csv_path, filtered_dict_path, output_csv_path):
    # Load filtered dictionary
    filtered_dict= utl.loadDictionaryFromPickleFile(filtered_dict_path)
    
    # Load CSV
    df = pd.read_csv(os.path.abspath(csv_path))

    # Filter rows where query_table_name is in the filtered dict
    kept_query_tables = set(filtered_dict.keys())
    filtered_df = df[df['query_table_name'].isin(kept_query_tables)]

    # Save filtered CSV
    output_csv_path = os.path.abspath(output_csv_path)
    filtered_df.to_csv(output_csv_path, index=False)
    
    print(f"Filtered CSV saved to: {output_csv_path}")
    print(f"Kept {len(filtered_df)} rows out of {len(df)}")

    return filtered_df


unionable_tables_dic= utl.loadDictionaryFromPickleFile("/u6/bkassaie/NAUS/data/santos/santos_union_groundtruth.pickle") 
filtered_path = deduplicate_and_copy(unionable_tables_dic,
                                "data/santos/query",
                                "data/santos/datalake_notdiluted",
                                "/u6/bkassaie/NAUS/data/santos/small/" )

filter_alignment_csv_by_dict("data/santos/Manual_Alignment_4gtruth_santos.csv",
                             filtered_path,
                             "data/santos/small/Manual_Alignment_4gtruth_santos_small.csv")


    