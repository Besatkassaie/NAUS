# this file has multiple function to make the data set ready to be used in the whole Naus process 
import pandas as pd
import os
import pickle


def convert_to_dict(csv_file_path):
    """
    Reads a CSV file and converts it to a dictionary mapping each query_table 
    to a list of corresponding data_lake_table values. Only rows where 'manual_unionable'
    is 1 or 2 are considered.
    
    Parameters:
        csv_file_path (str): Path to the CSV file.
        
    Returns:
        dict: A dictionary with query_table as keys and lists of data_lake_table values as values.
    """
    # Load CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Filter rows where 'manual_unionable' is either 1 or 2
    df_filtered = df[df['manual_unionable'].isin([1, 2])]
    
    # Create dictionary mapping query_table to list of data_lake_table values
    mapping_dict = {}
    for query, group in df_filtered.groupby('query_table'):
        mapping_dict[query] = group['data_lake_table'].unique().tolist()
    
    return mapping_dict

def add_query_to_values(mapping_dict):
    """
    For each key in the mapping dictionary, checks if the query_table (key) is present 
    in its corresponding list of data_lake_table values; if not, it adds it.
    
    Parameters:
        mapping_dict (dict): A dictionary mapping query_table to a list of data_lake_table values.
        
    Returns:
        dict: The updated dictionary with each key included in its list of values.
    """
    for query, tables in mapping_dict.items():
        if query not in tables:
            tables.append(query)
    return mapping_dict


def add_suffix_to_keys_and_values(mapping_dict):
    """
    Modifies a mapping dictionary by adding '_ugen_v2' to each key and each string in the value lists.
    If a string ends with '.csv', '_ugen_v2' is inserted just before '.csv'.
    
    Parameters:
        mapping_dict (dict): A dictionary where keys are strings and values are lists of strings.
        
    Returns:
        dict: A new dictionary with updated keys and values.
    """
    new_dict = {}
    for key, values in mapping_dict.items():
        # Update key: if it ends with '.csv', insert the suffix before it
        if key.endswith('.csv'):
            new_key = key[:-4] + '_ugen_v2.csv'
        else:
            new_key = key + '_ugen_v2'
        
        # Update values in the list similarly
        new_values = []
        for val in values:
            if isinstance(val, str) and val.endswith('.csv'):
                new_val = val[:-4] + '_ugen_v2.csv'
            elif isinstance(val, str):
                new_val = val + '_ugen_v2'
            else:
                new_val = val
            new_values.append(new_val)
        
        new_dict[new_key] = new_values

    return new_dict
# Example usage:
if __name__ == "__main__":
    csv_file_path = "/u6/bkassaie/NAUS/data/ugen_v2/manual_benchmark_validation_results/ugen_v2_eval_groundtruth_with_alignments.csv"
    
    # Convert CSV to dictionary
    mapping_dict = convert_to_dict(csv_file_path)
    
    # Ensure each query is added to its list of data_lake_tables if missing
    mapping_dict = add_query_to_values(mapping_dict)
    
    # Print the top 10 query mappings
    print("query mappings:")
    for i, (query, tables) in enumerate(mapping_dict.items()):
         
        print(f"{i} {query}: {len(tables)}")
    updated_mapping = add_suffix_to_keys_and_values(mapping_dict)
    print(updated_mapping)
    # Write the DataFrame to a pickle file
    
    filename = "/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_unionable_groundtruth.pickle"

    if not os.path.exists(filename):
        with open(filename, "wb") as f:
            pickle.dump(updated_mapping, f)
        print(f"{filename} has been created.")
    else:
        print(f"{filename} already exists.")