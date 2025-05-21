import pickle
import bz2
import os
import csv
import numpy as np
import pandas as pd
import _pickle as cPickle
from numpy.linalg import norm
from collections import defaultdict

def saveDictionaryAsPickleFile(dictionary, dictionaryPath):
    if dictionaryPath.rsplit(".")[-1] == "pickle":
        filePointer=open(dictionaryPath, 'wb')
        pickle.dump(dictionary,filePointer, protocol=pickle.HIGHEST_PROTOCOL)
        filePointer.close()
    else: #pbz2 format
        with bz2.BZ2File(dictionaryPath, "w") as f: 
            cPickle.dump(dictionary, f)

def loadDictionaryFromPickleFile(dictionaryPath):
    print("Loading dictionary at:", dictionaryPath)
    if (dictionaryPath.rsplit(".")[-1] == "pickle" or dictionaryPath.rsplit(".")[-1] == "pkl"):
        filePointer=open(dictionaryPath, 'rb')
        dictionary = pickle.load(filePointer)
        filePointer.close()
    else: #pbz2 format
        dictionary = bz2.BZ2File(dictionaryPath, "rb")
        dictionary = cPickle.load(dictionary)
    print("The total number of keys in the dictionary are:", len(dictionary))
    return dictionary

def load_alignment(alignmnet_file):
    """
    Load data from the specified source.

    The schema for the data is expected as:
    ['query_table_name', 'query_column', 'query_column#', 
    'dl_table_name', 'dl_column#', 'dl_column']
    """
    try:
        # Load the CSV file into a pandas DataFrame
        data = pd.read_csv(alignmnet_file)

        # Verify that the required columns are present
        required_columns = ['query_table_name', 'query_column', 'query_column#',
                            'dl_table_name', 'dl_column#', 'dl_column']
        if not all(column in data.columns for column in required_columns):
            missing_columns = [col for col in required_columns if col not in data.columns]
            raise ValueError(f"Missing required columns in data: {missing_columns}")

        print("alignment Data loaded successfully")
        return data
    
    except FileNotFoundError:
        print(f"Error: File not found at {alignmnet_file}")
    
    except ValueError as e:
        print(f"Error: {e}")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def loadDictionaryFromCsvFile(file_path):
    """
    Reads a CSV file and creates a dictionary where:
    - The first column serves as the key.
    - The second column values are stored as a list.

    If the file does not exist, it prints an error message.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None  # Return None if the file doesn't exist

    data_dict = defaultdict(list)  # Dictionary with lists as default values

    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:  # Ensure the row has at least two columns
                key, value = row[0].strip(), row[1].strip()
                data_dict[key].append(value)  # Append value to the list for the key
            else:
                print(f"Warning: Skipping malformed row {row}")

    return dict(data_dict)

def loadDictionaryFromCsvFile_withheader(filePath):
    with open(filePath, 'r') as f:
            # Read the file as a DataFrame
            main_df = pd.read_csv(f)
            groundtruth_dict = main_df.groupby('query_table')['data_lake_table'].apply(list).to_dict()
    return groundtruth_dict

def loadDictionaryFromCSV_ugenv2(filepath):
    # Path to the main CSV file
    main_csv_file = filepath
    print("testing common names")
    # Open the main CSV file
    with open(main_csv_file, 'r') as f:
        # Read the file as a DataFrame
        main_df = pd.read_csv(f)
        
    # Filter rows where the third column is 1 or 2 mean uionable: 1 means joinable (or unionable with 1-column) and 2 means unionable.
    filtered_df = main_df[(main_df['manual_unionable'] == 1) | (main_df['manual_unionable'] == 2)]
    filtered_df = filtered_df[['query_table', 'data_lake_table']]
    groundtruth_dict = filtered_df.groupby('query_table')['data_lake_table'].apply(list).to_dict()
    #add '_ugen_v2' to all file names 

    groundtruth_dict = {
        f"{key[:-4]}_ugen_v2.csv": [f"{value[:-4]}_ugen_v2.csv" for value in values]
        for key, values in groundtruth_dict.items()
    }
   
    return groundtruth_dict

def CosineSimilarity(array1, array2):
    return np.dot(array1,array2)/(norm(array1)*norm(array2))

def read_csv_file(gen_file):
    data = []
    try:
        if "ugen_v2" in gen_file:
           return pd.read_csv(gen_file, sep=';')
        data = pd.read_csv(gen_file, lineterminator='\n', low_memory=False)
        if data.shape[1] < 2:
            data = pd.read_csv(gen_file, sep='|')
    except:
        try:
            data = pd.read_csv(gen_file, sep='|')
        except:
            with open(gen_file) as curr_csv:
                curr_data = curr_csv.read().splitlines()
                curr_data = [len(row.split('|')) for row in curr_data]
                max_col_num = 0
                if len(curr_data) != 0:
                    max_col_num = max(curr_data)
                try:
                    if max_col_num != 0:
                        df = pd.read_csv(gen_file, sep='|', header=None, names=range(max_col_num), low_memory=False)
                        data = df
                        return data
                    else:
                        df = pd.read_csv(gen_file, lineterminator='\n', low_memory=False)
                        data = df
                        return data
                except:
                    df = pd.read_csv(gen_file, lineterminator='\n', low_memory=False)
                    data = df
                    return data
    return data