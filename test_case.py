import pickle
import evaluate_novelty
import os 

# # Path to the pickle file
# pickle_file = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/processed/santos/q_tbls_processed_set.pkl"
# data={}
# # Load the pickle file into a list
# with open(pickle_file, 'rb') as file:
#     data = pickle.load(file)

# # Check if the data is a list (optional)
# if isinstance(data, list):
#     print("Data successfully loaded into a list!")
# else:
#     print("The loaded data is not a list. Please check the pickle file.")

# # Print the loaded list (optional)
# print(data)
# gtpath="data/santos/santosUnionBenchmark.pickle"
# ugenv2="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/ugen_v2/groundtruth_old.pickle"

# groundtruth_dict = evaluate_novelty.loadDictionaryFromPickleFile(gtpath)
# groundtruth_dict_ugenv2 = evaluate_novelty.loadDictionaryFromPickleFile(ugenv2)
# #go through all the queries and filter based on having more than 10 unionables 
# output_dict={}
# for item in groundtruth_dict: 
#     if(len(item)<11):
#         print("found one")
#     else: 
#        print(len(groundtruth_dict[item]))     
    
    

ugenv2="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/ugen_v2/manual_benchmark_validation_results/ugen_v2_eval_groundtruth.csv"

import utilities as utl


import pandas as pd
import csv



# Path to the main CSV file
main_csv_file = ugenv2
print("testing common names")
# Open the main CSV file
with open(main_csv_file, 'r') as f:
    # Read the file as a DataFrame
    main_df = pd.read_csv(f, header=None, names=['query_table',	'data_lake_table','manual_unionable'])
    
# Filter rows where the third column is 1 or 2
filtered_df = main_df[(main_df['manual_unionable'] == '1') | (main_df['manual_unionable'] == '2')]
# folder 
folder="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/ugen_v2/datalake/"
# Iterate over the filtered rows
for _, row in filtered_df.iterrows():
    file1, file2 = row['query_table'], row['data_lake_table']
    base_name, ext = os.path.splitext(file1)
        
        # Construct the new file name by adding '_ugen_v2'
    file1 = f"{base_name}_ugen_v2{ext}"
    
    file1=folder+file1
    base_name, ext = os.path.splitext(file2)
        
        # Construct the new file name by adding '_ugen_v2'
    file2 = f"{base_name}_ugen_v2{ext}"
    file2=folder+file2
    try:
        # Open the CSV files corresponding to the first and second column values
        df1 = pd.read_csv(file1,
                           sep=';',  # Use ';' as the separator
    quoting=3,  # 3 corresponds to csv.QUOTE_NONE (no special quote processing)
    quotechar='',  # Disable quote character processing
    engine='python'  # Use Python engine for advanced quoting options
           )
                          
                        
    
        
        df2 = pd.read_csv(file2,
                           sep=';',  # Use ';' as the separator
    quoting=3,  # 3 corresponds to csv.QUOTE_NONE (no special quote processing)
    quotechar='',  # Disable quote character processing
    engine='python'  # Use Python engine for advanced quoting options
           )
        
        # Get the column names of both files
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        
        # Find common column names
        common_columns = cols1.intersection(cols2)
        
        # Output the result
        print(f"Common columns between '{file1}' and '{file2}': {common_columns}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error while processing '{file1}' and '{file2}': {e}")
    



