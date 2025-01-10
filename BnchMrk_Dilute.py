import csv
import pandas as pd
import pickle
import os


from GMC_search   import GMC_Search

# we need to call this once 
def dilute_datalake():
    
       dilation_degree=0.2
       query_directory="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/query"
       datalake_directory="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/datalake"
       diluted_datalake_directory="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/datalake_dilute"
       ground_truth_path="data/santos/santosUnionBenchmark.pickle"    
       groundtruth_dict = GMC_Search.loadDictionaryFromPickleFile(ground_truth_path)
       no_common_column={}
       missing_files_in_datalake=[]
       missing_files_in_query=[]
       
       
       
       for key_, values in groundtruth_dict.items():
         # open file  
           # Load the CSV file
           
           
           
                
          try:
                # Attempt to read the CSV file
                df_q= pd.read_csv(query_directory+"/"+key_)
                df_q.columns = [col.lower() for col in df_q.columns]
                print("File read successfully!")
          except FileNotFoundError:
                # Handle the case where the file does not exist
                print(f"Error: The file '{query_directory}/{key_}' does not exist.")
                missing_files_in_query.append(key_)
          except Exception as e:
                # Handle other potential errors
                print(f"An unexpected error occurred: {e}")
          num_rows_q = df_q.shape[0]
            # Display the DataFrame and column names
          q_cols= df_q.columns.tolist()

          for dl_tb in values:
              
              # open the file 
              try:
                  df_dl_tb= pd.read_csv(datalake_directory+"/"+dl_tb)
                  df_dl_tb.columns = [col.lower() for col in df_dl_tb.columns]

              except FileNotFoundError:
                  print(f"Error: The file '{missing_files_in_datalake}/{dl_tb}' does not exist.")
                  missing_files_in_datalake.append(dl_tb)
              except Exception as e:
                 # Handle other potential errors
                  print(f"An unexpected error occurred: {e}") 
                  
              num_rows_tb = df_dl_tb.shape[0]
              q_sample_size=int(num_rows_tb*dilation_degree)
            # Display the DataFrame and column names
              dl_tb_cols= df_dl_tb.columns.tolist()

              
            # now pick a simple random sample of size q_sample_size from df_q
              if(q_sample_size>num_rows_q):
                  q_sample_size=num_rows_q # uperbound is reached
              random_sample = df_q.sample(n=q_sample_size, random_state=42)   
              # dilute to the dl_table
              # first get the common columns 
              common_col=set(dl_tb_cols).intersection(set(q_cols))
              if(len(common_col)==0):
                 # keep the table name in a predefined list 
                print("no common columns added")
                if key_ in no_common_column:
                    no_common_column[key_].append(dl_tb)
                else:
                    no_common_column[key_]=[dl_tb]
              else:
                  # Iterate over rows of df1
                    for _, row in random_sample.iterrows():
                       
                        # Create a new row with value under the same column name
                            new_row = {col: row[col] if col.lower() in common_col else None for col in df_dl_tb.columns}

                            
                            df_dl_tb = pd.concat([df_dl_tb, pd.DataFrame([new_row])], ignore_index=True)  
                            
              output_file = diluted_datalake_directory+ "/"+dl_tb
              df_dl_tb.to_csv(output_file, index=False)  
        
        # Write the dictionary to a CSV file
       with open("not_diluted.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
    
            # Write each key and its list of values as a row
            for key, values in no_common_column.items():
                writer.writerow([key] + values)
        # Write the list to a text file
       with open("missing_files.csv", mode="w") as file:
            file.write("Missing files in data lake which exist in ground truth:\n")
            for item in missing_files_in_datalake:
                file.write(f"{item}\n")
            file.write("Missing files in query folder which exist in ground truth:\n")
            for item in missing_files_in_query:
                file.write(f"{item}\n")

def dilute_groundtruth(): 
  # now modify the ground truth: 
  
  ground_truth_path="data/santos/santosUnionBenchmark.pickle"  
  ground_truth_path_diluted="data/santos/santosUnionBenchmark_diluted.pickle"  
  groundtruth_dict = GMC_Search.loadDictionaryFromPickleFile(ground_truth_path)
  new_groundtruth_dict={}
  # 1- remove the missing files from the groundtruth : stockport_contracts_4.csv
  # 2- also for all the files add another file with _dlt added to their names  
  missings=['stockport_contracts_4.csv']
  for key_, values in groundtruth_dict.items():
      #remove msiising files
      values = list(set(values) - set(missings))

      new_values=values
      updated_file_names = [
      f"{name[:-4]}_dlt.csv" if name.endswith(".csv") else name
             for name in values]                
           
      
      new_values=new_values+updated_file_names
      new_groundtruth_dict[key_]=new_values
  
  print("done")
  with open(ground_truth_path_diluted, "wb") as file:
    pickle.dump(new_groundtruth_dict, file)
    
def dilute_dust_alignemnt():
    dust_alignemnt="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/DUST_Alignment_Santos_original.csv"   
    dust_alignemnt_dilute= "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/DUST_Alignment_Santos.csv" 
    

    if os.path.exists(dust_alignemnt_dilute):
        print("The dust alignment file is already diluted")
    else:
        print("Diluting dust alignment file starts .... ")
        
                # Input and output file names
        input_file = dust_alignemnt  
        output_file = dust_alignemnt_dilute  # Output file name

       
            # Read the input CSV file
        df = pd.read_csv(input_file)

        # Initialize an empty DataFrame with the same columns as the input DataFrame
        output_df = pd.DataFrame(columns=df.columns)
        i=0
        # Iterate over each row in the input DataFrame
        for _, row in df.iterrows():
            i=i+1
            print("iteration: "+ str(i))
            # Convert the row to a dictionary for easy manipulation
            row_dict = row.to_dict()

            # Add the original row to the output DataFrame
            output_df = pd.concat([output_df, pd.DataFrame([row_dict])], ignore_index=True)

            # Case 1: Add `_dlt` to both `query_table_name` and `dl_table_name`
            row_case1 = row_dict.copy()
            if ".csv" in row_case1["query_table_name"]:
                row_case1["query_table_name"] = row_case1["query_table_name"].replace(".csv", "_dlt.csv")
            if ".csv" in row_case1["dl_table_name"]:
                row_case1["dl_table_name"] = row_case1["dl_table_name"].replace(".csv", "_dlt.csv")
            output_df = pd.concat([output_df, pd.DataFrame([row_case1])], ignore_index=True)

            # Case 2: Add `_dlt` to `dl_table_name` only
            row_case2 = row_dict.copy()
            if ".csv" in row_case2["dl_table_name"]:
                row_case2["dl_table_name"] = row_case2["dl_table_name"].replace(".csv", "_dlt.csv")
            output_df = pd.concat([output_df, pd.DataFrame([row_case2])], ignore_index=True)

            # Case 3: Add `_dlt` to `query_table_name` only
            row_case3 = row_dict.copy()
            if ".csv" in row_case3["query_table_name"]:
                row_case3["query_table_name"] = row_case3["query_table_name"].replace(".csv", "_dlt.csv")
            output_df = pd.concat([output_df, pd.DataFrame([row_case3])], ignore_index=True)
        # Write the output DataFrame to the CSV file
        output_df.to_csv(output_file, index=False)

        print(f"Output CSV file {output_file} created successfully.")
        
    
    
    
      
           
if __name__ == "__main__":
  # we need to call these function once 
  
  dilute_datalake ()
  dilute_groundtruth()
  

    