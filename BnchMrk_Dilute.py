import csv
import pandas as pd
import pickle
import os
import utilities as utl
import glob



from GMC_search   import GMC_Search

def dilute_alignment4Groundtruth(input_file, output_file):

        
        """
        Reads a CSV file into a DataFrame, creates new rows for every existing row by modifying
        the 'dl_table_name' column (inserting '_dlt' before '.csv'), and appends the new rows to
        the original DataFrame.
        
        Parameters:
                file_path (str): The path to the CSV file.
                
            Returns:
                pd.DataFrame: A new DataFrame containing both the original rows and the new modified rows.
            """
        # Step 1: Read the CSV file into a DataFrame.
        df = pd.read_csv(input_file)
        
        # Ensure the required column exists.
        if 'dl_table_name' not in df.columns:
            raise ValueError("The CSV file must contain a column named 'dl_table_name'.")
        
        # Step 2: Create new rows by modifying 'dl_table_name'
        # We'll make a copy of the DataFrame and then modify the column.
        new_rows = df.copy()
        
        # Define a function that adds '_dlt' before '.csv'
        def add_dlt(filename):
            # Check if filename is a string and ends with '.csv'
            if isinstance(filename, str) and filename.endswith('.csv'):
                # Remove the ending '.csv' and append '_dlt.csv'
                return filename[:-4] + '_dlt.csv'
            else:
                # If the file name does not end with '.csv', return it unchanged
                return filename
        
        new_rows['dl_table_name'] = new_rows['dl_table_name'].apply(add_dlt)
        
        # Step 3: Append the new rows to the original DataFrame.
        result_df = pd.concat([df, new_rows], ignore_index=True)
        result_df.to_csv(output_file, index=False)

def process_csv_files(input_path, output_path):
    """
    Reads all CSV files from the input path, removes their first column,
    and writes them to the output path.
    """
    if  os.path.exists(output_path):
        print(f"{output_path} already exists ")
    else: 
        # Create the output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # List all files in the input directory
        for file_name in os.listdir(input_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(input_path, file_name)

                # Read the CSV file with ';' as a separator
                df = pd.read_csv(file_path, sep=';')

                # Remove the first column
                df = df.iloc[:, 1:]

                # Write the modified DataFrame to the output directory
                output_file_path = os.path.join(output_path, file_name)
                df.to_csv(output_file_path, index=False, sep=';')

                print(f"Processed and saved: {output_file_path}")

# we need to call this once 
def dilute_datalake_by_column_lable(dilation_degree,query_directory,datalake_directory,diluted_datalake_directory, ground_truth_path, dataset):
    


       groundtruth_dict = GMC_Search.loadDictionaryFromPickleFile(ground_truth_path)
       if('csv' in ground_truth_path):
          groundtruth_dict=utl.loadDictionaryFromCsvFile(ground_truth_path)
       else: 
          groundtruth_dict = GMC_Search.loadDictionaryFromPickleFile(ground_truth_path)

       no_common_column={}
       missing_files_in_datalake=[]
       missing_files_in_query=[]
       delim=','

         
   
       for key_, values in groundtruth_dict.items():
         # open file  
           # Load the CSV file
          df_q=pd.DataFrame()
          
          try:
                # Attempt to read the CSV file
                df_q= pd.read_csv(query_directory+"/"+key_,  sep=delim)
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
          df_dl_tb=pd.DataFrame()
          for dl_tb in values:
              
              # open the file 
              try:
                  df_dl_tb= pd.read_csv(datalake_directory+"/"+dl_tb,  sep=delim)
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
       with open("not_diluted_dltdegree0.4.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
    
            # Write each key and its list of values as a row
            for key, values in no_common_column.items():
                writer.writerow([key] + values)
        # Write the list to a text file
       with open("missing_files_dltdegree0.4.csv", mode="w") as file:
            file.write("Missing files in data lake which exist in ground truth:\n")
            for item in missing_files_in_datalake:
                file.write(f"{item}\n")
            file.write("Missing files in query folder which exist in ground truth:\n")
            for item in missing_files_in_query:
                file.write(f"{item}\n")
                
                
 
def dilute_datalake_by_alignment(dilation_degree,query_directory,datalake_directory,diluted_datalake_directory, ground_truth_path,alignment_file,notdiluted_file, dataset):
   
       if('csv' in ground_truth_path):
          groundtruth_dict=utl.loadDictionaryFromCsvFile_withheader(ground_truth_path)
       else: 
          groundtruth_dict = GMC_Search.loadDictionaryFromPickleFile(ground_truth_path)
       alignment_=utl.load_alignment(alignment_file)

       no_common_column={}
       missing_files_in_datalake=[]
       missing_files_in_query=[]
       delim=','

    
       for key_, values in groundtruth_dict.items():
         # open file  
           # Load the CSV file
          df_q=pd.DataFrame()
          
          try:
                # Attempt to read the CSV file
                df_q= pd.read_csv(query_directory+"/"+key_,  sep=delim)
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
          df_dl_tb=pd.DataFrame()
          for dl_tb in values:
              
              # open the file 
              try:
                  df_dl_tb= pd.read_csv(datalake_directory+"/"+dl_tb,  sep=delim)

              except FileNotFoundError:
                  print(f"Error: The file '{missing_files_in_datalake}/{dl_tb}' does not exist.")
                  missing_files_in_datalake.append(dl_tb)
              except Exception as e:
                 # Handle other potential errors
                  print(f"An unexpected error occurred: {e}") 
                  
              num_rows_tb = df_dl_tb.shape[0]
              q_sample_size=int(num_rows_tb*dilation_degree)
            # Display the DataFrame and column names

              
            # now pick a simple random sample of size q_sample_size from df_q
              if(q_sample_size>num_rows_q):
                  q_sample_size=num_rows_q # uperbound is reached
              random_sample = df_q.sample(n=q_sample_size, random_state=42)   
              # dilute to the dl_table
              # we want to get alignment between df_q and dl_table
              # columns: ['query_table_name', 'query_column', 'query_column#','dl_table_name', 'dl_column#', 'dl_column']
              # alignment_ 
              filtered_alignment_= alignment_[
                            (alignment_['query_table_name'] == key_) &
                            (alignment_['dl_table_name'] == dl_tb)
                        ]
              mapping = {row['dl_column']: row['query_column#'] for _, row in filtered_alignment_.iterrows()}

            
              if(len(filtered_alignment_)==0):
                 # keep the table name in a predefined list 
                print("no aligned columns exists")
                if key_ in no_common_column:
                    no_common_column[key_].append(dl_tb)
                else:
                    no_common_column[key_]=[dl_tb]
              else:
                  # Iterate over rows of df1
                    for _, row in random_sample.iterrows():
                            new_row = {col: None for col in df_dl_tb.columns}
                            # Create a new row with value under the same column name
                            
                            for col in df_dl_tb.columns:
                                    dl_col_index=df_dl_tb.columns.get_loc(col)
                                    if dl_col_index in mapping:
                                        query_index = mapping[dl_col_index]
                                        # Attempt to retrieve the value from source_row at the specified index
                                        try:
                                            value = row.iloc[query_index]
                                        except IndexError:
                                            value = None  # In case query_index is out of range for source_row
                                        new_row[col] = value
                                    else:
                                        new_row[col] = None                            
                            
                            df_dl_tb = pd.concat([df_dl_tb, pd.DataFrame([new_row])], ignore_index=True)  
                            
                    output_file = diluted_datalake_directory+ "/"+dl_tb
                    df_dl_tb.to_csv(output_file, index=False)  
        
        # Write the dictionary to a CSV file
       with open(notdiluted_file, mode="w", newline="") as file:
            writer = csv.writer(file)
    
            # Write each key and its list of values as a row
            for key, values in no_common_column.items():
                writer.writerow((key, values))
        # Write the list to a text file
       with open(f"missing_files_dltdegree0.4{dataset}.csv", mode="w") as file:
            file.write("Missing files in data lake which exist in ground truth:\n")
            for item in missing_files_in_datalake:
                file.write(f"{item}\n")
            file.write("Missing files in query folder which exist in ground truth:\n")
            for item in missing_files_in_query:
                file.write(f"{item}\n") 
                
def dilute_datalake_ugen(dilation_degree,query_directory,datalake_directory,diluted_datalake_directory, ground_truth_path,alignmnet_file, dataset='ugenv2'): 
       # in the ground truth  we have query and only unionables 
       groundtruth_dict=utl.loadDictionaryFromCSV_ugenv2(ground_truth_path)
       no_common_column={}
       missing_files_in_datalake=[]
       missing_files_in_query=[]
       delim=';'
       alignemnts_=pd.read_csv(alignmnet_file)
       alignemnts_ = alignemnts_[['query_table_name', 'query_column#', 'dl_table_name', 'dl_column']]

   
       for key_, values in groundtruth_dict.items():
         # open file  
           # Load the CSV file
          df_q=pd.DataFrame()
          
          try:
                # Attempt to read the CSV file
                df_q= utl.read_csv_file(query_directory+"/"+key_)
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
          df_dl_tb=pd.DataFrame()
          for dl_tb in values:
              
              # open the file 
              try:
                  df_dl_tb=  utl.read_csv_file(datalake_directory+"/"+dl_tb)
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

              
            # now pick a simple random sample of size q_sample_size from df_q
              if(q_sample_size>num_rows_q):
                  q_sample_size=num_rows_q # upperbound is reached
              random_sample = df_q.sample(n=q_sample_size, random_state=42)   
              # dilute to the dl_table
              #first filter alignments to query and datalake table alignmnet
              alignemnts_q_dlt = alignemnts_[(alignemnts_['query_table_name'] == key_) & (alignemnts_['dl_table_name'] == dl_tb)]

              # here we need to load  alignment 
              if(len(alignemnts_q_dlt)==0):
                 # keep the table name in a predefined list 
                print("no alignment exist")
                if key_ in no_common_column:
                    no_common_column[key_].append(dl_tb)
                else:
                    no_common_column[key_]=[dl_tb]
              else:
                  # Iterate over rows of df1
                    alignemnts_tuples = list(alignemnts_q_dlt[['query_column#', 'dl_column']].itertuples(index=False, name=None))

                    for _, row in random_sample.iterrows():
                         
                        # Create a new row with value under the same column name
                            new_row = {
                                col: (
                                    row[next((tup[1] for tup in alignemnts_tuples if tup[0] == idx), None)]
                                    if next((tup for tup in alignemnts_tuples if tup[0] == idx), None) is not None
                                    else None
                                )
                                for idx, col in enumerate(df_dl_tb.columns)
                            }
                            

                            
                            df_dl_tb = pd.concat([df_dl_tb, pd.DataFrame([new_row])], ignore_index=True)  
                            
              output_file = diluted_datalake_directory+ "/"+dl_tb
              df_dl_tb.to_csv(output_file,sep=';',index=False)  
        
        # Write the dictionary to a CSV file
       with open("not_diluted_dltdegree0.4_genv2.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
    
            # Write each key and its list of values as a row
            for key, values in no_common_column.items():
                writer.writerow([key] + values)
        # Write the list to a text file
       with open("missing_files_dltdegree0.4genv2.csv", mode="w") as file:
            file.write("Missing files in data lake which exist in ground truth:\n")
            for item in missing_files_in_datalake:
                file.write(f"{item}\n")
            file.write("Missing files in query folder which exist in ground truth:\n")
            for item in missing_files_in_query:
                file.write(f"{item}\n")







def dilute_groundtruth(ground_truth_path,ground_truth_path_diluted, notdiluted_tnames_file, missing_tables): 
  #  modify the ground truth to accomodate diluted tables: 
    notdiluted_tnames = {}
    notdiluted_tnames= utl.loadDictionaryFromCsvFile(notdiluted_tnames_file)
    if('csv' in ground_truth_path):
          groundtruth_dict=utl.loadDictionaryFromCsvFile_withheader(ground_truth_path)
    else: 
          groundtruth_dict = GMC_Search.loadDictionaryFromPickleFile(ground_truth_path)
    new_groundtruth_dict={}
    # 1- remove the missing files from the groundtruth santos: stockport_contracts_4.csv
    # 2- also for all the files add another file with _dlt added to their names  
    missings=missing_tables
    for key_, values in groundtruth_dict.items():
        #remove msiising files
        dlstring=""
        values = list(set(values) - set(missings))
        if(key_ in notdiluted_tnames ):
            dlstring=notdiluted_tnames[key_]
            dlstring=dlstring.replace('[\'','')
            dlstring=dlstring.replace('\']','')
            dlstring=dlstring.replace('\'','')
        

        notdiluted_tnames_q=set([item.strip() for item in dlstring.split(",")])
        new_values=values

       # now remove from values those that do not have diluted version 
        values = list(set(values) - set(notdiluted_tnames_q))
        updated_file_names = [
        f"{name[:-4]}_dlt.csv" if name.endswith(".csv") else name
                for name in values]                
            
        
        new_values=new_values+updated_file_names
        new_groundtruth_dict[key_]=new_values
    
    print("done")
    if('csv' in ground_truth_path):
        with open(ground_truth_path_diluted, mode="w", newline="") as file:
            writer = csv.writer(file)
        
        # Optionally write a header row        
            writer.writerow(["query_table", "data_lake_table"])
            
            # Write each key-value pair from the dictionary
            for key, value_ in new_groundtruth_dict.items():
                for tbl in value_: 
                    writer.writerow([key, tbl])
    else:
        with open(ground_truth_path_diluted, "wb") as file:
            pickle.dump(new_groundtruth_dict, file)
                
        
        

      
           
if __name__ == "__main__":
  # we need to call these function once 
    dataset="santos"
    dilation_degree=0.4
    # santos: 
    if dataset=='santos':
        missing_tables=['stockport_contracts_4.csv']
        ground_truth_path="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/santos_union_groundtruth.pickle"  
        ground_truth_path_diluted="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/santos_union_groundtruth.pickle_diluted.pickle"  
        query_directory="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/query"
        datalake_directory="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/datalake"
        diluted_datalake_directory="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/datalake_diluteonly_dltdeg04"
        ground_truth_path="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/santos_union_groundtruth.pickle"  
        alignmnet_file="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/Santos_CL_KMEANS_alignment_cosine.csv"
        notdiluted04_file="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/notdiluted04_file.csv"
        
        
        # if the dataset if ugenv2 we do not have queries as unionable in the data lake and also in the 
        #dilute_datalake_by_column_lable(dilation_degree,query_directory,datalake_directory,diluted_datalake_directory, ground_truth_path, dataset) 
       
             # make sure that all queries are in the 
        dilute_datalake_by_alignment(dilation_degree,query_directory, datalake_directory, 
                                       diluted_datalake_directory,ground_truth_path, alignmnet_file,notdiluted04_file,dataset)
             #dilute_groundtruth(ground_truth_path,ground_truth_path_diluted,missing_tables )

    elif dataset=='ugen-v2':
        query_directory_withindex="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/ugen_v2/query_original"
        query_directory_noindex="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/ugen_v2/query"
        datalake_directory_withindex="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/ugen_v2/datalake_original"
        datalake_directory_noindex="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/ugen_v2/datalake_nodiluted"  
        query_directory=query_directory_noindex
        datalake_directory=datalake_directory_noindex
        diluted_datalake_directory="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/ugen_v2/datalake_diluteonly_dltdegree04"
        ground_truth_path="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/ugen_v2/manual_benchmark_validation_results/ugen_v2_eval_groundtruth.csv"  
        # we eliminate all the index column form all the csv file_
        alignmnet_file="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/DUST_Alignment_4gtruth_ugen_v2.csv"

        process_csv_files(query_directory_withindex, query_directory_noindex)
        process_csv_files(datalake_directory_withindex, datalake_directory_noindex)
        dilute_datalake_ugen(dilation_degree,query_directory, datalake_directory, diluted_datalake_directory,ground_truth_path, alignmnet_file,dataset)
        
        
    elif dataset=='TUS_small':
        missing_tables=[]
        query_directory="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/table-union-search-benchmark/small/query"
        datalake_directory="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/table-union-search-benchmark/small/datalake"  
  
        diluted_datalake_directory="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/table-union-search-benchmark/small/datalake_diluted0.4_only"
        ground_truth_path="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/table-union-search-benchmark/small/tus_small_noverlap_groundtruth.csv"  
        alignmnet_file="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/table-union-search-benchmark/small/DUST_Alignment_4gtruth_tus_benchmark.csv"
        notdiluted04_file="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/table-union-search-benchmark/small/notdiluted04_file.csv"
        ground_truth_path_diluted=f"/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/table-union-search-benchmark/small/tus_small_noverlap_groundtruth_dlt_{dilation_degree}.csv"  
        alignmnet_file_diluted="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/table-union-search-benchmark/small/DUST_Alignment_4gtruth_tus_benchmark_all.csv"

    #     dilute_datalake_by_alignment(dilation_degree,query_directory, datalake_directory, 
    #                                   diluted_datalake_directory,ground_truth_path, alignmnet_file,notdiluted04_file,dataset)
      #  dilute_groundtruth(ground_truth_path,ground_truth_path_diluted,notdiluted04_file,missing_tables)
       #dilute_alignment4Groundtruth(alignmnet_file, alignmnet_file_diluted)


        

  
  

     