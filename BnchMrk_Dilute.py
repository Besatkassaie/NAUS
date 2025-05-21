import csv
import pandas as pd
import pickle
import os
import utilities as utl
import glob





def dilute_alignmentfile(input_file, output_file):

        
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
                
                
 
def refine_ground_truth(missing_queries, ground_truth_path, refined_ground_truth_path):
       if('csv' in ground_truth_path):
          groundtruth_dict=utl.loadDictionaryFromCsvFile(ground_truth_path)
       else: 
          groundtruth_dict = GMC_Search.loadDictionaryFromPickleFile(ground_truth_path)
        
        # Iterate over a copy of the keys to safely modify the dictionary.
       for key in list(groundtruth_dict.keys()):
            # Remove the entire entry if the key is in missing_queries.
            if key in missing_queries:
                del groundtruth_dict[key]
            else:
                # Now, if the value is a list, filter out any element that is in missing_queries.
                value = groundtruth_dict[key]
                if isinstance(value, list):
                    filtered_value = [v for v in value if v not in missing_queries]
                    groundtruth_dict[key] = filtered_value
                # (Optional) If the value is a single string and you wish to remove it if it's missing:
                elif isinstance(value, str) and value in missing_queries:
                    del groundtruth_dict[key]
     # Write the updated dictionary to a pickle file
       if('csv' in ground_truth_path):
           utl.write_dict_to_csv(groundtruth_dict, refined_ground_truth_path)
       else: 
         with open(refined_ground_truth_path, 'wb') as f:
            pickle.dump(groundtruth_dict, f)
           
    

def dilute_datalake_by_alignment(dilation_degree,query_directory,datalake_directory,diluted_datalake_directory, ground_truth_path,alignment_file,notdiluted_file, dataset, missingfiles):
   
       if('csv' in ground_truth_path):
          groundtruth_dict=utl.loadDictionaryFromCsvFile(ground_truth_path)
       else: 
          groundtruth_dict = utl.loadDictionaryFromPickleFile(ground_truth_path)
       alignment_=utl.load_alignment(alignment_file)

       no_common_column={}
       missing_files_in_datalake=[]
       missing_files_in_query=[]
       delim=','
       if dataset=='ugen-v2' or dataset=="ugen-v2_small":
        delim=';'
 
       
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
                    df_dl_tb.to_csv(output_file, sep=delim, index=False)  
        
        # Write the dictionary to a CSV file
       if not os.path.exists(notdiluted_file):
          with open(notdiluted_file, 'w') as f:
             f.write("") 
       with open(notdiluted_file, mode="w", newline="") as file:
            writer = csv.writer(file)
    
            # Write each key and its list of values as a row
            for key, values in no_common_column.items():
                writer.writerow((key, values))
        # Write the list to a text file
       with open(missingfiles, mode="w") as file:
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







def dilute_groundtruth(ground_truth_path,ground_truth_path_diluted, notdiluted_tnames_file): 
  #  modify the ground truth to accomodate diluted tables: 
    notdiluted_tnames = {}
    notdiluted_tnames= utl.loadDictionaryFromCsvFile(notdiluted_tnames_file)
    if('csv' in ground_truth_path):
          groundtruth_dict=utl.loadDictionaryFromCsvFile(ground_truth_path)
    else: 
          groundtruth_dict = utl.loadDictionaryFromPickleFile(ground_truth_path)
    new_groundtruth_dict={}
    # 1- remove the missing files from the groundtruth santos: stockport_contracts_4.csv
    # 2- also for all the files add another file with _dlt added to their names  
    missings={}
    for key_, values in groundtruth_dict.items():
        #remove msiising files
        dlstring=""
        values = list(set(values) - set(missings))
        if(key_ in notdiluted_tnames ):
            dlstring=notdiluted_tnames[key_]
            if isinstance(dlstring, list):
               dlstring=dlstring[0]
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
                
        
        


def add_dlt_to_csv_filenames(folder_path):
    # Iterate over all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file has a .csv extension
        if filename.endswith('.csv'):
            # Split the filename into the base name and extension
            base, ext = os.path.splitext(filename)
            # Create the new filename by appending '_dlt' before the extension
            new_filename = f"{base}_dlt{ext}"
            # Form full file paths
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} --> {new_filename}")      
           


def all_query_in_datalake(query_folder, datalake_folder):
        # Get a set of file names (base names) from the query folder
    query_files = {os.path.basename(f) for f in glob.glob(os.path.join(query_folder, "*")) if os.path.isfile(f)}
    
    # Get a set of file names (base names) from the data lake folder
    datalake_files = {os.path.basename(f) for f in glob.glob(os.path.join(datalake_folder, "*")) if os.path.isfile(f)}
    
    # Find files that are in the query folder but not in the data lake folder
    missing_files = query_files - datalake_files

    if missing_files:
        print("The following files in the query folder are missing in the data lake folder:")
        for file in sorted(missing_files):
            print(file)
        return False    
    else:
        print("All files in the query folder are present in the data lake folder.")
        return True

def verify_groungtruth(groundtruth_file, query_folder):
    _, ext = os.path.splitext(groundtruth_file)  # Extract file extension
    ext = ext.lower()  # Convert to lowercase for consistency
    
    if ext == ".csv":
           query_to_datalake_dict = utl.loadDictionaryFromCsvFile(groundtruth_file)
    elif ext==".pickle" or ext==".pkl" :
             query_to_datalake_dict = utl.loadDictionaryFromPickleFile(groundtruth_file)
 
        # Get a set of all file names (base names) in the query folder
    query_files = {os.path.basename(f) for f in glob.glob(os.path.join(query_folder, "*")) if os.path.isfile(f)}
    
    # Check 1: Ensure every file in the query folder is a key in the dictionary.
    missing_keys = [fname for fname in query_files if fname not in query_to_datalake_dict]
    if missing_keys:
        print("The following query files are missing as keys in the dictionary:")
        for key in missing_keys:
            print(f"  - {key}")
        return False    
    else:
        print("All query files are present as keys in the dictionary.")
    
    # Check 2: Ensure that for each key, the corresponding list contains that key.
    missing_self = [key for key, table_list in query_to_datalake_dict.items() if key not in table_list]
    # check to make sur ethe missing is not query_table -> data_lake_table
    if 'query_table' in missing_keys:
        missing_self.remove('query_table')
    if len(missing_self)>0:
      
        print("\nThe following dictionary keys do not appear in their associated list:")
        for key in missing_self:
            print(f"  - {key} -> {query_to_datalake_dict[key]}")
            return False 
    else:
        print("All dictionary keys are mapped to lists that contain themselves.")
    return True

if __name__ == "__main__":
  # we need to call these function once 
    dataset="santos_small"
    dilation_degree=0.4
    # santos: 
    if dataset=='santos':
        missing_tables=[]
        ground_truth_path="/u6/bkassaie/NAUS/data/santos/santos_union_groundtruth.pickle"  
        ground_truth_path_diluted="/u6/bkassaie/NAUS/data/santos/santos_union_groundtruth.pickle_diluted.pickle"  
        query_directory="/u6/bkassaie/NAUS/data/santos/query"
        datalake_directory="/u6/bkassaie/NAUS/data/santos/datalake"
        diluted_datalake_directory="/u6/bkassaie/NAUS/data/santos/datalake_diluteonly_dltdeg04"
        alignmnet_file="/u6/bkassaie/NAUS/data/santos/Manual_Alignment_4gtruth_santos.csv"
        alignmnet_file_diluted="/u6/bkassaie/NAUS/data/santos/Manual_Alignment_4gtruth_santos_all.csv"
        notdiluted04_file="/u6/bkassaie/NAUS/data/santos/notdiluted04_file.csv"
        missingfiles=f"/u6/bkassaie/NAUS/data/santos/missing_files_dltdegree{dilation_degree}_{dataset}.csv"
        #refined_ground_truth_path="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/santos_union_refined_groundtruth.pickle"  

        #depricated: dilute_datalake_by_column_lable(dilation_degree,query_directory,datalake_directory,diluted_datalake_directory, ground_truth_path, dataset) 
       
       ####################################start verification#################################################### 
        queries_are_duplicated=all_query_in_datalake(query_directory,datalake_folder=datalake_directory)
        if queries_are_duplicated:
            # make sure for every query you have mapping beween queries in the ground truth 
            grthrut_has_query=verify_groungtruth(ground_truth_path, query_directory)   
            if grthrut_has_query:
               print("there is mapping in the groundtruth for blatant duplicates") 
            else:
                raise RuntimeError("there is missing mapping in the groundtruth for blatant duplicates")
   
        else: 
            raise RuntimeError("blatant duplicate not in the folders")
        print("dataset and grnd truth file Passed the test ")  
        
        ####################################end verification#################################################### 
        
       
        #1-make sure that all queries are in the 
        #dilute_datalake_by_alignment(dilation_degree,query_directory, datalake_directory, 
                                       #diluted_datalake_directory,ground_truth_path, alignmnet_file,notdiluted04_file,dataset, missingfiles)
        
        # remove from ground truth the queries that do not have their files in datalake  listed in missingfiles:
        # I manullay open the file and copied not existed files 
        # missin_files_names={"stockport_contracts_4.csv"}
        # refine_ground_truth(missin_files_names,ground_truth_path, refined_ground_truth_path)
         
         # modify the file names 
        #add_dlt_to_csv_filenames(diluted_datalake_directory)
        
        #dilute_groundtruth(ground_truth_path,ground_truth_path_diluted,notdiluted04_file )
        dilute_alignmentfile(alignmnet_file, alignmnet_file_diluted)


    elif dataset=='santos_small':
        missing_tables=[]
        ground_truth_path="/u6/bkassaie/NAUS/data/santos/small/santos_small_union_groundtruth.pkl"  
        ground_truth_path_diluted="/u6/bkassaie/NAUS/data/santos/small/santos_small_union_groundtruth_diluted.pickle"  
        query_directory="/u6/bkassaie/NAUS/data/santos/small/query"
        datalake_directory="/u6/bkassaie/NAUS/data/santos/small/datalake"
        diluted_datalake_directory="/u6/bkassaie/NAUS/data/santos/small/datalake_diluteonly_dltdeg04"
        alignmnet_file="/u6/bkassaie/NAUS/data/santos/small/Manual_Alignment_4gtruth_santos_small.csv"
        alignmnet_file_diluted="/u6/bkassaie/NAUS/data/santos/small/Manual_Alignment_4gtruth_santos_small_all.csv"
        notdiluted04_file="/u6/bkassaie/NAUS/data/santos/small/notdiluted04_file.csv"
        missingfiles=f"/u6/bkassaie/NAUS/data/santos/small/missing_files_dltdegree{dilation_degree}_{dataset}.csv"
       
       ####################################start verification#################################################### 
        queries_are_duplicated=all_query_in_datalake(query_directory,datalake_folder=datalake_directory)
        if queries_are_duplicated:
            # make sure for every query you have mapping beween queries in the ground truth 
            grthrut_has_query=verify_groungtruth(ground_truth_path, query_directory)   
            if grthrut_has_query:
               print("there is mapping in the groundtruth for blatant duplicates") 
            else:
                raise RuntimeError("there is missing mapping in the groundtruth for blatant duplicates")
   
        else: 
            raise RuntimeError("blatant duplicate not in the folders")
        print("dataset and grnd truth file Passed the test ")  
        
        ####################################end verification#################################################### 
        
       
        #1-make sure that all queries are in the 
        # dilute_datalake_by_alignment(dilation_degree,query_directory, datalake_directory, 
        #                                diluted_datalake_directory,ground_truth_path, alignmnet_file,notdiluted04_file,dataset, missingfiles)
        
        # remove from ground truth the queries that do not have their files in datalake  listed in missingfiles:
        # I manullay open the file and copied not existed files 
        # missin_files_names={"stockport_contracts_4.csv"}
        # refine_ground_truth(missin_files_names,ground_truth_path, refined_ground_truth_path)
         
         # modify the file names 
        #add_dlt_to_csv_filenames(diluted_datalake_directory)
        
        #dilute_groundtruth(ground_truth_path,ground_truth_path_diluted,notdiluted04_file )
        #dilute_alignmentfile(alignmnet_file, alignmnet_file_diluted)
 

    elif dataset=='ugen-v2':

        query_directory="/u6/bkassaie/NAUS/data/ugen_v2/query"
        datalake_directory="/u6/bkassaie/NAUS/data/ugen_v2/datalake"
        diluted_datalake_directory="/u6/bkassaie/NAUS/data/ugen_v2/datalake_dilutedonly_dg0.4"
        ground_truth_path="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_unionable_groundtruth.pickle"  
        ground_truth_path_diluted="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_unionable_groundtruth_diluted.pickle"
        # we eliminate all the index column form all the csv file_
        alignmnet_file="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_CL_KMEANS_cosine_alignment_notdiluted.csv"
        alignmnet_file_diluted="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_CL_KMEANS_cosine_alignment_diluted.csv"
        notdiluted04_file="/u6/bkassaie/NAUS/data/ugen_v2/notdiluted04_file.csv"
        missingfiles=f"/u6/bkassaie/NAUS/data/ugen_v2/missing_files_dltdegree{dilation_degree}_{dataset}.csv"

      
        #process_csv_files(query_directory_withindex, query_directory_noindex)
        #process_csv_files(datalake_directory_withindex, datalake_directory_noindex)
        # not used anymore:  dilute_datalake_ugen(dilation_degree,query_directory, datalake_directory, diluted_datalake_directory,ground_truth_path, alignmnet_file)
        # dilute_datalake_by_alignment(dilation_degree,query_directory, datalake_directory, 
        #                               diluted_datalake_directory,ground_truth_path, alignmnet_file,notdiluted04_file,dataset, missingfiles)
        #add_dlt_to_csv_filenames(diluted_datalake_directory)
        #dilute_groundtruth(ground_truth_path,ground_truth_path_diluted,notdiluted04_file )
        #dilute_alignmentfile(alignmnet_file, alignmnet_file_diluted)
        
        
    elif dataset=='TUS_small':
        missing_tables=[]
        query_directory="/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/query"
        datalake_directory="/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/datalake"  
  
        diluted_datalake_directory="/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/datalake_diluted0.4_only"
        ground_truth_path="data/table-union-search-benchmark/small/tus_small_noverlap_groundtruth_not_dlt.csv"  
        alignmnet_file="data/table-union-search-benchmark/small/Manual_Alignment_4gtruth_tus_benchmark.csv"
        notdiluted04_file="/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/notdiluted04_file.csv"
        ground_truth_path_diluted=f"/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/tus_small_noverlap_groundtruth_dlt_{dilation_degree}.csv"  
        alignmnet_file_diluted="/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/manual_alignment_tus_benchmark_all.csv"

        missingfiles=f"/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/missing_files_dltdegree{dilation_degree}_{dataset}.csv"
        refined_ground_truth_path="/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/tus_small_noverlap_refined_groundtruth.csv"  

       
        ####################################start verification#################################################### 
        queries_are_duplicated=all_query_in_datalake(query_directory,datalake_folder=datalake_directory)
        if queries_are_duplicated:
            # make sure for every query you have mapping beween queries in the ground truth 
            grthrut_has_query=verify_groungtruth(ground_truth_path, query_directory)   
            if grthrut_has_query:
               print("there is mapping in the groundtruth for blatant duplicates") 
            else:
                raise RuntimeError("there is missing mapping in the groundtruth for blatant duplicates")
   
        else: 
            raise RuntimeError("blatant duplicate not in the folders")
        print("dataset and grnd truth file Passed the test ")  
        
        ####################################end verification#################################################### 
        
        
        # dilute_datalake_by_alignment(dilation_degree,query_directory, datalake_directory, 
        #                                 diluted_datalake_directory,ground_truth_path, alignmnet_file,notdiluted04_file,dataset, missingfiles)
        
        # remove from ground truth the queries that do not have their files in datalake  listed in missingfiles:
        # I manullay open the file and copied not existed files 
        # missin_files_names={"stockport_contracts_4.csv"}
        #refine_ground_truth(missin_files_names,ground_truth_path, refined_ground_truth_path)
         
         # modify the file names 
        #add_dlt_to_csv_filenames(diluted_datalake_directory)
        #dilute_groundtruth(ground_truth_path,ground_truth_path_diluted,notdiluted04_file )
        # the next call only make sense when we have manula alignemnt otherwise we should call the alignemnt algorithm again 
        # dilute_alignmentfile(alignmnet_file, alignmnet_file_diluted)
        
        
    elif dataset=='ugen-v2_small':

        query_directory="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/query"
        datalake_directory="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/datalake"
        diluted_datalake_directory="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/datalake_dilutedonly_dg0.4"
        ground_truth_path="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/ugenv2_small_unionable_groundtruth.pickle"  
        ground_truth_path_diluted="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/ugenv2_small_unionable_groundtruth_diluted.pickle"
        # we eliminate all the index column form all the csv file_
        alignmnet_file="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/ugenv2_small_manual_alignment.csv"
        #alignmnet_file_diluted="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/ugenv2_CL_KMEANS_cosine_alignment_diluted.csv"
        notdiluted04_file="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/notdiluted04_file.csv"
        missingfiles=f"/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/missing_files_dltdegree{dilation_degree}_{dataset}.csv"

      
        #process_csv_files(query_directory_withindex, query_directory_noindex)
        #process_csv_files(datalake_directory_withindex, datalake_directory_noindex)
        #dilute_datalake_by_alignment(dilation_degree,query_directory, datalake_directory, 
        #                              diluted_datalake_directory,ground_truth_path, alignmnet_file,notdiluted04_file,dataset, missingfiles)
        #add_dlt_to_csv_filenames(diluted_datalake_directory)
        dilute_groundtruth(ground_truth_path,ground_truth_path_diluted,notdiluted04_file )





        

  
  

     