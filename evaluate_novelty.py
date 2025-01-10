import csv
import pickle5 as p
import pandas as pd
from naive_search_Novelty import NaiveSearcherNovelty
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer


#This file containes all the metrics used in our paper to evalute the novelty/diversity 
#of the reranking 

def Avg_executiontime_by_k(resultfile, caller_alg):
        file_path = resultfile
        df = pd.read_csv(file_path, usecols=[0, 1, 2, 3])

        # Extract column names for reference
        column_names = df.columns.tolist()

        # Assuming 'k', 'query_name', and 'tables' are part of the loaded columns
        k_column = column_names[3]  # Replace with the actual column name for k
        time_column=column_names[2]
        grouped = df.groupby(k_column)[time_column].mean()

# Print the results
        for k, avg_time in grouped.items():
             print(f"k: {k}, Average execution time for {caller_alg} is: {avg_time*1000}")        


def compute_counts(dataframe, k_column,query_name_column, tables_column):
        exclude={"workforce_management_information_a.csv",
                 "workforce_management_information_b.csv"}
        print("numebr of row before excluding two queries"+ str(len(dataframe)))
        dataframe = dataframe[~dataframe[query_name_column].isin(exclude)]
        print("numebr of row after excluding two queries"+ str(len(dataframe)))

        result = []
        unique_k_values = dataframe[k_column].unique()
        for k in unique_k_values:
            filtered_df = dataframe[dataframe[k_column] == k]
            count = filtered_df.apply(
                lambda row: row[query_name_column] in row[tables_column].split(','), axis=1
            ).sum()
            result.append({'k': k, 'count': count})
        return pd.DataFrame(result)

def query_duplicate_returned(result_file, output_file):
        # Load the CSV file and read the first four columns, using the first row as column names
        file_path = result_file  # Replace with the path to your CSV file
        df = pd.read_csv(file_path, usecols=[0, 1, 2, 3])

        # Extract column names for reference
        column_names = df.columns.tolist()

        # Assuming 'k', 'query_name', and 'tables' are part of the loaded columns
        k_column = column_names[3]  # Replace with the actual column name for k
        query_name_column = column_names[0]  # Replace with the actual column name for query name
        tables_column = column_names[1]  # Replace with the actual column name for tables

   

        # Compute the results
        results_df = compute_counts(df, k_column,query_name_column, tables_column)

        # Output the results
   
        results_df.to_csv(output_file, index=False)


def analyse(file_penalize,file_gmc):


            # Load the CSV files into DataFrames
            x1 = pd.read_csv(file_penalize)  # Replace with your first CSV file name
            x2 = pd.read_csv(file_gmc)  # Replace with your second CSV file name


            # Find common queries
            common_queries = set(x1['query']).intersection(set(x2['query']))

            # Keep only rows with common queries
            x1_filtered = x1[x1['query'].isin(common_queries)]
            x2_filtered = x2[x2['query'].isin(common_queries)]

    
            
 
            # Merge filtered DataFrames on 'query' and 'k' to align rows for comparison
            merged = pd.merge(x1_filtered, x2_filtered, on=['query', 'k'], suffixes=('_x1', '_x2'))

            # Compare precision values and group by 'k' to count queries with greater precision in x1
            result = merged.groupby('k').apply(lambda df: (df['Prec_x1'] >= df['Prec_x2']).sum())

            # Display the result
            print("Number of queries with greater or equal precision in "+file_penalize+" for each k:")
            print(result)
 
            
                        # Compare precision values and group by 'k' to count queries with greater precision in x1
            result2 = merged.groupby('k').apply(lambda df: (df['Recall_x1'] >= df['Recall_x2']).sum())

            # Display the result
            print("Number of queries with greater or equal recall in "+file_penalize+" for each k:")
            print(result2)

            
    # Group by 'k' and compute statistics for Prec, Recall, and Ideal Recall
            stats_x1 = x1_filtered.groupby('k').agg({
                'Prec': ['mean', 'median', 'std'],
                'Recall': ['mean', 'median', 'std'],
                'Ideal Recall': ['mean', 'median', 'std']
            })

            # Rename columns for clarity
            stats_x1.columns = ['_'.join(col).strip() for col in stats_x1.columns.values]

            # Display the statistics
            print("Grouped Statistics for x1_filtered by k:")
            print(stats_x1)
    
    
            stats_x2 = x2_filtered.groupby('k').agg({
                'Prec': ['mean', 'median', 'std'],
                'Recall': ['mean', 'median', 'std'],
                'Ideal Recall': ['mean', 'median', 'std']
            })

            # Rename columns for clarity
            stats_x2.columns = ['_'.join(col).strip() for col in stats_x2.columns.values]

            # Display the statistics
            print("Grouped Statistics for x2_filtered by k:")
            print(stats_x2)


def loadDictionaryFromPickleFile(dictionaryPath):
        ''' Load the pickle file as a dictionary
        Args:
            dictionaryPath: path to the pickle file
        Return: dictionary from the pickle file
        '''
        filePointer=open(dictionaryPath, 'rb')
        dictionary = p.load(filePointer)
        filePointer.close()
        return dictionary

  # a function to calculate  precision recall and MAP per query
def Cal_P_R_Map(resultFile, gtPath, output_file_):
        ''' Calculate and log the performance metrics: MAP, Precision@k, Recall@k
    Args:
        max_k: the maximum K value (e.g. for SANTOS benchmark, max_k = 10. For TUS benchmark, max_k = 60)
        k_range: different k s that are reported in the result
        gtPath: file path to the groundtruth
        resPath: file path to the raw results from the model
    Return: MAP, P@K, R@K'''
        groundtruth = loadDictionaryFromPickleFile(gtPath)
    # resultFile = loadDictionaryFromPickleFile(resPath)
        file_path = resultFile
        df = pd.read_csv(file_path, usecols=[0, 1, 2, 3])

        # Extract column names for reference
        column_names = df.columns.tolist()

        # Assuming 'k', 'query_name', and 'tables' are part of the loaded columns
        k_column = column_names[3]  # Replace with the actual column name for k
        query_name_column = column_names[0]  # Replace with the actual column name for query name
        tables_column = column_names[1]  # Replace with the actual column name for tables
        unique_k_values = df[k_column].unique() # existing k
        precision_array = []
        recall_array = []
        ideal_recall=[]
        result_query_k={}
        for k in unique_k_values:
                # for this k go through the df and create a dictionary from query to list of returned tables 
                filtered_df = df[df['k'] == k]

                # Initialize the dictionary
                resultFile = {}

                # Iterate over filtered rows
                for _, row in filtered_df.iterrows():
                    query_name = row['query_name']
                    tablenames_set = set(row['tables'].split(','))  # Split and convert to set for uniqueness
                    
                    # Add or update the dictionary
                    if query_name not in resultFile:
                        resultFile[query_name] = list(tablenames_set)
                    else:
                        resultFile[query_name] = list(set(resultFile[query_name]) | tablenames_set)

                true_positive = 0
                false_positive = 0
                false_negative = 0
                rec = 0
                
                for table in resultFile:
                    # t28 tables have less than 60 results. So, skipping them in the analysis.
                        if table in groundtruth:
                            groundtruth_set = set(groundtruth[table])
                            groundtruth_set = {x.split(".")[0] for x in groundtruth_set}
                            result_set = resultFile[table][:k]
                            result_set = [x.split(".")[0] for x in result_set]
                            result_set = [item.strip() for item in result_set]
                            result_set = [item.replace("]", '') for item in result_set]


                            # find_intersection = true positives
                            find_intersection = set(result_set).intersection(groundtruth_set)
                            tp = len(find_intersection)
                            fp = k - tp
                            fn = len(groundtruth_set) - tp
                            if len(groundtruth_set)>=k: 
                                true_positive += tp
                                false_positive += fp
                                false_negative += fn
                            rec += tp / (tp+fn)
                            pecs_q_k= tp/(tp+fp) # prec for k for query
                            rec_q_k=tp / (tp+fn)
                            if(k<len(groundtruth[table])):
                                ideal_recall_q_k=k/len(groundtruth[table])
                            else:
                                ideal_recall_q_k=1
                            result_query_k[(k, table)]=[pecs_q_k,rec_q_k,ideal_recall_q_k]
                            
                ideal_recall.append(k/float(len(groundtruth[table]))) 
                precision = true_positive / (true_positive + false_positive)
                recall = rec/len(resultFile)
                precision_array.append(precision)
                recall_array.append(recall)



            
        print("k values:")
        print(unique_k_values)
        print("precision_array")
        print(precision_array)
        print("ideal_recall")
        print(ideal_recall)

        print("recall_array")
        print(recall_array)
        
        # Output CSV file name
        output_file = output_file_

        # Write data to CSV
        with open(output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            
            # Write the header
            writer.writerow(["k", "query", "Prec", "Recall", "Ideal Recall"])
            
            # Write the rows
            for (k1, k2), values in result_query_k.items():
                writer.writerow([k1, k2] + values)

        print(f"Data successfully written to {output_file}")

        #return mean_avg_pr, precision_array[max_k-1], recall_array[max_k-1] 
        return None
    

def compute_syntactic_novelty_measure_simplified(groundtruth_file, search_result, snm_avg_result_path_, snm_whole_result_path_):
        """
        this function only makes sense to be run over diluted dataset
        we assume 2 things: 1- the list of dl_tables for each query is sorted descending bu unionability score
                            2-  search_result file is a csv has atleast  these columns: query_name,	tables,	execution_time,	k  
                            groundtruth: is the gound truth havinf unionable tables for each query
        """                    
        groundtruth = loadDictionaryFromPickleFile(groundtruth_file)

        input_file = search_result
        columns_to_load = ['query_name', 'tables', 'k']
        df = pd.read_csv(input_file, usecols=columns_to_load)

        # Get the unique values of 'k' in the dataframe
        unique_k_values = df['k'].unique()

        # Initialize an empty list to collect the results
        results = []
        results_whole=[]

        # Iterate over each unique 'k' and process the dataframe
        for k in unique_k_values:
            # Filter the dataframe for the current value of 'k'
            subset_df = df[df['k'] == k]
            
            # Calculate the average SNM using the custom function
            print("k is: "+str(k))
            avg_snm, invalid_snm = get_ssnm_average(subset_df, groundtruth)
            temp_whole=get_ssnm_whole(subset_df, groundtruth, k)
            results_whole.extend(temp_whole)
            # Append the result as a dictionary
            results.append({'k': k, 'avg_snm': avg_snm, 'q_invalid_snm:':invalid_snm})
           
        # Convert the results into a dataframe
        result_df = pd.DataFrame(results)
    
        # Write the result dataframe to a new CSV file
        output_file_agv = snm_avg_result_path_
        result_df.to_csv(output_file_agv, index=False)
        
        result_whole=pd.DataFrame(results_whole)
        result_whole.to_csv(snm_whole_result_path_, index=False)

      
      
        
        
        print("caluculate Simplified SNM for the input result file for different k")

def get_ssnm_average(df_k, groundtruth_dic):
        """we go through all queries except for these two that do not exist in dust alignment 
        workforce_management_information_a.csv
        workforce_management_information_b.csv
        """
        exclude={"workforce_management_information_a.csv",
                 "workforce_management_information_b.csv"}
        queries = df_k['query_name'].unique()
        number_queries=0
        snm_total=0.0
        q_not_valid_snm=set()
        for q in queries:
                        if q not in exclude: 
                          df_k_q = df_k[df_k['query_name'] == q]
                          groundtruth_dic_q=groundtruth_dic[q]
                          q_snm,L, G=get_ssnm_query(df_k_q, groundtruth_dic_q)
                          if(q_snm==-1):
                           q_not_valid_snm.add(q)
                          else:
                            number_queries=number_queries+1
 
                            snm_total=snm_total+q_snm
                        else: 
                            print("excluded query is found")    
        return  (float(snm_total)/ float(number_queries), q_not_valid_snm) 
       
def get_ssnm_whole(df_k, groundtruth_dic, k):
        """we go through all queries except for these two that do not exist in dust alignment:
        workforce_management_information_a.csv
        workforce_management_information_b.csv
        """
        exclude={"workforce_management_information_a.csv",
                 "workforce_management_information_b.csv"}
        queries = df_k['query_name'].unique()
        results_k = []
        for q in queries:
                        if q not in exclude: 
                          df_k_q = df_k[df_k['query_name'] == q]
                          groundtruth_dic_q=groundtruth_dic[q]
                          q_snm, L, G=get_ssnm_query(df_k_q, groundtruth_dic_q)
                          results_k.append({'k': k, 'query': q,'snm': q_snm, 'L':L, 'G':G})
                        else: 
                            print("excluded query is found")    
        return   results_k    
    
    
def get_ssnm_query(df_q, groundtruth_tables):
       # list of unionable tables in groundtruth
     
       
       tables_result_list=[x.strip() for x in df_q['tables'].tolist()[0].split(',')]
       tables_result_set=set(tables_result_list)
       #these two holds the expected pair name of the visited file names which are not yet paired   
       visited_diluted_waiting_for_set=set()
       # the not deluted seen so far
       visited_no_diluted_waiting_for_set=set()
       
       groundtruth_tables_set=set(groundtruth_tables)
       G=len(tables_result_set.intersection(groundtruth_tables_set))
       L=0.0  #number of unionable diluted comes alone in result
       if (G==0):
            print("no unionable tables in the results")
            return -1, L, G # not valid snm exists for this query
       else: 
           for t in tables_result_list:
                if t in groundtruth_tables_set: # is unionable
                    deluted_=is_diluted_version(t)
                    if(deluted_==-1):
                        # check whther you have seen its diluted pair or not 
                         if(t in visited_diluted_waiting_for_set):
                                visited_diluted_waiting_for_set.remove(t)    # pair is seen so remove from  waiting list
                         else: 
                             visited_no_diluted_waiting_for_set.add(t)
                    
                    else: 
                        if deluted_ in visited_no_diluted_waiting_for_set:
                            visited_no_diluted_waiting_for_set.remove(deluted_)  # good pair original came before dilutes
                        else:    
                            visited_diluted_waiting_for_set.add(deluted_)    # 
                        
       L= len(visited_diluted_waiting_for_set) # not paired with original
       ssnm= 1-(float(L)/G)
       return ssnm, L, G   
   
   
def is_diluted_version(fname):
       """if it has _dlt showes is diluted then retrun original file name 
           else retrun -1"""
       if('_dlt' in fname) :
            return fname.replace('_dlt', '')
       else: 
        return -1    
    
    
def compute_syntactic_novelty_measure(groundtruth_file, search_result, snm_avg_result_path_, snm_whole_result_path_):
        """
        this function only makes sense to be run over diluted dataset
        we assume 2 things: 1- the list of dl_tables for each query is sorted descending bu unionability score
                            2-  search_result file is a csv has atleast  these columns: query_name,	tables,	execution_time,	k  
                            groundtruth: is the gound truth havinf unionable tables for each query
        """                    
        groundtruth = loadDictionaryFromPickleFile(groundtruth_file)

        input_file = search_result
        columns_to_load = ['query_name', 'tables', 'k']
        df = pd.read_csv(input_file, usecols=columns_to_load)

        # Get the unique values of 'k' in the dataframe
        unique_k_values = df['k'].unique()

        # Initialize an empty list to collect the results
        results = []
        results_whole=[]

        # Iterate over each unique 'k' and process the dataframe
        for k in unique_k_values:
            # Filter the dataframe for the current value of 'k'
            subset_df = df[df['k'] == k]
            
            # Calculate the average SNM using the custom function
            print("k is: "+str(k))
            avg_snm, invalid_snm = get_snm_average(subset_df, groundtruth)
            temp_whole=get_snm_whole(subset_df, groundtruth, k)
            results_whole.extend(temp_whole)
            # Append the result as a dictionary
            results.append({'k': k, 'avg_snm': avg_snm, 'q_invalid_snm:':invalid_snm})
           
        # Convert the results into a dataframe
        result_df = pd.DataFrame(results)
    
        # Write the result dataframe to a new CSV file
        output_file_agv = snm_avg_result_path_
        result_df.to_csv(output_file_agv, index=False)
        
        result_whole=pd.DataFrame(results_whole)
        result_whole.to_csv(snm_whole_result_path_, index=False)

      
      
        
        
        print("caluculate SNM for the input result file for different k")   
       
def get_snm_average(df_k, groundtruth_dic):
        """we go through all queries except for these two that do not exist in dust alignment 
        workforce_management_information_a.csv
        workforce_management_information_b.csv
        """
        exclude={"workforce_management_information_a.csv",
                 "workforce_management_information_b.csv"}
        queries = df_k['query_name'].unique()
        number_queries=0
        snm_total=0.0
        q_not_valid_snm=set()
        for q in queries:
                        if q not in exclude: 
                          df_k_q = df_k[df_k['query_name'] == q]
                          groundtruth_dic_q=groundtruth_dic[q]
                          q_snm, B, L, G=get_snm_query(df_k_q, groundtruth_dic_q)
                          if(q_snm==-1):
                           q_not_valid_snm.add(q)
                          else:
                            number_queries=number_queries+1
 
                            snm_total=snm_total+q_snm
                        else: 
                            print("excluded query is found")    
        return  (float(snm_total)/ float(number_queries), q_not_valid_snm) 
    
    
def get_snm_query(df_q, groundtruth_tables):
       # list of unionable tables in groundtruth
     
       
       tables_result_list=[x.strip() for x in df_q['tables'].tolist()[0].split(',')]
       tables_result_set=set(tables_result_list)
       #these two holds the expected pair name of the visited file names which are not yet paired   
       visited_diluted_waiting_for_set=set()
       # the not deluted seen so far
       visited_no_diluted_waiting_for_set=set()
       
       groundtruth_tables_set=set(groundtruth_tables)
       G=len(tables_result_set.intersection(groundtruth_tables_set))
       L=0.0  #number of unionable diluted comes alone in result
       B=00.0 # diluted item comes before its original peer
       if (G==0):
            print("no unionable tables in the results")
            return -1,B, L, G # not valid snm exists for this query
       else: 
           for t in tables_result_list:
                if t in groundtruth_tables_set: # is unionable
                    deluted_=is_diluted_version(t)
                    if(deluted_==-1):
                        # check whther you have seen its diluted pair or not 
                         if(t in visited_diluted_waiting_for_set):
                                B=B+1 #update B
                                visited_diluted_waiting_for_set.remove(t)    # pair is seen so remove from  waiting list
                         else: 
                             visited_no_diluted_waiting_for_set.add(t)
                    
                    else: 
                        if deluted_ in visited_no_diluted_waiting_for_set:
                            visited_no_diluted_waiting_for_set.remove(deluted_)  # good pair original came before dilutes
                        else:    
                            visited_diluted_waiting_for_set.add(deluted_)    # 
                        
       L= len(visited_diluted_waiting_for_set) # not paired with original
       snm= 1-((float(B)+float(L))/G)
       return snm, B, L, G             
   

def get_snm_whole(df_k, groundtruth_dic, k):
        """we go through all queries except for these two that do not exist in dust alignment:
        workforce_management_information_a.csv
        workforce_management_information_b.csv
        """
        exclude={"workforce_management_information_a.csv",
                 "workforce_management_information_b.csv"}
        queries = df_k['query_name'].unique()
        results_k = []
        for q in queries:
                        if q not in exclude: 
                          df_k_q = df_k[df_k['query_name'] == q]
                          groundtruth_dic_q=groundtruth_dic[q]
                          q_snm, B, L, G=get_snm_query(df_k_q, groundtruth_dic_q)
                          results_k.append({'k': k, 'query': q,'snm': q_snm, 'B':B, 'L':L, 'G':G})
                        else: 
                            print("excluded query is found")    
        return   results_k 
    
    
def compute_union_size_simple(result_file, output_file, alignments_file, query_path_ , table_path_, normalized=0):
        """computes  union size between query table and data lake tables
           result_file has at least columns: query_name, tables, and k  
           alignments is alignemnts betweeen  columns of  query and tables  has query_table_name, query_column, query_column#, dl_table_name, dl_column#, dl_column
           simple version means that we do not use outer union here so q, t1 and q,t2 might have different alignments which makes them distinct in counting
           query_path_ and table_path_ are the content of the tables if it is for raw version we deal differently than  for normlized version 
           normalized  0 means we are deeling with raw content of the tables 1 means that the content of each cell is normalized  
        """
        try:
            # Load the CSV file into a pandas DataFrame
               alignments_ = pd.read_csv(alignments_file)

            # Verify that the required columns are present
               required_columns = ['query_table_name', 'query_column', 'query_column#',
                                'dl_table_name', 'dl_column#', 'dl_column']
               if not all(column in alignments_.columns for column in required_columns):
                missing_columns = [col for col in required_columns if col not in alignments_.columns]
                raise ValueError(f"Missing required columns in data: {missing_columns}")

               print("alignments_file loaded successfully")
        
        except FileNotFoundError:
            print(f"Error: File not found at {alignments_file}")
        
        except ValueError as e:
            print(f"Error: {e}")
        
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
            
        # find out how many groups of unionable there are in the alignment file for each query: 
        # to do so find the unique combinations of  query_table_name, query_column#  for each query 

        tbles_=NaiveSearcherNovelty.read_csv_files_to_dict(table_path_)
        qs=NaiveSearcherNovelty.read_csv_files_to_dict(query_path_)
            
        # Initialize an empty DataFrame with the desired columns
        columns = ["query", "k", "union_size", "normalized"]
        df_output = pd.DataFrame(columns=columns)  
        columns_to_load = ['query_name', 'tables', 'k']
        df_search_results = pd.read_csv(result_file, usecols=columns_to_load)

        # Get the unique values of 'k' in the dataframe
        unique_k_values = df_search_results['k'].unique()    
        
        for k_ in unique_k_values:
              print("k: "+str(k_))
            # Filter the dataframe for the current value of 'k'
              k_df_search_results= df_search_results[df_search_results['k'] == k_]
              #DUST does not create alignemnt for these two queries
              exclude={"workforce_management_information_a.csv",
                 "workforce_management_information_b.csv"}
              queries_k = k_df_search_results['query_name'].unique()
              for q in queries_k:
                # get the unionabe tables 
                if q in exclude:
                    print("a query that should not exist!!!")
                else: 
                    q_k_df_search_results= k_df_search_results[k_df_search_results['query_name'] == q]
                    q_unionable_tables=q_k_df_search_results["tables"]
                    print("query table: "+q)
                    aligned_columns_tbl={}
                    q_unionable_tables_list= [x.strip() for x in q_unionable_tables.iloc[0].split(',')]
                    for dl_t in q_unionable_tables_list:
                    # get from alignmnet which columns are aligned for each table 
                      condition1 = alignments_['query_table_name'] == q
                      condition2 = alignments_['dl_table_name'] == dl_t

                      # Filter rows that satisfy both conditions
                      filtered_align = alignments_[condition1 & condition2]
                      
                      
                      aligned_q_cols =tuple (sorted(set(filtered_align['query_column#'])))
                      if aligned_q_cols in aligned_columns_tbl.keys():
                        aligned_columns_tbl[aligned_q_cols].add(dl_t)
                      else:   
                        aligned_columns_tbl[aligned_q_cols]={dl_t}
                      #aligned_columns_tbl  is a dictionary from set of columns in query to  datalake table names getting union on those columns  
                #perform union
                    (union_size, all_tables_size, query_size)=perform_union_get_size(q, aligned_columns_tbl,alignments_, qs , tbles_, normalized)    
                    
                    new_row = {
                            "query": q,
                            "query_size":query_size, 
                            "k": k_, 
                            "union_size":union_size,
                            "all_tables_size":all_tables_size,
                            "normalized":  normalized
                         }       

                    df_output = pd.concat([df_output, pd.DataFrame([new_row])], ignore_index=True)
    
                
        
        df_output.to_csv(output_file, index=False)
              

def perform_union_get_size(q_name, aligned_columns_tbl,alignments_,qs, tables, normalize): 
    # return: dataframe q_name, size of union of all the unionbales with query
        # load the content to dictionaries  

        all_tables_size=0
        union_size=0
        query_size=len(qs[q_name][0])


        for key in aligned_columns_tbl.keys():
           # key is a set of  query columns and values are table  names 
           # sort key ascendingly and make a list of columns 
            lst_query=qs[q_name]

            # Selecting the columns specified by 'key'
            lst_query_selected_cols = [lst_query[i] for i in list(key)]

            # Transpose the column-major data to row-major for DataFrame creation
            df_query = pd.DataFrame(data=list(zip(*lst_query_selected_cols)), columns=key)
            df_all=df_query
            values=    aligned_columns_tbl [key]
            # Convert the set to a list and sort it in ascending order
            sorted_list_q_cols= sorted(list(key))
            # project the query on sorted_list_q_cols and add records to the union set
            for tbl in values:
                # get the aligned columns with sorted_list_q_cols and project on them and
                condition1 = alignments_['query_table_name'] == q_name
                condition2 = alignments_['dl_table_name'] == tbl
                cols_tb=[] # same order as columns in sorted_list_q_cols
                for qcol in sorted_list_q_cols:
                    condition3= alignments_['query_column#']==qcol
                    cols_tb.append(alignments_[condition1 & condition2 & condition3]['dl_column'].values[0])
                lst_tbl=tables[tbl]
                all_tables_size=all_tables_size+len(tables[tbl][0])
                lst_tbl_selected_cols = [lst_tbl[i] for i in cols_tb]
                df_tbl = pd.DataFrame(data=list(zip(*lst_tbl_selected_cols)), columns=cols_tb)    
                df_tbl.columns=df_all.columns #doing this to align columns when appending the two dataframes 
                df_all = pd.concat([df_all, df_tbl], ignore_index=True)
                # cols_tb has the columns from dl table  
                # project dl table and append to union_datframe_partition
                

            if(normalize==1):
               df_all = df_all.applymap(normalize) 
            union_size=union_size+len(df_all.drop_duplicates())

        return (union_size, all_tables_size, query_size)                 
    
    
def normalize(cell):
        """apply three normalization  to  a cell
           1- tokenization
           2- case folding 
           3- stemming
           """
        stemmer = PorterStemmer()
           # Tokenizer to split by space, period, underscore, and dash
        tokenizer = RegexpTokenizer(r'[^ \._\-]+')
                # Tokenization
        tokens = tokenizer.tokenize(cell)
            # Case Folding
        tokens = case_fold(tokens)
            # Stemming
        stemmed_tokens = stem(stemmer, tokens)
        merged_string = ' '.join(stemmed_tokens)
        return merged_string     
    
def case_fold(tokens):
        """Converts all tokens to lowercase."""
        return [token.lower() for token in tokens]

def stem( stemmer, tokens):
        """Applies stemming to the tokens."""
        return [stemmer.stem(token) for token in tokens]          