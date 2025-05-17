import csv
import pickle5 as p
import pandas as pd
from naive_search_Novelty import NaiveSearcherNovelty
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import numpy as np
import utilities as utl
import itertools
import time
import multiprocessing as mp
import os

# Set the multiprocessing start method to spawn to avoid CUDA reinitialization issues.
mp.set_start_method('spawn', force=True)
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, lit
# from pyspark.sql.types import StructType, StructField, StringType, ArrayType
# import pyspark.sql.functions as F

#This file containes all the metrics used in our paper to evalute the novelty/diversity 
#of the reranking 

def Avg_executiontime_by_k(resultfile, outputfile):
        file_path = resultfile
        df = pd.read_csv(file_path, usecols=[0, 1, 2, 3])

        # Extract column names for reference
        column_names = df.columns.tolist()

        # Assuming 'k', 'query_name', and 'tables' are part of the loaded columns
        k_column = column_names[3]  # Replace with the actual column name for k
        time_column=column_names[2]
        grouped = df.groupby(k_column)[time_column].mean()
        result_df = grouped.reset_index()

        # Rename the columns to 'k' and 'exec_time'
        result_df.columns = ['k', 'exec_time']

        # Write the DataFrame to a CSV file named 'outputfile.csv' without the index
        result_df.to_csv(outputfile, index=False)


# Print the results
        for k, avg_time in grouped.items():
             print(f"k: {k}, Average execution time is: {avg_time}")        

def contains_query(row):
    tables = row['tables']
    if pd.isna(tables):
        return False
    # split on commas, strip whitespace
    parts = [x.strip() for x in tables.split(',')]
    return row['query_name'] in parts


def compute_counts(dataframe, k_column,query_name_column, tables_column):
        exclude=set()
        print("numebr of row before excluding two queries"+ str(len(dataframe)))
        dataframe = dataframe[~dataframe[query_name_column].isin(exclude)]
        print("numebr of row after excluding two queries"+ str(len(dataframe)))

        result = []
        unique_k_values = dataframe[k_column].unique()
        for k in unique_k_values:
            filtered_df = dataframe[dataframe[k_column] == k]
            # count = filtered_df.apply(
            # lambda row: row[query_name_column] in [x.strip() for x in row[tables_column].split(',')],
            # axis=1
            #   ).sum()
            
            count = filtered_df.apply(contains_query, axis=1).sum()
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
    

def compute_syntactic_novelty_measure_simplified(groundtruth_file, search_result, snm_avg_result_path_, snm_whole_result_path_, remove_dup=0):
        """
        this function only makes sense to be run over diluted dataset
        we assume 2 things: 1- the list of dl_tables for each query is sorted descending bu unionability score
                            2-  search_result file is a csv has atleast  these columns: query_name,	tables,	execution_time,	k  
                            groundtruth: is the gound truth havinf unionable tables for each query
        """                    
        if('csv' in groundtruth_file ): 
            groundtruth = utl.loadDictionaryFromCsvFile(groundtruth_file)
        else: 
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
            avg_snm, invalid_snm = get_ssnm_average(subset_df, groundtruth, remove_dup)
            temp_whole=get_ssnm_whole(subset_df, groundtruth, k, remove_dup)
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

def get_ssnm_average(df_k, groundtruth_dic, remove_dup):
        """we go through all queries except for these two that do not exist in dust alignment 
        workforce_management_information_a.csv
        workforce_management_information_b.csv
        """
        exclude=set()
        queries = df_k['query_name'].unique()
        number_queries=0
        snm_total=0.0
        q_not_valid_snm=set()
        for q in queries:
                        if q not in exclude: 
                          df_k_q = df_k[df_k['query_name'] == q]
                          groundtruth_dic_q=groundtruth_dic[q]
                          q_snm,L, G=get_ssnm_query(df_k_q, groundtruth_dic_q, remove_dup)
                          if(q_snm==-1):
                           q_not_valid_snm.add(q)
                          else:
                            number_queries=number_queries+1
 
                            snm_total=snm_total+q_snm
                        else: 
                            print("excluded query is found")    
        return  (float(snm_total)/ float(number_queries), q_not_valid_snm) 
       
def get_ssnm_whole(df_k, groundtruth_dic, k, remove_duplicates):
        """we go through all queries except for these two that do not exist in dust alignment:
        workforce_management_information_a.csv
        workforce_management_information_b.csv
        """
        exclude=set()
        queries = df_k['query_name'].unique()
        results_k = []
        for q in queries:
                        if q not in exclude: 
                          df_k_q = df_k[df_k['query_name'] == q]
                          groundtruth_dic_q=groundtruth_dic[q]
                          q_snm, L, G=get_ssnm_query(df_k_q, groundtruth_dic_q, remove_duplicates)
                          results_k.append({'k': k, 'query': q,'snm': q_snm, 'L':L, 'G':G})
                        else: 
                            print("excluded query is found")    
        return   results_k    
    
    
def get_ssnm_query(df_q, groundtruth_tables, remove_duplicates):
       # list of unionable tables in groundtruth
     
       df_q.dropna(subset=['tables'], inplace=True)
       if len(df_q)==0:
           return 0, 0, 0
       tables_result_list=[x.strip() for x in df_q['tables'].tolist()[0].split(',')]
       tables_result_list = [ item.replace("[", "").replace("'", "").replace("]", "") for item in tables_result_list ]

       
       tables_result_set=set(tables_result_list)
       #these two holds the expected pair name of the visited file names which are not yet paired   
       visited_diluted_waiting_for_set=set()
       # the not deluted seen so far
       visited_no_diluted_waiting_for_set=set()
       groundtruth_tables_set=set(groundtruth_tables)

       groundtruth_tables_set = { item.replace("[", "").replace("'", "").replace("]","") for item in groundtruth_tables_set }
       tables_result_set = { item.replace("[", "").replace("'", "").replace("]","") for item in tables_result_set }
       
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
       if remove_duplicates==1:
           #consider blatant duplicates in the computation
           query_name=df_q['query_name'].tolist()[0]
           dilutedname_query_name=query_name.replace('.csv', '_dlt.csv')
           if(query_name in tables_result_list and dilutedname_query_name in tables_result_list):
               L=L+1
           if(query_name in tables_result_list and dilutedname_query_name  not in tables_result_list):
               L=L+1
       
       
       
       ssnm= 1-(float(L)/G)
       return ssnm, L, G   
   
   
def is_diluted_version(fname):
       """if it has _dlt showes is diluted then retrun original file name 
           else retrun -1"""
       if('_dlt' in fname) :
            return fname.replace('_dlt', '')
       else: 
        return -1    
    
    
def compute_syntactic_novelty_measure(groundtruth_file, search_result, snm_avg_result_path_, snm_whole_result_path_, remove_duplicate=0):
        """
        this function only makes sense to be run over diluted dataset
        we assume 2 things: 1- the list of dl_tables for each query is sorted descending bu unionability score
                            2-  search_result file is a csv has atleast  these columns: query_name,	tables,	execution_time,	k  
                            groundtruth: is the gound truth having unionable tables for each query
        """                    
        if('csv' in groundtruth_file ): 
            groundtruth = utl.loadDictionaryFromCsvFile(groundtruth_file)
        else: 
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
            avg_snm, invalid_snm = get_snm_average(subset_df, groundtruth, remove_duplicate)
            temp_whole=get_snm_whole(subset_df, groundtruth, k, remove_duplicate)
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
       
def get_snm_average(df_k, groundtruth_dic, remove_duplicate):
        """we go through all queries except for these two that do not exist in dust alignment 
        workforce_management_information_a.csv
        workforce_management_information_b.csv
        """
        exclude=set()
        queries = df_k['query_name'].unique()
        number_queries=0
        snm_total=0.0
        q_not_valid_snm=set()
        for q in queries:
                        if q not in exclude: 
                          df_k_q = df_k[df_k['query_name'] == q]
                          groundtruth_dic_q=groundtruth_dic[q]
                          q_snm, B, L, G=get_snm_query(df_k_q, groundtruth_dic_q, remove_duplicate)
                          if(q_snm==-1):
                           q_not_valid_snm.add(q)
                          else:
                            number_queries=number_queries+1
 
                            snm_total=snm_total+q_snm
                        else: 
                            print("excluded query is found")    
        return  (float(snm_total)/ float(number_queries), q_not_valid_snm) 
    
    
def get_snm_query(df_q, groundtruth_tables, remove_duplicates):
       # list of unionable tables in groundtruth
     
       df_q.dropna(subset=['tables'], inplace=True)
       if len(df_q)==0: return 0,0,0,0
       tables_result_list=[x.strip() for x in df_q['tables'].tolist()[0].split(',')]
       
       
       tables_result_list = [ item.replace("[", "").replace("'", "").replace("]", "") for item in tables_result_list ]

       tables_result_set=set(tables_result_list)
       

    
       #these two holds the expected pair name of the visited file names which are not yet paired   
       visited_diluted_waiting_for_set=set()
       # the not deluted seen so far
       visited_no_diluted_waiting_for_set=set()
       
       groundtruth_tables_set=set(groundtruth_tables)
       
       groundtruth_tables_set = { item.replace("[", "").replace("'", "").replace("]", "") for item in groundtruth_tables_set }
       tables_result_set = { item.replace("[", "").replace("'", "").replace("]", "") for item in tables_result_set }

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
       if remove_duplicates==1:
           #consider blatant duplicates in the computation
           query_name=df_q['query_name'].tolist()[0]
           dilutedname_query_name=query_name.replace('.csv', '_dlt.csv')
           if(query_name in tables_result_list and dilutedname_query_name in tables_result_list):
               if(tables_result_list.index(query_name) < tables_result_list.index(dilutedname_query_name)):
                       B=B+1
           if(query_name in tables_result_list and dilutedname_query_name  not in tables_result_list):
               L=L+1
       
       
       
       snm= 1-((float(B)+float(L))/G)
       return snm, B, L, G             
   

def get_snm_whole(df_k, groundtruth_dic, k, remove_duplicate):
        """we go through all queries except for these two that do not exist in dust alignment:
        workforce_management_information_a.csv
        workforce_management_information_b.csv
        """
        exclude=set()
        queries = df_k['query_name'].unique()
        results_k = []
        for q in queries:
                        if q not in exclude: 
                          df_k_q = df_k[df_k['query_name'] == q]
                          groundtruth_dic_q=groundtruth_dic[q]
                          q_snm, B, L, G=get_snm_query(df_k_q, groundtruth_dic_q, remove_duplicate)
                          results_k.append({'k': k, 'query': q,'snm': q_snm, 'B':B, 'L':L, 'G':G})
                        else: 
                            print("excluded query is found")    
        return   results_k 



def can_merge_sequential(row1, row2):
    """
    Determine if two rows can be merged.
    """
    for val1, val2 in zip(row1, row2):
        if pd.notna(val1) and pd.notna(val2) and val1 != val2:
            return False
    return True

def merge_rows_sequential(row1, row2):
    """
    Merge two rows into one.
    """
    return [val1 if pd.notna(val1) else val2 for val1, val2 in zip(row1, row2)]

def merge_dataframe_sequential(df):
    """
    Merge rows in a DataFrame based on the given logic.
    """
    print("performing merge of original size " +str(len(df)))
    
    
    rows = df.values.tolist()  # Convert to a list of rows for easier manipulation
    merged = True

    while merged:
        merged = False
        skip_indices = set()
        new_rows = []

        # Process rows efficiently
        for i in range(len(rows)):
            print("# number of merged so far "+ str(len(skip_indices)))
            if i in skip_indices:
                continue
            merged_row = rows[i]
            for j in range(i + 1, len(rows)):  # Only compare rows after the current one
                if j not in skip_indices and can_merge_sequential(merged_row, rows[j]):
                    merged_row = merge_rows_sequential(merged_row, rows[j])
                    skip_indices.add(j)  # Skip merged row
                    merged = True
            new_rows.append(merged_row)
        
        rows = new_rows  # Update the rows for the next iteration

    # Convert back to a DataFrame
    print("return from merger")
    return pd.DataFrame(rows, columns=df.columns)
    
    
def can_merge_rowwise(row):
    """
    Row-wise merge logic for Cartesian join.
    Returns True if two rows can be merged, False otherwise.
    """
    row1 = row['row1']
    row2 = row['row2']
    for val1, val2 in zip(row1, row2):
        if pd.notna(val1) and pd.notna(val2) and val1 != val2:
            return False
    return True

def merge_rows(row1, row2):
    """
    Merge two rows into one.
    """
    return [val1 if pd.notna(val1) else val2 for val1, val2 in zip(row1, row2)]

def merge_dataframe(df):
    """
    Merge rows in a DataFrame using a Cartesian join for efficiency.
    """
    print(f"Performing merge for original DataFrame of size: {len(df)}")

    rows = df.values.tolist()
    merged = True

    while merged:
        merged = False
        print(f"Performing merge for original DataFrame of size: {len(rows)}")
        # Create a DataFrame for the Cartesian product
        cartesian_df = pd.DataFrame({
            'row1': rows,
        }).merge(pd.DataFrame({
            'row2': rows,
        }), how='cross')

        # Apply merge logic row-wise
        cartesian_df['can_merge'] = cartesian_df.apply(can_merge_rowwise, axis=1)

        # Identify mergeable pairs
        mergeable_pairs = cartesian_df[cartesian_df['can_merge']]

        if not mergeable_pairs.empty:
            merged = True

            # Take the first mergeable pair and merge
            merged_rows = set()
            new_rows = []
            for _, row in mergeable_pairs.iterrows():
                if tuple(row['row1']) not in merged_rows and tuple(row['row2']) not in merged_rows:
                    new_row = merge_rows(row['row1'], row['row2'])
                    new_rows.append(new_row)
                    merged_rows.add(tuple(row['row1']))
                    merged_rows.add(tuple(row['row2']))

            # Add rows that were not merged
            for row in rows:
                if tuple(row) not in merged_rows:
                    new_rows.append(row)

            rows = new_rows

    # Convert back to a DataFrame
    print(f"Returning merged DataFrame of size: {len(rows)}")
    return pd.DataFrame(rows, columns=df.columns)



def merge_dataframe_spark(df, spark_session=None):
    """
    Merges rows in a DataFrame using PySpark. Accepts either pandas or PySpark DataFrame and
    includes conversion logic.
    
    :param df: Input DataFrame (pandas or PySpark)
    :param spark_session: SparkSession instance (required if input is pandas DataFrame)
    :return: Merged PySpark DataFrame
    """
    # Ensure the input is a Spark DataFrame
    if spark_session is None:
        spark_session = SparkSession.builder.appName("MergeRows").master("local[10]"). config("spark.driver.memory", "60g") \
    .config("spark.executor.memory", "40g") \
    .config("spark.driver.maxResultSize", "40g")\
    .getOrCreate()
    df = spark_session.createDataFrame(df)

   
    def merge_rows(row1, row2):
        """
        Custom logic for merging two rows.
        """
        merged_row = []
        for val1, val2 in zip(row1, row2):
            if val1 is None and val2 is None:
                merged_row.append(None)
            elif val1 is None:
                merged_row.append(val2)
            elif val2 is None:
                merged_row.append(val1)
            elif val1 == val2:
                merged_row.append(val1)
            else:
                # Return None if values conflict (no merge)
                return None
        return merged_row

    # Define a UDF to apply merge_rows logic
    def merge_udf(row1, row2):
        merged = merge_rows(row1, row2)
        return merged

    merge_udf_spark = F.udf(merge_udf, ArrayType(StringType()))

    merged = True

    while merged:
        # Perform Cartesian join
        cartesian_df = df.alias("row1").crossJoin(df.alias("row2"))

        # Apply merge_rows logic row-wise
        cartesian_df = cartesian_df.withColumn(
            "merged",
            merge_udf_spark(
                F.struct(*[col(f"row1.{c}") for c in df.columns]),
                F.struct(*[col(f"row2.{c}") for c in df.columns])
            )
        )
        print("merged ")
        # Separate merged rows and unmerged rows
        merged_rows_ = cartesian_df.filter(cartesian_df["merged"].isNotNull()).select("merged").distinct()
        merged_rows=merged_rows_.collect()
        unmerged_rows = df.subtract(
            merged_rows.selectExpr([f"merged[{i}] AS {col}" for i, col in enumerate(df.columns)])
        )

        # Combine merged and unmerged rows
        df = merged_rows.union(unmerged_rows)

        # If no new merges occurred, stop the loop
        if merged_rows.count() == 0:
            merged = False

    # Convert back to PySpark DataFrame format
    return df.select([F.col(f"merged[{i}]").alias(df.columns[i]) for i in range(len(df.columns))])


def perform_concat(q_name,dl_t_name,filtered_align,df_query,df_dl, normalized):
    """This function do a special concat and retrun the result as dataframe"""   
    print("performing concat for datalake table: "+dl_t_name)
    q_num_columns = df_query.shape[1]
    #retrive tuples (q_col#, dl_col#) from alignamnet
    
    # Create a list of tuples (query_column#, dl_column)
    aligned_tuple_list = [(row['query_column#'], row['dl_column']) for _, row in filtered_align.iterrows()]

    # Sort the list by the first element of each tuple (query_column#)
    sorted_tuple_list = sorted(aligned_tuple_list, key=lambda x: x[0])
    mapping = dict(sorted_tuple_list)
    
    # Mapping from q_df column indices to dl_df column names
    #mapping = {0: 'a', 1: 'b'}  # Map q_df column 0 to dl_df column 'a', column 1 to 'b'

    # Create a new DataFrame for dl_df with columns arranged as per mapping
    mapped_dl_df = pd.DataFrame()

    for col_index_q in range(q_num_columns):
         q_column_name = df_query.columns[col_index_q]
         if col_index_q in mapping:
            dl_col = mapping[col_index_q]
            mapped_dl_df[q_column_name]= df_dl.iloc[:, dl_col]
         else:
            mapped_dl_df[q_column_name] = np.nan # Column doesn't exist in  mapping


    # Concatenate q_df and the newly arranged dl_df
    result_df = pd.concat([df_query, mapped_dl_df], axis=0)
    if(normalized==1):
               print("performing normalization")
               result_df = result_df.applymap(normalize) 
    # Reset index to make it consistent
    result_df.reset_index(drop=True, inplace=True)
    return result_df
    
##
# These functions must be defined elsewhere in your code.
# They are used by the per‑query processing function.
# from your_module import perform_concat, nscore_table


# These functions must be defined elsewhere in your code.
# They are used by the per‑query processing function.
# from your_module import perform_concat, nscore_table

def process_single_query(args):
    """
    Process a single query to compute its nscore.
    
    Parameters:
      args: a tuple containing:
        - q: the query name
        - k_df_search_results: DataFrame filtered for the current k value
        - alignments_: the alignments DataFrame
        - qs: dictionary of query tables (from NaiveSearcherNovelty)
        - tbles_: dictionary of datalake tables (from NaiveSearcherNovelty)
        - normalized: flag indicating if normalization is applied
        - alph: parameter alpha for nscore_table
        - beta: parameter beta for nscore_table
        - batchsize: the quesry batch size
        - output_folder_by_query: output_folder_by_query

    Returns:
      The computed nscore (float) for the given query.
    """
    q, k, k_df_search_results, alignments_, qs, tbles_, normalized, alph, beta,batchsize,output_folder_by_query = args
    start_time = time.time_ns()
    # Filter the search results for the current query.
    q_k_df_search_results = k_df_search_results[k_df_search_results['query_name'] == q]
    q_unionable_tables = q_k_df_search_results["tables"]
    print("Processing query table: " + q)
    
    # Get the unionable tables for the query (assumes a comma-separated string)
    q_unionable_tables_list = [x.strip() for x in q_unionable_tables.iloc[0].split(',')]
    
    # Build the query DataFrame
    lst_query = qs[q]
    columns_ = list(range(len(lst_query)))
    df_query = pd.DataFrame(data=list(zip(*lst_query)), columns=columns_)
    df_constructed_table = df_query
    print("Number of rows in query dataframe: " + str(len(df_query)))
    
    # Process each datalake table for the query.
    for dl_t in q_unionable_tables_list:
        # Get alignment rows for this query and datalake table.
        condition1 = alignments_['query_table_name'] == q
        condition2 = alignments_['dl_table_name'] == dl_t
        filtered_align = alignments_[condition1 & condition2]
        
        # Clean up table name
        dl_t_clean = dl_t.replace("[", "").replace("'", "").replace("]", "")
        dl_t_clean = dl_t_clean.replace("]", "").replace("'", "").replace("]", "")
        
        # Build the datalake table DataFrame.
        dl_list = tbles_[dl_t_clean]
        columns_dl = list(range(len(dl_list)))
        df_dl = pd.DataFrame(data=list(zip(*dl_list)), columns=columns_dl)
        
        print("Number of rows in constructed dataframe before concat: " + str(len(df_constructed_table)))
        # Concatenate using your function (assumed to be defined elsewhere)
        df_constructed_table = perform_concat(q, dl_t_clean, filtered_align, df_constructed_table, df_dl, normalized)
    
    # Compute the nscore for the constructed table.
    nscore_query = nscore_table(df_constructed_table, alph, beta)

    # Compute the processing time in seconds
    time_taken = round((time.time_ns() - start_time) / 10**9, 2)
    # Number of rows in the constructed DataFrame
    num_rows = len(df_constructed_table)
    # Number of unique rows (using drop_duplicates)
    num_unique = len(list(itertools.combinations(df_constructed_table.index, 2)))
    output_df = pd.DataFrame({
        "K": [k],
        "time_taken": [time_taken],
        "query_name": [q],
        "num_rows": [num_rows],
        "num_unique": [num_unique],
        "nscore": [nscore_query],
    })

    # Write output to a CSV file named after the query (e.g., "myquery.csv")
    output_filename = f"{q}_k{k}_NQ{batchsize}.csv"
    output_df.to_csv(os.path.join(output_folder_by_query,output_filename), index=False)
    print(f"Written output file for query '{q}' to {output_filename}")
    
    return nscore_query

def choose_queries(N, Queries, excluded_queries):
    """
    Returns the top N queries after filtering out the excluded ones and sorting the remaining 
    queries first by name (alphabetically) then by their size in descending order.
    
    Parameters:
    N : int
        Number of returned queries.
    Queries : dict
        A dictionary where keys are query names and values are lists of columns.
    excluded_queries : list or set
        The queries that need to be excluded.
    
    Returns:
    list
        List of top query names after filtering and sorting.
    """
    
    # Filter out the excluded queries.
    filtered_queries = {query: rows for query, rows in Queries.items() if query not in excluded_queries}
    
    # First, sort by query name (alphabetically).
    sorted_by_name = sorted(filtered_queries.items(), key=lambda item: item[0])
    
    # Then, sort by size (length of list) in descending order. 
    # Because Python's sort is stable, the alphabetical order is preserved among equal sizes.
    sorted_items = sorted(sorted_by_name, key=lambda item: len(item[1][0])) #item [1][0] would be the number of rows in the first column: e.g ('t_1934eacab8c57857____c10_0____0.csv', [[...], [...], [...], [...], [...], [...], [...], [...], [...], [...]])
    
    # Return the list of query names for the top N items.
    return [(query,len(columns[0])) for query, columns in sorted_items[:N]]


def nscore_result(output_folder_by_query,result_file, output_file, alignments_file, query_path_, table_path_, alph, beta, Ks={2},Q_N=10,normalized=0):
    """
    Compute the nscore for a given query and its unionable tables.
    This version parallelizes the per‑query computation.

    Parameters:
      output_folder_by_query: the folder to write partial results by query
      result_file: CSV file containing search results (with columns 'query_name', 'tables', 'k')
      output_file: path to the CSV file to write the averaged nscore per k value
      alignments_file: CSV file containing alignment information
      query_path_: path to query tables (to be loaded by NaiveSearcherNovelty)
      table_path_: path to datalake tables (to be loaded by NaiveSearcherNovelty)
      alph, beta: parameters for the nscore_table function
      normalized: flag for normalization

    Returns:
      A dictionary mapping k to the average nscore.
    """
    try:
        alignments_ = pd.read_csv(alignments_file)
        required_columns = ['query_table_name', 'query_column', 'query_column#',
                            'dl_table_name', 'dl_column#', 'dl_column']
        if not all(col in alignments_.columns for col in required_columns):
            missing = [col for col in required_columns if col not in alignments_.columns]
            raise ValueError(f"Missing required columns in data: {missing}")
        print("Alignments file loaded successfully")
        print(f"working on {result_file}")
    except FileNotFoundError:
        print(f"Error: File not found at {alignments_file}")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # Load datalake and query tables using NaiveSearcherNovelty.
    tbles_ = NaiveSearcherNovelty.read_csv_files_to_dict(table_path_)
    qs = NaiveSearcherNovelty.read_csv_files_to_dict(query_path_)
    
    #set the returned number to 2
    excluded_queries = set()
    top_queries=choose_queries(Q_N,qs,excluded_queries)
    top_query_names=[q[0] for q in top_queries]
    # Load search results.
    columns_to_load = ['query_name', 'tables', 'k']
    df_search_results = pd.read_csv(result_file, usecols=columns_to_load)
    
    # Process for each unique k value.
    result = {}
    # In your code you force unique_k_values to {2, 3}.
    unique_k_values = Ks
    
    for k_ in unique_k_values:
        print("Processing k: " + str(k_))
        k_df_search_results = df_search_results[df_search_results['k'] == k_]
        # Exclude queries not processed by DUST.

        queries_k = k_df_search_results['query_name'].unique()
        if len(queries_k) == 0:
            print("No query found for k = " + str(k_))
            continue
    
        #include only the ones that are in top_query_name
        queries_k = np.intersect1d(queries_k, np.array(list(top_query_names)))
        
        number_of_queries = len(queries_k)
        print(f"Found {number_of_queries} queries for k = {k_}")
        
        # Prepare arguments for each query.
        args_list = [
            (q, k_, k_df_search_results, alignments_, qs, tbles_, normalized, alph, beta,Q_N,output_folder_by_query)
            for q in queries_k
        ]
        
        # Use multiprocessing to process queries in parallel.
        with mp.Pool() as pool:
            query_nscores = pool.map(process_single_query, args_list)
        
        # Average the nscore over all queries.
        sum_nscore = sum(query_nscores)
        avg_nscore = sum_nscore / number_of_queries
        result[k_] = avg_nscore
        print(f"k {k_}  avg_nscore {avg_nscore}")
    
    # Write the results to the output CSV.
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in result.items():
            writer.writerow([key, value])
    
    return result    
def nscore_result_old(result_file, output_file, alignments_file, query_path_ , table_path_, alph, beta,normalized=0):
    #this function computse the nscore for a given query and its unionable tables 
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
        columns = ["query", "k", "null_union_size", "normalized"]
        df_output = pd.DataFrame(columns=columns)  
        columns_to_load = ['query_name', 'tables', 'k']
        df_search_results = pd.read_csv(result_file, usecols=columns_to_load)

        # Get the unique values of 'k' in the dataframe
        unique_k_values = df_search_results['k'].unique()    
        result={}
        unique_k_values={2, 3}
        for k_ in unique_k_values:
              print("k: "+str(k_))
            # Filter the dataframe for the current value of 'k'
              k_df_search_results= df_search_results[df_search_results['k'] == k_]
              #DUST does not create alignemnt for these two queries
              exclude=set()
              queries_k = k_df_search_results['query_name'].unique()
              if(len(queries_k==0)):
                  print("no query ")
              # remove excluded 
              queries_k = np.setdiff1d(queries_k, np.array(list(exclude)))

              number_of_queries=len(queries_k)
              sum_nscore=0
              for q in queries_k:
                # get the unionabe tables 
                    q_k_df_search_results= k_df_search_results[k_df_search_results['query_name'] == q]
                    q_unionable_tables=q_k_df_search_results["tables"]
                    print("query table: "+q)
                    q_unionable_tables_list= [x.strip() for x in q_unionable_tables.iloc[0].split(',')]
                    lst_query=qs[q]
                    # Create column names as the indices of the inner lists
                    columns_ = [i for i in range(len(lst_query))]

                     # Transpose the column-major data to row-major for DataFrame creation
                    df_query = pd.DataFrame(data=list(zip(*lst_query)), columns=columns_)
                    #intially has tha same column as the query but grow horizontally
                    df_constructed_table=df_query
                    print("number of rows in the query dataframe+ "+str(len(df_query)))

                    for dl_t in q_unionable_tables_list:
                    # get from alignmnet which columns are aligned for each table 
                      condition1 = alignments_['query_table_name'] == q
                      condition2 = alignments_['dl_table_name'] == dl_t

                      # Filter rows that satisfy both conditions
                      filtered_align = alignments_[condition1 & condition2]
                      # get datalake table and make if dataframe
                      dl_t=dl_t.replace("[", "").replace("'", "").replace("]","")
                      dl_t=dl_t.replace("]", "").replace("'", "").replace("]", "")
                      dl_list=tbles_[dl_t]
                      columns_dl= [i for i in range(len(dl_list))]
                      df_dl = pd.DataFrame(data=list(zip(*dl_list)), columns=columns_dl)


                      print("number of rows in the concatinated dataframe+ "+str(len(df_constructed_table)))
                      df_constructed_table=perform_concat(q,dl_t,filtered_align,df_constructed_table,df_dl, normalized )
                    nscore_query=nscore_table(df_constructed_table,alph, beta)  
                    sum_nscore= nscore_query+sum_nscore
              avg_sncore=  sum_nscore/number_of_queries
              result[k_]=avg_sncore  
              print(f"k {k_}  avg_sncore {avg_sncore}")  
                
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for key, value in result.items():
                writer.writerow([key, value])
        return result

def nscore_table(table_datafram, alph, beta):
    
    l = len(table_datafram)
    expected_pairs = l * (l - 1)

    pairs = list(itertools.permutations(table_datafram.index, 2))
    # number_of_pairs = len(pairs)
    # #compute number of pairs
    # unordered_pairs = list(itertools.combinations(table_datafram.index, 2))

    assert len(pairs) == expected_pairs, f"Expected {expected_pairs} pairs, but got {len(pairs)}"

    # Store the number of unique pairs
    number_of_unique_pairs = len(pairs)
    print("Number of unique pairs:", number_of_unique_pairs)
    print ("Number of table rows:",number_of_unique_pairs)

# Create a dictionary to store the max nscore for each i
    min_scores = {}

# Iterate over all pairs and update the maximum nscore for each i
    for i, j in pairs:
        row1 = table_datafram.loc[i]
        row2 = table_datafram.loc[j]
        score = nscore_pair(row1, row2, alph, beta)
        if i not in min_scores or score < min_scores[i]:
            min_scores[i] = score
    
    assert len(min_scores) == len(table_datafram), f"Expected {len(table_datafram)} number of max score, but got {len(min_scores)}"

    average_min_score = sum(min_scores.values()) / len(min_scores)    
    return average_min_score



def nscore_pair(row1, row2, alph, beta):

    # Create boolean masks for NaN values in each row.
    na1 = row1.isna()
    na2 = row2.isna()
    
    # Identify columns where one is NaN and the other is not.
    one_nan = na1 ^ na2
    
    # Identify columns where both are non-NaN and different.
    both_non_nan_diff = (~na1 & ~na2) & (row1 != row2)
    
    # Compute the total score.
    total = alph * both_non_nan_diff.sum() + beta * one_nan.sum()
    
    # Return the average score per column.
    return total / len(row1)

def compute_union_size_with_null(result_file, output_file, alignments_file, query_path_ , table_path_, normalized=0):
        """computes  union size between query table and data lake tables
           result_file has at least columns: query_name, tables, and k  
           alignments is alignemnts betweeen  columns of  query and tables  has query_table_name, query_column, query_column#, dl_table_name, dl_column#, dl_column
           normalized  0 means we are deeling with raw content of the tables 1 means that the content of each cell is normalized  
           multiset if 0 eans that we get rid of duplicate rows before reporting the uniona size
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
        columns = ["query", "k", "null_union_size", "normalized"]
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
              exclude=set()
              queries_k = k_df_search_results['query_name'].unique()
              for q in queries_k:
                # get the unionabe tables 
                if q in exclude:
                    print("a query that should not exist!!!")
                else: 
                    q_k_df_search_results= k_df_search_results[k_df_search_results['query_name'] == q]
                    q_unionable_tables=q_k_df_search_results["tables"]
                    print("query table: "+q)
                    q_unionable_tables_list= [x.strip() for x in q_unionable_tables.iloc[0].split(',')]
                    lst_query=qs[q]
                    # Create column names as the indices of the inner lists
                    columns_ = [i for i in range(len(lst_query))]

                     # Transpose the column-major data to row-major for DataFrame creation
                    df_query = pd.DataFrame(data=list(zip(*lst_query)), columns=columns_)
                    #intially has tha same column as the query but grow horizontally and vertically
                    df_constructed_table=df_query
                    print("number of rows in the query dataframe+ "+str(len(df_query)))

                    for dl_t in q_unionable_tables_list:
                    # get from alignmnet which columns are aligned for each table 
                      condition1 = alignments_['query_table_name'] == q
                      condition2 = alignments_['dl_table_name'] == dl_t

                      # Filter rows that satisfy both conditions
                      filtered_align = alignments_[condition1 & condition2]
                      # get datalake table and make if dataframe
                      dl_t=dl_t.replace("[", "").replace("'", "").replace("]","")
                      dl_t=dl_t.replace("]", "").replace("'", "").replace("]", "")
                      dl_list=tbles_[dl_t]
                      columns_dl= [i for i in range(len(dl_list))]
                      df_dl = pd.DataFrame(data=list(zip(*dl_list)), columns=columns_dl)


                      print("number of rows in the concatinated dataframe+ "+str(len(df_constructed_table)))
                      df_constructed_table=perform_concat(q,dl_t,filtered_align,df_constructed_table,df_dl, normalized )
                      # union size will have the final value in the last iteration
                      union_size=len(df_constructed_table.drop_duplicates())
                    
   

                    # union_result_null=merge_dataframe_spark(df_constructed_table)
                    
                    columns = ["query", "k", "null_union_size", "normalized"]

                    print ("done with the merge")
                    new_row = {
                            "query": q,
                            "k": k_, 
                            "null_union_size":union_size,
                            "all_tables_size_bofore_duplicate":df_constructed_table.shape[0],
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
        merged_string=cell        
        try:
            tokens = tokenizer.tokenize(cell)
                # Case Folding
            tokens = case_fold(tokens)
                # Stemming
            stemmed_tokens = stem(stemmer, tokens)
            merged_string = ' '.join(stemmed_tokens)
        except Exception as e:
                print("\033[31m exception happeded in normalization\033[0m")
                if(cell==np.nan):
                    merged_string=cell
        return merged_string     
    
def case_fold(tokens):
        """Converts all tokens to lowercase."""
        return [token.lower() for token in tokens]

def stem( stemmer, tokens):
        """Applies stemming to the tokens."""
        return [stemmer.stem(token) for token in tokens]          
    

if __name__=='__main__':



  
    
    gmc_diversity_data_path="data/ugen_v2/ugenv2_small/diveristy_data/search_results/GMC/"
    penalized_diversity_data_path="data/ugen_v2/ugenv2_small/diveristy_data/search_results/Penalized/"
    starmie_diversity_data_path="data/ugen_v2/ugenv2_small/diveristy_data/search_results/Starmie/"
    semNovelty_diversity_data_path="data/ugen_v2/ugenv2_small/diveristy_data/search_results/semanticNovelty/"
    starmie0_diversity_data_path="data/ugen_v2/ugenv2_small/diveristy_data/search_results/starmie0/"
    starmie1_diversity_data_path="data/ugen_v2/ugenv2_small/diveristy_data/search_results/starmie1/"
    
    gmc_result_file="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/diveristy_data/search_results/GMC/gmc_results_diluted04_restricted.csv"
    penalize_result_file="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/diveristy_data/search_results/Penalized/search_result_penalize_04diluted_restricted_pdeg1.csv"
    starmie_result_file=f"{starmie_diversity_data_path}starmie_results_04diluted_restricted.csv"
    semNovelty_result_file="data/ugen_v2/ugenv2_small/diveristy_data/search_results/semanticNovelty/search_result_semNovelty_04diluted_restricted_pdeg1.csv"
    starmie0_result_file="data/ugen_v2/ugenv2_small/diveristy_data/search_results/starmie0/search_result_starmie0_04diluted_restricted_pdeg1.csv"
    starmie1_result_file="data/ugen_v2/ugenv2_small/diveristy_data/search_results/starmie1/search_result_starmie1_04diluted_restricted_pdeg1.csv"


    from multiprocessing import freeze_support
    freeze_support()  # optional on some platforms
    alpha=1.0
    beta=0.9

    
    diveristy_path=starmie1_diversity_data_path
    res_file=starmie1_result_file
    
    output_folder=os.path.join(diveristy_path,"nscore")
    Q_N= 19 #number of queries
    k_=2
    Ks={k_}# top k
    nscore_result(output_folder,res_file,
                diveristy_path+f"/nscore_04diluted_restricted_notnormal_K{k_}_QN{Q_N}_parallel.csv", 
                                            "/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/ugenv2_small_manual_alignment_all.csv",
                                            "/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/query", 
                                            "/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/datalake",alpha,beta,Ks,Q_N,0) 
    
