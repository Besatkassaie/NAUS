import pandas as pd
from naive_search_Novelty import NaiveSearcherNovelty
import test_naive_search_Novelty
from SetSimilaritySearch import SearchIndex
from  process_column import TextProcessor
import pickle
import os
from preprocess_align import gmc_alignmnet_by_query_efficient
from preprocess_align import initialize_globally
import csv
import numpy as np
import time
from numpy.linalg import norm
from scipy.spatial import distance
from collections import Counter
import utilities as utl
import pickle5 as p
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import sys
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import argparse
import logging
from pathlib import Path

# Set the multiprocessing start method to spawn to avoid CUDA reinitialization issues.
mp.set_start_method('spawn', force=True)


from SetSimilaritySearch import all_pairs
class GMC_Search:
    """
    GMC_Search: A class for performing Greedy Marginal Contribution (GMC) for Novelty based unionable table search.
    """

    def __init__(self, q_dl_alignemnt_file, domain_size, dl_table_vectors_path, 
                 query_table_vectors_path, benchmark_folder, search_parameters=None):
        """
        Initialize GMC_Search instance.

        Args:
            q_dl_alignemnt_file: Path to query-datalake alignment file
            domain_size: Size of domain for diversity calculations
            dl_table_vectors_path: Path to datalake table vectors
            query_table_vectors_path: Path to query table vectors
            benchmark_folder: Path to benchmark data folder
            search_parameters: Optional dictionary of search parameters
        """
        self.k = 0
        self.q_dl_alignemnt_file = q_dl_alignemnt_file
        self.search_parameters = search_parameters or {}
        self.results = []
        self.q_dl_alignemnt = utl.load_alignment(self.q_dl_alignemnt_file)
        self.unionability = None
        self.diversity = None
        self.diversity_scores = None
        self.unionability_scores = None
        self.unionable_data = None
        self.Dsize = domain_size

        # Load vector representations
        self.starmie_vectors = self.load_starmie_vectors(dl_table_vectors_path, query_table_vectors_path)

        # Load benchmark data
        table_path = benchmark_folder + "/vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl"
        table_path_raw = benchmark_folder + "/datalake"
        processed_path = benchmark_folder + "/proccessed/"
        self.load_all_data_elements(table_path, table_path_raw, processed_path)

    def load_all_data_elements(self, table_path, table_path_raw, processed_path):
        """Load all necessary data elements for the search."""
        text_processor = TextProcessor()
        self.tables_raw = NaiveSearcherNovelty.read_csv_files_to_dict(table_path_raw)
        
        # Process tables and store results
        dl_tbls_processed_set_file_name = "dl_tbls_processed_set.pkl"
        self.table_raw_proccessed_los = test_naive_search_Novelty.getProcessedTables(
            text_processor, dl_tbls_processed_set_file_name, processed_path, 
            self.tables_raw, "los", 1, 1
        )

        dl_tbls_processed_lol_file_name = "dl_tbls_processed_lol.pkl"
        self.dl_tbls_processed_lol_file_name = dl_tbls_processed_lol_file_name
        self.table_raw_lol_proccessed = test_naive_search_Novelty.getProcessedTables(
            text_processor, dl_tbls_processed_lol_file_name, processed_path,
            self.tables_raw, "lol", 1, 1
        )

    def load_starmie_vectors(self, dl_table_vectors_path, query_table_vectors_path):
        """Load vector representations for datalake and query tables."""
        with open(query_table_vectors_path, 'rb') as f:
            queries = pickle.load(f)
        queries_dict = {item[0]: item[1] for item in queries}

        with open(dl_table_vectors_path, 'rb') as f:
            tables = pickle.load(f)
        dl_dict = {item[0]: item[1] for item in tables}
        
        return (queries_dict, dl_dict)

    def _cosine_sim(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    def calculate_unionability(self, query_name, dl_table_name):
        """Calculate unionability score between query and datalake table."""
        all_vectors = self.starmie_vectors
        queries_dict = all_vectors[0]
        dl_dict = all_vectors[1]
        
        q_vectors = queries_dict[query_name]
        query_rows = self.q_dl_alignemnt[self.q_dl_alignemnt['query_table_name'] == query_name]
        similarity_score = 0.0

        dl_t_vectors = dl_dict[dl_table_name]
        specific_rows = query_rows[query_rows['dl_table_name'] == dl_table_name]

        num_rows = float(specific_rows.shape[0])
        if num_rows == 0:
            return 0.0

        for _, row in specific_rows.iterrows():
            query_column = row['query_column#']
            dl_column = row['dl_column']
            similarity_col = self._cosine_sim(q_vectors[query_column], dl_t_vectors[dl_column])
            similarity_score += similarity_col

        return similarity_score

    def calculate_diversity(self, query_name, s_i, s_j):
        """Calculate diversity score between two datalake tables."""
        print("Computing diversity scores for given s_i and s_j wrt query..")
        DSize = self.Dsize
        div_score = 0.0

        # Get alignments for both tables
        query_rows_si = self.q_dl_alignemnt[
            (self.q_dl_alignemnt['query_table_name'] == query_name) & 
            (self.q_dl_alignemnt['dl_table_name'] == s_i)
        ]
        query_rows_sj = self.q_dl_alignemnt[
            (self.q_dl_alignemnt['query_table_name'] == query_name) & 
            (self.q_dl_alignemnt['dl_table_name'] == s_j)
        ]

        aligned_col_q_si = set(query_rows_si['query_column#'])
        aligned_col_q_sj = set(query_rows_sj['query_column#'])
        alignemd_same_qcolumns = aligned_col_q_si.intersection(aligned_col_q_sj)

        if len(alignemd_same_qcolumns) == 0:
            return 0.0

        for query_column_number in alignemd_same_qcolumns:
            s_i_column_number = query_rows_si[query_rows_si['query_column#'] == query_column_number]['dl_column'].item()
            s_j_column_number = query_rows_sj[query_rows_sj['query_column#'] == query_column_number]['dl_column'].item()

            dl_column_s_i = self.table_raw_lol_proccessed.get(s_i)[s_i_column_number]
            dl_column_set_s_i = self.table_raw_proccessed_los.get(s_i)[s_i_column_number]
            dl_column_s_j = self.table_raw_lol_proccessed.get(s_j)[s_j_column_number]
            dl_column_set_s_j = self.table_raw_proccessed_los.get(s_j)[s_j_column_number]

            domain_estimate = set.union(set(dl_column_s_j), set(dl_column_s_i))
            if len(domain_estimate) < DSize:
                distance = self.Jensen_Shannon_distances(dl_column_s_i, dl_column_s_j, domain_estimate)
            else:
                distance = 1 - self._lexicalsim_Pair(dl_column_set_s_i, dl_column_set_s_j)
            div_score += distance

        return div_score

    def Jensen_Shannon_distances(self, query_column, dl_column, domain_estimate):
        """Calculate Jensen-Shannon distance between two columns."""
        x_axis = {item: i for i, item in enumerate(domain_estimate)}
        
        frequency_q = dict(Counter(query_column))
        frequency_dl = dict(Counter(dl_column))
        
        list_length_q = len(query_column)
        list_length_dl = len(dl_column)
        
        array_q = np.zeros(len(domain_estimate))
        array_dl = np.zeros(len(domain_estimate))
        
        for item in domain_estimate:
            index_ = x_axis[item]
            array_q[index_] = frequency_q.get(item, 0) / float(list_length_q)
            array_dl[index_] = frequency_dl.get(item, 0) / float(list_length_dl)
        
        return distance.jensenshannon(array_q, array_dl)

    def _lexicalsim_Pair(self, query_column, table_column):
        """Calculate lexical similarity between two columns."""
        sets = [query_column, set(table_column)]
        pairs = all_pairs(sets, similarity_func_name="jaccard", similarity_threshold=0.0)
        l_pairs = list(pairs)
        return l_pairs[0][2] if l_pairs else 0

    def mmc_compute_div_sum(self, s_i: str, q_name, R_p: set) -> float:
        """Compute sum of diversity scores for a table with respect to selected set."""
        total_score = 0
        for s_j in R_p:
            total_score += self.calculate_diversity(q_name, s_i, s_j)
        return total_score

    def mmc_compute_div_large(self, s_i: str, q_name, remaining_s_set: set, max_l: int) -> float:
        """Compute diversity scores for remaining tables."""
        div_l_list = []
        for s_j in remaining_s_set:
            div_l_list.append(self.calculate_diversity(q_name, s_i, s_j))
        div_l_list = sorted(div_l_list, reverse=True)[:max_l - 1]
        return div_l_list

    def mmc(self, q_name, s_set: set, lmda: float, k: int, R_p: set) -> dict:
        """Compute Marginal Marginal Contribution for each table."""
        all_mmc = dict()
        p = len(R_p) - 1
        div_coefficient = lmda / (k - 1)
        
        for s_i in s_set:
            sim_term = (1 - lmda) * self.calculate_unionability(q_name, s_i)
            div_term1 = div_coefficient * self.mmc_compute_div_sum(s_i, q_name, R_p)
            temp_s = set(s_set) - R_p - {s_i}
            temp = self.mmc_compute_div_large(s_i, q_name, temp_s, k - p)
            temp_sum = sum(temp)
            div_term2 = div_coefficient * temp_sum
            current_mmc = sim_term + div_term1 + div_term2
            all_mmc[s_i] = current_mmc
            
        return all_mmc

    def gmc(self, S_names: set, query_name: str, k: int, 
            lmda: float = 0.7, metric: str = "cosine", 
            print_results: bool = False) -> tuple:
        """Perform Greedy Marginal Contribution search."""
        start_time = time.time_ns()
        R = set()
        ranked_div_result = []

        S_set = S_names
        for p in range(0, k):
            if len(S_set) == 0:
                break
            mmc_dict = self.mmc(query_name, S_set, lmda, k, R)
            s_i = max(mmc_dict, key=lambda k: mmc_dict[k])
            R.add(s_i)
            ranked_div_result.append(s_i)
            S_set = set(S_set) - {s_i}
            
        end_time = time.time_ns()
        total_time = round((end_time - start_time) / 10 ** 9, 2)
        if print_results:
            print("Total time taken: ", total_time, " seconds.")
        
        return ranked_div_result, total_time

    def load_unionable_tables(self, path):
        """Load the mapping between query and its unionable tables."""
        self.unionable_data = utl.loadDictionaryFromPickleFile(path)

    def execute_search(self):
        """Execute the GMC search for all queries."""
        k = self.k
        lmda = 0.7  # best practice from original paper
        all_results = {}
        queries = self.unionable_data.keys()
        
        for query in queries:
            S = self.unionable_data[query]
            (gmc_results, time_took) = self.gmc(S, query, k=k, lmda=lmda)
            all_results[query] = (gmc_results, time_took)
            
        return all_results

    def load_unionable_tables(self, path):
        #load the mapping between query and its unionnable tables generated by a system like Starmie
        
         self.unionable_data= utl.loadDictionaryFromPickleFile(path)    
    
    def sum_vectors(self, emb_dic, table_name, columns):
        table_vector=np.zeros(768)             
        for _col in columns:
            table_vector=table_vector+emb_dic[table_name][_col]
        return table_vector
    
    def average_vectors(self, emb_dic, table_name, columns):
        table_vector=np.zeros(768) 
        col_number= len(columns)
        table_vector=self.sum_vectors(emb_dic, table_name, columns)
        table_vector=np.divide(table_vector,float(col_number) )
        return    table_vector
            
    
    def item_frequency(self, lst):
        return dict(Counter(lst))
    
    def min_div_score(self, s_dict: dict, metric = "cosine", normalize = False) -> float:
        if len(s_dict) == 0:
            return [0]
        min_scores = [] # all possible distances
        for current_s1, current_s1_embedding in s_dict.items():
            for current_s2, current_s2_embedding in s_dict.items():
                if current_s1 != current_s2:
                    if metric == "l1":
                        if normalize == True:
                            max_possible_l1 = 2 * len(current_s1_embedding)
                        else:
                            max_possible_l1 = 1
                        current_sim = np.linalg.norm(current_s1_embedding - current_s2_embedding, ord = 1) / max_possible_l1  # utl.CosineSimilarity(current_s1_embedding, current_s2_embedding)
                    elif metric == "l2":
                        if normalize == True:
                            max_possible_l2 = np.sqrt(2 * len(current_s1_embedding))
                        else:
                            max_possible_l2 = 1
                        current_sim = np.linalg.norm(current_s1_embedding - current_s2_embedding, ord = 2) / max_possible_l2 # utl.CosineSimilarity(current_s1_embedding, current_s2_embedding)
                    else: # metric = cosine
                        if normalize == True:
                            current_sim = 1 - ((utl.CosineSimilarity(current_s1_embedding, current_s2_embedding) + 1 ) / 2)
                        else:
                            current_sim = 1 - utl.CosineSimilarity(current_s1_embedding, current_s2_embedding)
                    min_scores.append(current_sim)
        return min_scores

    def min_mix_div_score(self, s_dict: dict, q_set:set, metric = "cosine", normalize = False) -> float:
        if len(s_dict) == 0:
            return [0]
        min_scores = [] # all possible distances
        for current_s1, current_s1_embedding in s_dict.items():
            for current_s2, current_s2_embedding in s_dict.items():
                if current_s1 in q_set and current_s2 in q_set:
                    continue
                if current_s1 != current_s2:
                    if metric == "l1":
                        if normalize == True:
                            max_possible_l1 = 2 * len(current_s1_embedding)
                        else:
                            max_possible_l1 = 1
                        current_sim = np.linalg.norm(current_s1_embedding - current_s2_embedding, ord = 1) / max_possible_l1  # utl.CosineSimilarity(current_s1_embedding, current_s2_embedding)
                    elif metric == "l2":
                        if normalize == True:
                            max_possible_l2 = np.sqrt(2 * len(current_s1_embedding))
                        else:
                            max_possible_l2 = 1
                        current_sim = np.linalg.norm(current_s1_embedding - current_s2_embedding, ord = 2) / max_possible_l2 # utl.CosineSimilarity(current_s1_embedding, current_s2_embedding)
                    else: # metric = cosine
                        if normalize == True:
                            current_sim = 1 - ((utl.CosineSimilarity(current_s1_embedding, current_s2_embedding) + 1 ) / 2)
                        else:
                            current_sim = 1 - utl.CosineSimilarity(current_s1_embedding, current_s2_embedding)
                    min_scores.append(current_sim)
        return min_scores

    def f_prime(self, s_set: set, lmda : float, k: int, sim_dict: dict, div_dict: dict) -> float:
        if len(s_set) == 0:
            return 0
        total_sim_score = 0
        total_div_score = 0
        for s_i in s_set:
            total_sim_score += sim_dict[s_i]
        for s_i in s_set:
            for s_j in s_set:
                if (s_i, s_j) in div_dict:
                    total_div_score += div_dict[(s_i, s_j)]
                else:
                    total_div_score += div_dict[(s_j, s_i)]
        total_sim_score *= (k-1) * (1 - lmda)
        total_div_score *= 2 * lmda
        return (total_sim_score + total_div_score)

    
    def avg_div_sim(self, q_name, s_prime ):
        # compute average of similarity component 
        sigma=0
        for s_i in s_prime: 
            similarity_score_value=self.calculate_unionability(q_name,s_i)
            sigma=sigma+similarity_score_value
            
        sim_comp=  sigma/float(len(s_prime)) 
            
        # compute average of div component
            
        sigma=0
        
        for i in range(0, len(s_prime)-1):
            for j in range (i+1, len(s_prime)):
                s_i=s_prime[i]
                s_j=s_prime[j] 
                diversity_score=self.calculate_diversity( q_name,s_i,s_j)
                sigma=sigma+diversity_score
        
        div_comp=sigma/float(len(s_prime)*(len(s_prime)-1))
        
        res= sim_comp+div_comp
        
        return res/2.0
        
     
    def max_div_sim(self, q_name, s_prime):
        
        max_value_sim=0
        
        for s_i in s_prime: 
             similarity_score_value=self.calculate_unionability(q_name,s_i)
             if similarity_score_value>max_value_sim: 
                 max_value_sim=similarity_score_value
       
        max_value_div=0
        for i in range(0, len(s_prime)-1):
            for j in range (i+1, len(s_prime)):
                s_i=s_prime[i]
                s_j=s_prime[j] 
                diversity_score=self.calculate_diversity( q_name,s_i,s_j)
                if diversity_score>max_value_div:
                    max_value_div=diversity_score
        overall_max = max(max_value_sim, max_value_div)
        return overall_max

    def min_div_sim(self, q_name, s_prime):  
        ''' the minimum score amon all the sim and div scores for a q in S_prime'''
        min_value_sim=0
        
        for s_i in s_prime: 
            similarity_score_value=self.calculate_unionability(q_name,s_i)
            if similarity_score_value<min_value_sim: 
                 min_value_sim=similarity_score_value
       
        min_value_div=0
        for i in range(0, len(s_prime)-1):
            for j in range (i+1, len(s_prime)):
                s_i=s_prime[i]
                s_j=s_prime[j] 
                diversity_score=self.calculate_diversity( q_name,s_i,s_j)
                if diversity_score<min_value_div:
                    min_value_div=diversity_score
     
        overall_min = min(min_value_sim, min_value_div)
        # retrun minumum 
        return overall_min     
    
    
    def F(self, q_name, s_prime, lambda_, k):  
        '''compute F in GMC '''
        # compute average of similarity component 
        sigma_sim=0
        for s_i in s_prime: 
            similarity_score_value=self.calculate_unionability(q_name,s_i)
            sigma_sim=sigma_sim+similarity_score_value
            
     
            
        # compute average of div component
            
        sigma_div=0
        
        for i in range(0, len(s_prime)-1):
            for j in range (i+1, len(s_prime)):
                s_i=s_prime[i]
                s_j=s_prime[j] 
                diversity_score=self.calculate_diversity( q_name,s_i,s_j)
                sigma_div=sigma_div+diversity_score

        # retrun minumum 
        F_q=(k-1)*(1-lambda_)*sigma_sim+ 2.0*sigma_div
        return F_q    

    

    
    def compute_F(self, result_file, alg_, csv_file_path, k):
        
         # Define column names
        #open file to 
            lmda = 0.7
            columns = ["query_table","F","k", alg_+"_exec_time_secs"]

# Create an empty DataFrame with the specified columns
            metrics_df = pd.DataFrame(columns=columns)
        
            file_path = result_file  # Replace with the path to your CSV file
            df = pd.read_csv(file_path, usecols=[0, 1, 2, 3])

            # Extract column names for reference
            column_names = df.columns.tolist()

            # Assuming 'k', 'query_name', and 'tables' are part of the loaded columns
            k_column = column_names[3]  # Replace with the actual column name for k
            query_name_column = column_names[0]  # Replace with the actual column name for query name
            tables_column = column_names[1]  # Replace with the actual column name for tables
            time_sec = column_names[2]
            df = df[df[k_column] == k]
            # go through result and for every query compute two functions: 
            # 1. Split into lists and strip spaces:
            df = df[df[tables_column].notna()].copy()
            df[tables_column] = (
                df[tables_column]
                .str.split(',')
                .apply(lambda files: [f.strip() for f in files])
            )
            
            
        

            # 2. Iterate through each query:
            for _, row in df.iterrows():
                q = row[query_name_column]
                s_prime = row[tables_column]

                 
                print("Genarating Evaluation for query: "+q)
     
                F_value=self.F(q,s_prime,lmda,k)
                
                # Define the row to add (values must match the column order)
                new_row = {
        "query_table": q,
        "F":F_value ,
        "k":k,
        alg_+"_exec_time_secs":time_sec
    }

    # Add the row to the DataFrame
                metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True) 

    # Write the DataFrame to a CSV file
            
            if os.path.exists(csv_file_path):
            # File exists, append without writing headers
                metrics_df.to_csv(csv_file_path, mode='a', header=False, index=False)
                print(f"Appended data to existing file: {csv_file_path}")
            else:
            # File does not exist, create new file with headers
                metrics_df.to_csv(csv_file_path, mode='w', header=True, index=False)
                print(f"Created new file and wrote data: {csv_file_path}")
            
            return csv_file_path
        
    
    
    
    def compute_metrics(self, result_file, alg_, csv_file_path, k):
            '''alg_: {gmc, penal}'''
        # Define column names
        #open file to 
            lmda = 0.7
            columns = ["query_table", "avg_div_sim", "max_div_sim", "min_div_score","k", alg_+"_exec_time_secs"]

# Create an empty DataFrame with the specified columns
            metrics_df = pd.DataFrame(columns=columns)
        
            file_path = result_file  # Replace with the path to your CSV file
            df = pd.read_csv(file_path, usecols=[0, 1, 2, 3])

            # Extract column names for reference
            column_names = df.columns.tolist()

            # Assuming 'k', 'query_name', and 'tables' are part of the loaded columns
            k_column = column_names[3]  # Replace with the actual column name for k
            query_name_column = column_names[0]  # Replace with the actual column name for query name
            tables_column = column_names[1]  # Replace with the actual column name for tables
            time_sec = column_names[2]
            df = df[df[k_column] == k]
            # go through result and for every query compute two functions: 
            # 1. Split into lists and strip spaces:
            df = df[df[tables_column].notna()].copy()
            df[tables_column] = (
                df[tables_column]
                .str.split(',')
                .apply(lambda files: [f.strip() for f in files])
            )
            
            
            # df[tables_column] = (
            #     df[tables_column]
            #     .str.split(',') 
            #     .apply(lambda files: [f.strip() for f in files])
            # )

            # 2. Iterate through each query:
            for _, row in df.iterrows():
                q = row[query_name_column]
                s_prime = row[tables_column]

                 
                print("Genarating Evaluation for query: "+q)
                avg_div_sim=self.avg_div_sim(q,s_prime)
                max_div_sim=self.max_div_sim(q,s_prime)
                min_div_score=self.min_div_sim(q,s_prime)
             
                
                # Define the row to add (values must match the column order)
                new_row = {
        "query_table": q,
        "avg_div_sim": avg_div_sim,
        "max_div_sim": max_div_sim, 
        "min_div_score": min_div_score,
        "k":k,
        alg_+"_exec_time_secs":time_sec
    }

    # Add the row to the DataFrame
                metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True) 

    # Write the DataFrame to a CSV file
            
            if os.path.exists(csv_file_path):
            # File exists, append without writing headers
                metrics_df.to_csv(csv_file_path, mode='a', header=False, index=False)
                print(f"Appended data to existing file: {csv_file_path}")
            else:
            # File does not exist, create new file with headers
                metrics_df.to_csv(csv_file_path, mode='w', header=True, index=False)
                print(f"Created new file and wrote data: {csv_file_path}")
            
            return csv_file_path
  
 
    @staticmethod

    def analyse_gmc_metrics(file1_path: str, baseline_path: str, k_value: int):
            #  you can call this function for all three following metrics: 
            
            #   avg_div_sim
            #   max_div_sim
            #   min_div_score
            
            #the input files have these columns: 
            # query_table,avg_div_sim,max_div_sim,min_div_score,k,Penalized_exec_time_secs
            #the output will have this columns: 
            # k,avg_div_sim,max_div_sim,min_div_score which
            # shows number of queries that file 1 was better than or equal to file 2 for each of the metrics
        
            # 1. Load
            df1 = pd.read_csv(file1_path)
            df2 = pd.read_csv(baseline_path)
            
            

            # Find common queries
            common_queries = set(df1['query_table']).intersection(set(df2['query_table']))
            # if len(common_queries) != len (df1['query_table']): 
            #     raise RuntimeError("the number of queries are different in input files")


            df1 = df1[df1['query_table'].isin(common_queries)].copy()
            df2 = df2[df2['query_table'].isin(common_queries)].copy()
            # 2. Filter by k
            df1_k = df1[df1['k'] == k_value]
            df2_k = df2[df2['k'] == k_value]

            # 3. Merge on query_table
            merged = pd.merge(
                df1_k,
                df2_k,
                on='query_table',
                suffixes=('_1', '_2'),
                how='inner'  # only keep query_tables present in both
            )

            # 4. Count how many times file1 >= baseline for each metric
            results = {
                'k': k_value,
                'avg_div_sim':   (merged['avg_div_sim_1']   >= merged['avg_div_sim_2']).sum(),
                'max_div_sim':   (merged['max_div_sim_1']   >= merged['max_div_sim_2']).sum(),
                'min_div_score': (merged['min_div_score_1'] >= merged['min_div_score_2']).sum(),
            }

            # 5. Write results
            return pd.DataFrame([results])


    def analyse_gmc_F(file1_path: str, baseline_path: str, k_value: int):
          
            
            #the input files have these columns: 
            # query_table,avg_div_sim,max_div_sim,min_div_score,k,Penalized_exec_time_secs
            #the output will have this columns: 
            # k,F which
            # shows number of queries that file 1 was better than or equal to file 2 for each of the metrics
        
            # 1. Load
            df1 = pd.read_csv(file1_path)
            df2 = pd.read_csv(baseline_path)
            
            

            # Find common queries
            common_queries = set(df1['query_table']).intersection(set(df2['query_table']))
            if len(common_queries) != len (df1['query_table']): 
                       print("the number of queries are different in input files for F calculation")


            df1 = df1[df1['query_table'].isin(common_queries)].copy()
            df2 = df2[df2['query_table'].isin(common_queries)].copy()
            # 2. Filter by k
            df1_k = df1[df1['k'] == k_value]
            df2_k = df2[df2['k'] == k_value]

            # 3. Merge on query_table
            merged = pd.merge(
                df1_k,
                df2_k,
                on='query_table',
                suffixes=('_1', '_2'),
                how='inner'  # only keep query_tables present in both
            )

            # 4. Count how many times file1 >= baseline for each metric
            results = {
                'k': k_value,
                'F':   (merged['F_1']   >= merged['F_2']).sum()
            }

            # 5. Write results
            return pd.DataFrame([results])   
 
           
    

    def gmc(self, S_names: set, query_name:str, k: int, 
            lmda: float = 0.7, metric = "cosine", print_results = False,
            normalize = False, max_metric = True, compute_metric = True) -> set: #S_dict is a dictionary with tuple id as key and its embeddings as value. 

        '''adopted from https://anonymous.4open.science/r/dust-B79B/diversity_algorithms/div_utilities.py'''

        start_time = time.time_ns()
        R = set()
        ranked_div_result = []

        S_set = S_names
        for p in range(0, k):
            if len(S_set) == 0:
                break
            mmc_dict = self.mmc(query_name,S_set, lmda, k, R) # send S to mmc and compute MMC for each si
            s_i  = max(mmc_dict, key=lambda k: mmc_dict[k])
            R.add(s_i)
            ranked_div_result.append(s_i)
            S_set = set(S_set) - {s_i}
        end_time = time.time_ns()
        total_time = round((end_time - start_time) / 10 ** 9, 2)
        print("Total time taken: ", total_time, " seconds.")
        
        return ranked_div_result,total_time
        
    def _cosine_sim(self, vec1, vec2):
        ''' Get the cosine similarity of two input vectors: vec1 and vec2
        '''
        assert vec1.ndim == vec2.ndim
        return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))
    
    
    
    def load_starmie_vectors(self, dl_table_vectors_path,query_table_vectors ):
        '''load starmie vectors for query and data lake and retrun as dictionaries'''
  
        dl_table_vectors =dl_table_vectors_path
        query_table_vectors = query_table_vectors
        qfile = open(query_table_vectors,"rb")
         # queries is a list of tuples ; tuple of (str(filename), numpy.ndarray(vectors(numpy.ndarray) for columns) 
        queries = pickle.load(qfile)
        # make as dictnary from first item to secon item 
        # Convert to dictionary
        queries_dict = {item[0]: item[1] for item in queries}

        tfile = open(dl_table_vectors,"rb")
        tables = pickle.load(tfile)
        # tables is a list of tuples ; tuple of (str(filename), numpy.ndarray(vectors(each verstor is a numpy.ndarray) for columns) 
    
        dl_dict = {item[0]: item[1] for item in tables}
        
        return (queries_dict,dl_dict)
    
   
   
    def calculate_unionability(self, query_name, dl_table_name):
        ''' aligned columns are loaded in self.q_dl_alignemnt
            get the vector represenation from starmie 
            calulate the cosine similarity 
            sum col sims to get table similarity and this is unionability  or similaryty score and then retun the unionbility for the pair
        '''
        all_vectors=self.starmie_vectors
        #all_vectors=self.load_starmie_vectors()
        queries_dict = all_vectors[0]
        dl_dict=all_vectors[1]
        q_vectors=queries_dict[query_name]
        query_rows = self.q_dl_alignemnt[self.q_dl_alignemnt['query_table_name'] == query_name]
        similarity_score=0.0

        dl_t_vectors=dl_dict[dl_table_name]
        # Retrieve the relevant columns
        specific_rows = query_rows[query_rows['dl_table_name'] == dl_table_name]

        num_rows = float(specific_rows.shape[0])
        if num_rows==0:
            return 0.0 # no alignemnt exists so we return  0.0

        for _, row in specific_rows.iterrows():
                         # get their vectors 
                        query_column = row['query_column#']
                        dl_column = row['dl_column']
                        # Call the similarity function
                        similarity_col = self._cosine_sim(q_vectors[query_column],dl_t_vectors[dl_column])
                        similarity_score=similarity_col+similarity_score

         
       
        return   similarity_score         
   
   
    def item_frequency(self, lst):
        return dict(Counter(lst))
    
   
    def   Jensen_Shannon_distances(self,query_column,dl_column,domain_estimate):
        # build the x axis for both columns converting the  domain_estimate to a set of tuples
                        # each tuple <item label, item index in x axis>
            x_axis={}
            i=0
            for item in domain_estimate:
                x_axis[item]=i
                i=i+1
            
            #now build the probability array   
            frequency_q= self.item_frequency(query_column)
            frequency_dl= self.item_frequency(dl_column)
            
            list_length_q=len(query_column)
            list_length_dl=len(dl_column)
            
            
            #probability arrays
            array_q  = np.zeros(len(domain_estimate)) 
            array_dl = np.zeros(len(domain_estimate)) 
            
            for item in domain_estimate:
                index_= x_axis[item]
                if(item in frequency_q):
                    freq_q_item= frequency_q[item]
                    array_q[index_]=freq_q_item/float(list_length_q)
                else: 
                      array_q[index_]=0
                if (item in frequency_dl):   
                    freq_dl_item= frequency_dl[item]
                    array_dl[index_]=freq_dl_item/float(list_length_dl)
                else: 
                    array_dl[index_]=0    
                
            #The Jensen-Shannon distances
            dis_qdl=distance.jensenshannon(array_q,array_dl) 
            dis_dlq=distance.jensenshannon(array_dl,array_q)     
            if(dis_qdl!=dis_dlq):
                    raise ValueError('the distance metric is asymmetric')  
            else:  
                    return dis_dlq  
   
    def _lexicalsim_Pair(self, query_column, table_column):
    # The input sets must be a Python list of iterables (i.e., lists or sets).
        sets = [query_column, set(table_column)]
        #sets = [[1,2,3], [3,4,5], [2,3,4], [5,6,7]]

        # all_pairs returns an iterable of tuples.
        pairs = all_pairs(sets, similarity_func_name="jaccard", similarity_threshold=0.0)
        l_pairs=list(pairs)
        # [(1, 0, 0.2), (2, 0, 0.5), (2, 1, 0.5), (3, 1, 0.2)]
        # Each tuple is (<index of the first set>, <index of the second set>, <similarity>).
        # The indexes are the list indexes of the input sets.
        if len(l_pairs)==0:
            return 0
        else:
            return l_pairs[0][2]
    
    def calculate_diversity(self, query_name,s_i, s_j):
        print("computing diversity scores for given s_i and s_j wrt query..")
        """
           diversity between two tables are the sum of diversity between their columns
         """
        DSize=self.Dsize

        div_score=0.0
        # alignment between query  and s_i
        query_rows_si = self.q_dl_alignemnt[(self.q_dl_alignemnt['query_table_name'] == query_name) & (self.q_dl_alignemnt['dl_table_name'] == s_i)] 
        
        aligned_col_q_si = set(query_rows_si['query_column#'])

        # alignment between query  and s_j
        query_rows_sj = self.q_dl_alignemnt[(self.q_dl_alignemnt['query_table_name'] == query_name) & (self.q_dl_alignemnt['dl_table_name'] == s_j)] 
        aligned_col_q_sj = set(query_rows_sj['query_column#'])

  # for columns from i , j that are alignemed to identical query colmn we calculate the dicersity 
  # for those they are aligned with different columns of q  based on the null assumption we assign 1 for the score 

        alignemd_same_qcolumns=aligned_col_q_si.intersection(aligned_col_q_sj)
        # aligned_col_q_si_solo=aligned_col_q_si.difference(alignemd_same_qcolumns)
        # aligned_col_q_sj_solo=aligned_col_q_sj.difference(alignemd_same_qcolumns)
        
           
        # if there is no common aligned columns I am retrun zero for div 
        if(len(alignemd_same_qcolumns)==0):
            return 0.0
        # compute diversity on aligned to same column of q 
        else:  
                for query_column_number in alignemd_same_qcolumns:
                    
                    s_i_column_number = self.q_dl_alignemnt[(self.q_dl_alignemnt['query_table_name'] == query_name) & (self.q_dl_alignemnt['dl_table_name'] == s_i) & (self.q_dl_alignemnt['query_column#'] == query_column_number)] ['dl_column'].item()
                    s_j_column_number = self.q_dl_alignemnt[(self.q_dl_alignemnt['query_table_name'] == query_name) & (self.q_dl_alignemnt['dl_table_name'] == s_j) & (self.q_dl_alignemnt['query_column#'] == query_column_number)] ['dl_column'].item()

                    
                    #get the comlumn from data lake table
                    dl_column_s_i=self.table_raw_lol_proccessed.get(s_i)[s_i_column_number]
                    dl_column_set_s_i=self.table_raw_proccessed_los.get(s_i)[s_i_column_number]
                    
                    #get the comlumn from data lake table
                    dl_column_s_j=self.table_raw_lol_proccessed.get(s_j)[s_j_column_number]
                    dl_column_set_s_j=self.table_raw_proccessed_los.get(s_j)[s_j_column_number]

                
                    #see what is the number of unique values in the query+ dl columns that are list of list 
                    # we have a threshold to determine the smallness of domain called DS(domain size)
                    # Besat to change: here we do not merge tokens from all cell to 
                    # gether for each column maybe this will change later ?
                    domain_estimate=set.union(set(dl_column_s_j),set(dl_column_s_i) )
                    if(len(domain_estimate)<DSize):
                        distance=self.Jensen_Shannon_distances(dl_column_s_i,dl_column_s_j,domain_estimate)
                        # log the domian infomrmation
                    else: 
                        # jaccard distance
                        distance=1-self._lexicalsim_Pair(dl_column_set_s_i,dl_column_set_s_j)
                    div_score=div_score+distance
                    distance=0.0        
                    # # add one for each null columns 
                    # div_score =div_score+len(aligned_col_q_si_solo) 
                    #             # add one for each null columns 
                    # div_score =div_score+len(aligned_col_q_sj_solo) 
                    # num_rows=len(aligned_col_q_sj_solo) +len(aligned_col_q_si_solo) +len(alignemd_same_qcolumns)
        
           
            
        return div_score
 
    
      
    
    def d_sim(self, s_dict: dict, q_embedding: np.ndarray, metric="cosine", normalize=False) -> dict:
        sim_dict = dict()  # key: s_dict key i.e. s_id; value : similarity score
        for current_s, current_s_embedding in s_dict.items():
            if metric == "l1":
                if normalize == True:
                    max_possible_l1 = 2 * len(current_s_embedding)
                else:
                    max_possible_l1 = 1
                current_sim = np.linalg.norm(current_s_embedding - q_embedding, ord=1) / max_possible_l1
            elif metric == "l2":
                if normalize == True:
                    max_possible_l2 = np.sqrt(2 * len(current_s_embedding))
                else:
                    max_possible_l2 = 1
                current_sim = np.linalg.norm(current_s_embedding - q_embedding, ord=2) / max_possible_l2
            else:  # cosine
                if normalize == True:
                    current_sim = 1 - ((utl.CosineSimilarity(current_s_embedding, q_embedding) + 1) / 2)
                else:
                    current_sim = 1 - utl.CosineSimilarity(current_s_embedding, q_embedding)
            sim_dict[current_s] = current_sim
        return sim_dict

    def d_div(self, s_dict: dict, metric="cosine", normalize=False) -> dict:
        div_dict = dict()  # key: s_dict key i.e. s_id; value : similarity score
        for current_s1 in s_dict:
            for current_s2 in s_dict:
                if metric == "l1":
                    max_possible_l1 = 2 * len(s_dict[current_s1])
                    if normalize == True:
                        current_div = 1 - (np.linalg.norm(s_dict[current_s1] - s_dict[current_s2], ord=1) / max_possible_l1)
                    else:
                        current_div = max_possible_l1 - (np.linalg.norm(s_dict[current_s1] - s_dict[current_s2], ord=1) / 1)
                elif metric == "l2":
                    max_possible_l2 = np.sqrt(2 * len(s_dict[current_s1]))
                    if normalize == True:
                        current_div = 1 - (np.linalg.norm(s_dict[current_s1] - s_dict[current_s2], ord=2) / max_possible_l2)
                    else:
                        current_div = max_possible_l2 - (np.linalg.norm(s_dict[current_s1] - s_dict[current_s2], ord=2) / 1)
                else:  # cosine
                    if normalize == True:
                        current_div = (utl.CosineSimilarity(s_dict[current_s1], s_dict[current_s2]) + 1) / 2  # normalized score between 0 and 1
                    else:
                        current_div = utl.CosineSimilarity(s_dict[current_s1], s_dict[current_s2])
                div_dict[(current_s1, current_s2)] = current_div
        return div_dict


      
if __name__ == "__main__":
    import argparse
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    def validate_paths(args):
        """Validate that all required paths exist."""
        paths_to_check = {
            'benchmark_path': args.benchmark_path,
            'alignment_file': args.alignment_file
        }
        
        if args.result_file:
            paths_to_check['result_file'] = args.result_file
            
        for name, path in paths_to_check.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} path does not exist: {path}")
                
        # Create output directory if it doesn't exist
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

    def validate_k_range(k_range):
        """Validate the k range format and values."""
        try:
            start, end = map(int, k_range.split('-'))
            if start < 1 or end < start:
                raise ValueError("Invalid k range: start must be >= 1 and end must be >= start")
            return start, end
        except ValueError as e:
            raise ValueError(f"Invalid k range format. Expected 'start-end', got '{k_range}'. Error: {str(e)}")

    def parse_args():
        """Parse command line arguments with improved validation and help messages."""
        parser = argparse.ArgumentParser(
            description='GMC Search: Greedy Marginal Contribution for Novelty-based Unionable Table Search',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Required arguments
        parser.add_argument('--mode', type=str, required=True,
                          choices=['search', 'evaluate', 'F_value', 'integrateResult', 'integrate_F'],
                          help='Mode of operation:\n'
                               '- search: Perform GMC search\n'
                               '- evaluate: Generate evaluation metrics\n'
                               '- F_value: Calculate F-values\n'
                               '- integrateResult: Integrate search results\n'
                               '- integrate_F: Integrate F-values')
        
        parser.add_argument('--benchmark_path', type=str, required=True,
                          help='Path to benchmark data directory containing vectors and processed data')
        
        parser.add_argument('--alignment_file', type=str, required=True,
                          help='Path to query-datalake alignment file')
        
        # Optional arguments
        parser.add_argument('--system', type=str, default='GMC',
                          help='System name for evaluation (default: GMC)')
        
        parser.add_argument('--result_file', type=str,
                          help='Path to result file for evaluation modes')
        
        parser.add_argument('--output_dir', type=str,
                          help='Output directory for results (will be created if it\'s not exist)')
        
        parser.add_argument('--k_range', type=str, default='2-11',
                          help='Range of k values to evaluate in format "start-end" (default: 2-11)')
        
        parser.add_argument('--lambda', type=float, default=0.7,
                          help='Lambda parameter for GMC algorithm (default: 0.7)')
        
        parser.add_argument('--verbose', action='store_true',
                          help='Enable verbose logging')
        
        args = parser.parse_args()
        
        # Validate arguments
        validate_k_range(args.k_range)
        validate_paths(args)
        
        # Set logging level
        if args.verbose:
            logger.setLevel(logging.DEBUG)
            
        return args

    def initialize_gmc_search(args):
        """Initialize GMC search with common parameters."""
        try:
            search_params = {
                "keyword": "example",
                "max_results": 10,
                "lambda": args.lambda_
            }
            
            dl_table_vectors_path = os.path.join(args.benchmark_path, "vectors", 
                                               "cl_datalake_drop_col_tfidf_entity_column_0.pkl")
            query_table_vectors_path = os.path.join(args.benchmark_path, "vectors",
                                                  "cl_query_drop_col_tfidf_entity_column_0.pkl")
            
            logger.info("Initializing GMC search with parameters:")
            logger.info(f"- Domain size: 20")
            logger.info(f"- Lambda: {args.lambda_}")
            logger.info(f"- Vector paths: {dl_table_vectors_path}, {query_table_vectors_path}")
            
            gmc_search = GMC_Search(
                args.alignment_file,
                20,  # domain_size
                dl_table_vectors_path,
                query_table_vectors_path,
                args.benchmark_path,
                search_params
            )
            
            # Load unionable tables
            top_k_starmie = os.path.join(args.benchmark_path, "diveristy_data", "search_results", "Starmie",
                                        "top_20_Starmie_output_04diluted_restricted_noscore.pkl")
            gmc_search.load_unionable_tables(top_k_starmie)
            
            return gmc_search
            
        except Exception as e:
            logger.error(f"Failed to initialize GMC search: {str(e)}")
            raise

    def run_search_mode(args):
        """Execute search mode."""
        try:
            gmc_search = initialize_gmc_search(args)
            output_csv_file = os.path.join(args.benchmark_path, "diveristy_data", "search_results", "GMC",
                                         "gmc_results_diluted04_restricted.csv")
            
            start_k, end_k = map(int, args.k_range.split('-'))
            logger.info(f"Running search mode for k={start_k} to {end_k}")
            
            for k in range(start_k, end_k):
                logger.info(f"Processing k={k}")
                gmc_search.k = k
                results = gmc_search.execute_search()
                
                # Write results to CSV
                mode = 'a' if os.path.exists(output_csv_file) else 'w'
                with open(output_csv_file, mode, newline='') as file:
                    writer = csv.writer(file)
                    if mode == 'w':
                        writer.writerow(['query_name', 'tables', 'gmc_execution_time', 'k'])
                    
                    for query_name, (result, secs) in results.items():
                        result_str = ', '.join(result) if isinstance(result, list) else str(result)
                        writer.writerow([query_name, result_str, secs, gmc_search.k])
                        
            logger.info(f"Search completed. Results written to {output_csv_file}")
            
        except Exception as e:
            logger.error(f"Error in search mode: {str(e)}")
            raise

    def run_evaluation_mode(args):
        """Execute evaluation mode."""
        try:
            gmc_search = initialize_gmc_search(args)
            start_k, end_k = map(int, args.k_range.split('-'))
            logger.info(f"Running evaluation mode for k={start_k} to {end_k}")
            
            for k in range(start_k, end_k):
                file_ = os.path.join(args.benchmark_path, "diveristy_data", "search_results", args.system,
                                   f"evaluation_metrics_gmc_{k}.csv")
                if not os.path.exists(file_):
                    logger.info(f"Computing metrics for k={k}")
                    gmc_search.k = k
                    gmc_search.compute_metrics(
                        args.result_file,
                        args.system,
                        file_,
                        k
                    )
                else:
                    logger.info(f"Metrics file already exists for k={k}, skipping")
                    
        except Exception as e:
            logger.error(f"Error in evaluation mode: {str(e)}")
            raise

    def run_F_value_mode(args):
        """Execute F-value mode."""
        try:
            gmc_search = initialize_gmc_search(args)
            start_k, end_k = map(int, args.k_range.split('-'))
            logger.info(f"Running F-value mode for k={start_k} to {end_k}")
            
            for k in range(start_k, end_k):
                file_ = os.path.join(args.benchmark_path, "diveristy_data", "search_results", args.system,
                                   f"F_gmc_{k}.csv")
                if not os.path.exists(file_):
                    logger.info(f"Computing F-values for k={k}")
                    gmc_search.k = k
                    gmc_search.compute_F(
                        args.result_file,
                        args.system,
                        file_,
                        k
                    )
                else:
                    logger.info(f"F-value file already exists for k={k}, skipping")
                    
        except Exception as e:
            logger.error(f"Error in F-value mode: {str(e)}")
            raise

    def run_integration_mode(args, mode_type):
        """Execute integration mode for either results or F-values."""
        try:
            out_path = os.path.join(args.benchmark_path, "diveristy_data", "GMC_params")
            os.makedirs(out_path, exist_ok=True)
            
            systems = ['GMC', 'Starmie', 'semanticNovelty', 'starmie0', 'starmie1']
            prefix = 'F_' if mode_type == 'F' else ''
            
            logger.info(f"Running integration mode for {mode_type}")
            
            for system in systems:
                filename = f"{prefix}pen_to_{system.lower()}.csv"
                output_file = os.path.join(out_path, filename)
                
                if os.path.exists(output_file):
                    logger.info(f"Output file '{filename}' already exists. Skipping.")
                    continue
                
                logger.info(f"Processing system: {system}")
                all_results = []
                start_k, end_k = map(int, args.k_range.split('-'))
                
                for k in range(start_k, end_k):
                    if mode_type == 'F':
                        df_k = GMC_Search.analyse_gmc_F(
                            os.path.join(args.benchmark_path, "diveristy_data", "search_results", "Penalized",
                                       f"F_gmc_{k}.csv"),
                            os.path.join(args.benchmark_path, "diveristy_data", "search_results", system,
                                       f"F_gmc_{k}.csv"),
                            k
                        )
                    else:
                        df_k = GMC_Search.analyse_gmc_metrics(
                            os.path.join(args.benchmark_path, "diveristy_data", "search_results", "Penalized",
                                       f"evaluation_metrics_gmc_{k}.csv"),
                            os.path.join(args.benchmark_path, "diveristy_data", "search_results", system,
                                       f"evaluation_metrics_gmc_{k}.csv"),
                            k
                        )
                    all_results.append(df_k)
                
                final_df = pd.concat(all_results, ignore_index=True)
                final_df.to_csv(output_file, index=False)
                logger.info(f"Integration results written to {output_file}")
                
        except Exception as e:
            logger.error(f"Error in integration mode: {str(e)}")
            raise

    try:
        # Parse and validate arguments
        args = parse_args()
        
        # Run the appropriate mode
        if args.mode == 'search':
            run_search_mode(args)
        elif args.mode == 'evaluate':
            run_evaluation_mode(args)
        elif args.mode == 'F_value':
            run_F_value_mode(args)
        elif args.mode == 'integrateResult':
            run_integration_mode(args, 'result')
        elif args.mode == 'integrate_F':
            run_integration_mode(args, 'F')
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

