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

from multiprocessing import Pool, cpu_count
import multiprocessing as mp

# Set the multiprocessing start method to spawn to avoid CUDA reinitialization issues.
mp.set_start_method('spawn', force=True)


from SetSimilaritySearch import all_pairs
class GMC_Search:
    """
    GMC_Search: A class for performing Greedy  Marginal  Contribution (GMC) for Novelty based  unionable  table search.
    """

    def __init__(self, q_dl_alignemnt_file, 
                 domain_size, dl_table_vectors_path,query_table_vectors_path,
                 benchmark_folder, 
                 
                 search_parameters=None):
        """
        Initializes the GMC_Search instance.

        :param data_source: The source of the data (e.g., file path, database, etc.).
        :param search_parameters: Dictionary containing parameters for the search (optional).
        """
        self.k=0
        self.q_dl_alignemnt_file = q_dl_alignemnt_file
        self.search_parameters = search_parameters or {}
        self.results = []
        self.q_dl_alignemnt=utl.load_alignment(self.q_dl_alignemnt_file)
        self.unionability = None
        self.diversity = None
        self.diversity_scores=None
        self.unionability_scores=None
        self.unionable_data=None
        self.query_dl_table_vector_df=None # vector represenation of tables given (query) 
        self.query_s_i_s_j_vector_df=None # vector representation of tables given (query and si)
        self.Dsize=domain_size

        self.starmie_vectors=self.load_starmie_vectors(dl_table_vectors_path,query_table_vectors_path)
        ################################
        # dataFolder="santos"
        # table_path = "data/"+dataFolder+"/vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl"
        # table_path_raw = "data/"+dataFolder+"/datalake"
        # processed_path="data/"+dataFolder+"/proccessed/"
        # index_file_path="data/"+dataFolder+"/indices/Joise_Index_DL_santos_tokenized_bot.pkl"
        
        table_path = benchmark_folder+"/vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl"
        table_path_raw = benchmark_folder+"/datalake"
        processed_path=benchmark_folder+"/proccessed/"
        index_file_path=benchmark_folder+"/indices/Joise_Index_DL_tus_tokenized_bot.pkl"
        self.load_all_data_elements(table_path,table_path_raw, processed_path, index_file_path)
 
    def load_all_data_elements(self, table_path, table_path_raw, processed_path,index_file_path):

        
        text_processor = TextProcessor()

        self.tables_raw=NaiveSearcherNovelty.read_csv_files_to_dict(table_path_raw)
        
        #the normalized/tokenized/original with no duplicates  dl(data lake) tables are stored in table_raw_proccessed_los
        table_raw_proccessed_los={}
            # write the proccessed result having columns as set to a pickle file 
        dl_tbls_processed_set_file_name="dl_tbls_processed_set.pkl"
        
        table_raw_proccessed_los=test_naive_search_Novelty.getProcessedTables(text_processor, dl_tbls_processed_set_file_name, processed_path, self.tables_raw,"los", 1, 1)
        # process dl tables and save as list of lists 
        self.table_raw_proccessed_los=table_raw_proccessed_los
        table_raw_lol_proccessed={}
            # write the proccessed result having columns as set to a pickle file 
        dl_tbls_processed_lol_file_name="dl_tbls_processed_lol.pkl"
        self.dl_tbls_processed_lol_file_name=dl_tbls_processed_lol_file_name
        table_raw_lol_proccessed=test_naive_search_Novelty.getProcessedTables(text_processor,  dl_tbls_processed_lol_file_name,processed_path,self. tables_raw,"lol", 1, 1)
        self.table_raw_lol_proccessed=table_raw_lol_proccessed

        table_raw_index={}

       
        index_exists = os.path.isfile(index_file_path)
        if index_exists:
        #load it  
            print("loading Joise Index......")
            with open(index_file_path, 'rb') as file:
                table_raw_index = pickle.load(file)
        else:    
            
            for key, value in table_raw_proccessed_los.items():
                
                index = SearchIndex(value, similarity_func_name="jaccard", similarity_threshold=0.0)
                table_raw_index[key]= index   
                
            # write in a pickle file  
            with open(index_file_path, 'wb') as file:
                    pickle.dump(table_raw_index, file)   
        
        
        self.table_raw_index=table_raw_index
        self.table_path=table_path
        #DSize is a hyper parameter
        
        
        
        
  
    
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
            
    def vectorize_tables(self):
        '''we need to represent each table as a point in space so for  I represent each query table and unionable table produced from  their aligned columns 
        '''
        combining_methods={"sum", "average"}
        (q_emb_dict, dl_emb_dict)=self.starmie_vectors
        
        # every query table and its corresponind unionable tables is loaded to  self.data 
        # Columns in data:  required_columns = ['query_table_name', 'query_column', 'query_column#',
        #                         'dl_table_name', 'dl_column#', 'dl_column']
       
        # for every query and s_i pair we get alignments get the columns vector
        # representation and sum their vectors to represent query and data lake tables and write to a new dictionary 
        # key is (query, dl_table) value is (query_vector, dl_table_vector)
       
        
# Define the column names
        columns_query_dl_table_vector_df = ["query", "si", "query_sum_vector", "query_avg_vector", "s_i_sum_vector", "s_i_avg_vector"]

# Create an empty DataFrame
        query_dl_table_vector_df = pd.DataFrame(columns=columns_query_dl_table_vector_df)
        # get the query names 
        query_dl_table_vector_vectors_exists = os.path.isfile('query_dl_table_vector.csv')
        if  query_dl_table_vector_vectors_exists:
            #load 
            with open('query_dl_table_vector.pkl', "rb") as file:
                 self.query_dl_table_vector_df = pickle.load(file)
        else:   
                query_table_name_set = set(self.q_dl_alignemnt['query_table_name'])
                for query in query_table_name_set:
                    
                    # Filter rows where 'query_table_name' column has the value 4
                    filtered_df = self.q_dl_alignemnt[self.q_dl_alignemnt['query_table_name'] == query]

                    # Get the set of all values in column 'W' for the filtered rows
                    dl_table_names= set(filtered_df['dl_table_name'])
                    for dl_table in dl_table_names:
                        # get columns aligned from query 
                        filtered_rows = filtered_df[filtered_df['dl_table_name'] == dl_table]
                        
                        # Extract values from the 'query_column' column
                        query_columns= set(filtered_rows['query_column#'].tolist()  )             
                        # go through the couluns and add their starmie representaiotn 
                        query_table_vector_sum=self.sum_vectors(q_emb_dict,query,query_columns)
                        query_table_vector_avg=self.average_vectors(q_emb_dict,query,query_columns)
                
                
                        dl_table_columns = set(filtered_rows['dl_column'].tolist() )
                        dl_table_vector_sum=self.sum_vectors(dl_emb_dict,dl_table,dl_table_columns)
                        dl_table_vector_avg=self.average_vectors(dl_emb_dict,dl_table,dl_table_columns)
                
                            
                        
                        # Define values for a new row
                        new_row = {
                            "query": query,
                            "si": dl_table,
                            "query_sum_vector": query_table_vector_sum,  # Example vector
                            "query_avg_vector": query_table_vector_avg,  # Example vector
                            "s_i_sum_vector": dl_table_vector_sum,    # Example vector
                            "s_i_avg_vector": dl_table_vector_avg     # Example vector
                        }

                        # Append the new row to the DataFrame
                        self.query_dl_table_vector_df = pd.concat([self.query_dl_table_vector_df, pd.DataFrame([new_row])], ignore_index=True)
                self.query_dl_table_vector_df.to_csv('query_dl_table_vector.csv',columns=query_dl_table_vector_df.columns ,index=False)

                self.query_dl_table_vector_df.to_pickle('query_dl_table_vector.pkl')

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        # given a query alignemnt between its paor of uninable  tables are loaded to alignemnt_diversity_df: 
        # Define the required columns
        required_columns = ["q_main_table", "s_i", "s_i_column", "s_i_column#", "s_j_table_name", "s_j_column#", "s_j_column"]
        #required_columns = ["q_main_table", "s_i", "s_i_column#", "s_j_table_name", "s_j_column"]
        alignemnt_diversity_df = pd.read_csv(self.alignment_for_diversity_gmc_file, usecols=required_columns)
        
        
        
        # Define the column names
        columns_query_s_i_s_j_vector = ["query", "s_i", "s_j","s_i_sum_vector", "s_i_avg_vector", "s_j_sum_vector", "s_j_avg_vector"]

# Create an empty DataFrame
        query_s_i_s_j_vector_df = pd.DataFrame(columns=columns_query_s_i_s_j_vector)

        query_s_i_s_j_vector_exists = os.path.isfile('query_s_i_s_j_vector.pkl')
        if  query_s_i_s_j_vector_exists:
            #load 
            with open('query_s_i_s_j_vector.pkl', "rb") as file:
                 self.query_s_i_s_j_vector_df = pickle.load(file)
        
        # get the query names 
        else: 
                query_table_name_set = set(alignemnt_diversity_df['q_main_table'])
                for query in query_table_name_set:  
                    filtered_df = alignemnt_diversity_df[alignemnt_diversity_df['q_main_table'] == query]
                    si_names= set(filtered_df['s_i'])
                    for si in si_names:
                        # get columns aligned from query 
                        filtered_rows = filtered_df[filtered_df['s_i'] == si]
                        sj_tables=set(filtered_rows['s_j_table_name'])            
                        for sj in sj_tables:
                            s_j_part=filtered_rows[filtered_rows['s_j_table_name'] == sj]
                            si_columns = set(s_j_part['s_i_column#'].tolist())
                            sj_columns = set(s_j_part['s_j_column'].tolist())
                        
                            si_vector_sum=self.sum_vectors(dl_emb_dict,si,si_columns)
                            si_vector_avg=self.average_vectors(dl_emb_dict,si,si_columns)

                            sj_vector_sum=self.sum_vectors(dl_emb_dict,sj,sj_columns)
                            sj_vector_avg=self.average_vectors(dl_emb_dict,sj,sj_columns)
                            # get column aligned 

                            # Define values for a new row
                            # = ["query", "s_i", "s_j","s_i_sum_vector", "s_i_avg_vector", "s_j_sum_vector", "s_j_avg_vector"]

                            new_row = {
                            "query": query,
                            "s_i": si,
                            "s_j": sj,  # Example vector
                            "s_i_sum_vector": si_vector_sum,  # Example vector
                            "s_i_avg_vector": si_vector_avg,    # Example vector
                            "s_j_sum_vector": sj_vector_sum,   
                            "s_j_avg_vector": sj_vector_avg    # Example vector 
                            }

                        # Append the new row to the DataFrame
                            self.query_s_i_s_j_vector_df = pd.concat([self.query_s_i_s_j_vector_df, pd.DataFrame([new_row])], ignore_index=True)
                            
                self.query_s_i_s_j_vector_df.to_csv('query_s_i_s_j_vector.csv',columns=query_s_i_s_j_vector_df.columns ,index=False)
                self.query_s_i_s_j_vector_df.to_pickle('query_s_i_s_j_vector.pkl')


        print("table representaion generation is done")

    def execute_search(self):
        """
        Perform the GMC search based on the parameters.
        """
        # retrun k result
        k = self.k
        lmda = 0.7 # best practice from original paper
        all_results={}
        queries=self.unionable_data.keys()
      
            # apply filter
            
        for query in queries:
              #get the unionable tables for query
              S=self.unionable_data[query]
              (gmc_results, time_took)= self.gmc(S, query,k = k, lmda=lmda)
              all_results[query]=  (gmc_results, time_took)
        return all_results
    
    def filter_results(self, criteria):
        """
        Filters the search results based on specified criteria.

        :param criteria: Dictionary containing filtering rules.
        :return: List of filtered results
        """
        # TODO: Implement filtering logic
        pass

    def export_results(self, output_path):
        """
        Export the search results to a specified location.

        :param output_path: Path to save the results (e.g., file path).
        :return: None
        """
        # TODO: Implement logic to save results
        pass

    def summarize_results(self):
        """
        Provide a summary or statistics of the search results.

        :return: Dictionary with summary details
        """
        # TODO: Implement logic to summarize the results
        pass
    def mmc_compute_div_sum(self, s_i: str, q_name, R_p: set) -> float:
        total_score = 0
        for s_j in R_p:
            total_score +=self.calculate_diversity(q_name, s_i, s_j)
        return total_score

    def mmc_compute_div_large(self, s_i: str, q_name, remaining_s_set : set, max_l : int) -> float:  # max_l should be used as: < max_l ; not <= max_l
        div_l_list = [] # we will use the first l values after sorting this.
        for s_j in remaining_s_set: # the items that are not inserted in R_p yet
            div_l_list.append(self.calculate_diversity(q_name,s_i,s_j))
        div_l_list = sorted(div_l_list, reverse=True)[:max_l - 1]
        return div_l_list
    
    def mmc(self, q_name,s_set: set, lmda : float, k: int, R_p: set) -> dict: 
        all_mmc = dict()
        p = len(R_p) - 1
        div_coefficient = lmda / (k - 1)
        for s_i in s_set:
            sim_term= (1 - lmda)*self.calculate_unionability(q_name,s_i)
            div_term1 = div_coefficient * self.mmc_compute_div_sum(s_i, q_name, R_p)
            temp_s=set(s_set) - R_p - {s_i}
            temp=self.mmc_compute_div_large(s_i, q_name,temp_s , k - p)
            temp_sum=sum(temp)
            div_term2 = div_coefficient * temp_sum
            current_mmc = sim_term + div_term1 + div_term2
            all_mmc[s_i] = current_mmc
        return all_mmc
    
    
    


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

    @staticmethod 
    def avg_div_sim(q_name, s_prime, unionability_scores,diversity_scores ):
        'Besat code'
        # compute average of similarity component 
        filtered_unionability_scores = unionability_scores[unionability_scores['q_table'] == q_name]
        sigma=0
        for s_i in s_prime:
            filtered_row = filtered_unionability_scores[filtered_unionability_scores['dl_table'] == s_i]
            similarity_score_value = filtered_row['similarity_score'].values[0]
            sigma=sigma+similarity_score_value
            
        sim_comp=  sigma/float(len(s_prime)) 
            
        # compute average of div component
            
        sigma=0
        
        for i in range(0, len(s_prime)-1):
            for j in range (i+1, len(s_prime)):
                s_i=s_prime[i]
                s_j=s_prime[j] 
                
                row_1=diversity_scores[
                (diversity_scores['q_main_table'] == q_name) &  
                (diversity_scores['s_i'] == s_i) & (diversity_scores['s_j'] == s_j)]
               
                row_2=diversity_scores[
                (diversity_scores['q_main_table'] == q_name) &  
                (diversity_scores['s_i'] == s_j) & (diversity_scores['s_j'] == s_i)]
                
                if(row_1.shape[0]==1): 
                    diversity_score = row_1['diversity_score'].values[0]
                elif(row_2.shape[0]==1):    
                    diversity_score = row_2['diversity_score'].values[0]
                else:
                    # this means that the given the query no alignemnt between s_i and s_j has been found by DUST so we assign 0  
                    diversity_score=0
                    #raise ValueError('diversity score file has invalid entry')  

                sigma=sigma+diversity_score
        
        div_comp=sigma/float(len(s_prime)*(len(s_prime)-1))
        
        res= sim_comp+div_comp
        
        return res/2.0
        
    @staticmethod 
    def max_div_sim( q_name, s_prime, unionability_scores,diversity_scores):
        # Besat code
        filtered_unionability_scores = unionability_scores[unionability_scores['q_table'] == q_name]
        
        filtered_rows_1= filtered_unionability_scores[
    (filtered_unionability_scores['dl_table'].isin(s_prime)) 
        ]
        max_value_sim = filtered_rows_1['similarity_score'].max()
  
        '''Filter all rows those s_i and s_j is in s_prime'''
        filtered_rows_2=diversity_scores[diversity_scores['q_main_table']==q_name]
 
        filtered_rows = filtered_rows_2[
    (filtered_rows_2['s_i'].isin(s_prime)) &
    (filtered_rows_2['s_j'].isin(s_prime))
]
        max_value_div = filtered_rows['diversity_score'].max()
        overall_max = max(max_value_sim, max_value_div)
        return overall_max

    @staticmethod 
    def min_div_sim(q_name, s_prime,unionability_scores, diversity_scores):  
        '''Besat code: ind the minimum score amon all the sim and div scores for a q in S_prime'''
        filtered_unionability_scores = unionability_scores[unionability_scores['q_table'] == q_name]
        
        filtered_rows_1= filtered_unionability_scores[
    (filtered_unionability_scores['dl_table'].isin(s_prime)) 
        ]
        min_value_sim = filtered_rows_1['similarity_score'].min()
  
        '''Filter all rows those s_i and s_j is in s_prime'''
        filtered_rows_2=diversity_scores[diversity_scores['q_main_table']==q_name]
 
        filtered_rows = filtered_rows_2[
    (filtered_rows_2['s_i'].isin(s_prime)) &
    (filtered_rows_2['s_j'].isin(s_prime))
]
        min_value_div = filtered_rows['diversity_score'].min()
        overall_min = min(min_value_sim, min_value_div)
        # retrun minumum 
        return overall_min     
    # Besat designed function
    def compute_metrics_instancedBased(self, result):
  
        # Define column names
        columns = ["query_table", "avg_div_sim", "max_div_sim", "min_div_score", "k", "gmc_exec_time_secs"]

# Create an empty DataFrame with the specified columns
        metrics_df = pd.DataFrame(columns=columns)

        # go through result and for every query compute two functions: 
        for q, r in result.items():
            s_prime=r[0]
            time_sec=r[1]
    
            if(q == 'workforce_management_information_a.csv' or q== 'workforce_management_information_b.csv'):
                continue 
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
    "k":self.k,
    "gmc_exec_time_secs":time_sec
}

# Add the row to the DataFrame
            metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True)

# Write the DataFrame to a CSV file
        csv_file_path = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/evaluation_metrics_gmc.csv"
        
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
    
 
    @staticmethod
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
  
  
  
  # a function to calculate  precision recall and MAP for all queries all together
    @staticmethod
    def Cal_P_R_Map_Starmie(resultFile, gtPath):
        ''' Calculate and log the performance metrics: MAP, Precision@k, Recall@k
    Args:
        max_k: the maximum K value (e.g. for SANTOS benchmark, max_k = 10. For TUS benchmark, max_k = 60)
        k_range: different k s that are reported in the result
        gtPath: file path to the groundtruth
        resPath: file path to the raw results from the model
    Return: MAP, P@K, R@K'''
        groundtruth = GMC_Search.loadDictionaryFromPickleFile(gtPath)
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
                ideal_recall.append(k/float(len(groundtruth[table]))) 
                precision = true_positive / (true_positive + false_positive)
                recall = rec/len(resultFile)
                precision_array.append(precision)
                recall_array.append(recall)


  
  
  
    @staticmethod
  # a function to calculate  precision recall and MAP per query
    def Cal_P_R_Map(resultFile, gtPath, output_file_):
        ''' Calculate and log the performance metrics: MAP, Precision@k, Recall@k
    Args:
        max_k: the maximum K value (e.g. for SANTOS benchmark, max_k = 10. For TUS benchmark, max_k = 60)
        k_range: different k s that are reported in the result
        gtPath: file path to the groundtruth
        resPath: file path to the raw results from the model
    Return: MAP, P@K, R@K'''
        groundtruth = GMC_Search.loadDictionaryFromPickleFile(gtPath)
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
        
    
    
    @staticmethod
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
        tokens = GMC_Search.case_fold(tokens)
            # Stemming
        stemmed_tokens = GMC_Search.stem(stemmer, tokens)
        merged_string = ' '.join(stemmed_tokens)
        return merged_string
    
    
    @staticmethod
    def case_fold(tokens):
        """Converts all tokens to lowercase."""
        return [token.lower() for token in tokens]
    @staticmethod
    def stem( stemmer, tokens):
        """Applies stemming to the tokens."""
        return [stemmer.stem(token) for token in tokens]
    
    @staticmethod
      # Function to count rows where query_name appears in tables for each unique value of k
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
    @staticmethod
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
        results_df = GMC_Search.compute_counts(df, k_column,query_name_column, tables_column)

        # Output the results
   
        results_df.to_csv(output_file, index=False)


                
    
    @staticmethod 
    def compute_metrics(result, alg_, csv_file_path, k,unionability_scores, diversity_scores):
            '''alg_: {gmc, penal}'''
            # Define column names
            columns = ["query_table", "avg_div_sim", "max_div_sim", "min_div_score", "k", alg_+"_exec_time_secs"]

    # Create an empty DataFrame with the specified columns
            metrics_df = pd.DataFrame(columns=columns)

            # go through result and for every query compute two functions: 
            for q, r in result.items():
                s_prime=r[0]
                time_sec=r[1]
        
                if(q == 'workforce_management_information_a.csv' or q== 'workforce_management_information_b.csv'):
                    continue 
                print("Genarating Evaluation for query: "+q)
                avg_div_sim=GMC_Search.avg_div_sim(q,s_prime,unionability_scores,diversity_scores)
                max_div_sim=GMC_Search.max_div_sim(q,s_prime,unionability_scores,diversity_scores)
                min_div_score=GMC_Search.min_div_sim(q,s_prime,unionability_scores,diversity_scores)
                
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
    def is_diluted_version(fname):
       """if it has _dlt showes is diluted then retrun original file name 
           else retrun -1"""
       if('_dlt' in fname) :
            return fname.replace('_dlt', '')
       else: 
        return -1   
    
    @staticmethod 
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
                    deluted_=GMC_Search.is_diluted_version(t)
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
           
    
    @staticmethod 
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
                    deluted_=GMC_Search.is_diluted_version(t)
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
       
 
    @staticmethod 
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
                          q_snm,L, G=GMC_Search.get_ssnm_query(df_k_q, groundtruth_dic_q)
                          if(q_snm==-1):
                           q_not_valid_snm.add(q)
                          else:
                            number_queries=number_queries+1
 
                            snm_total=snm_total+q_snm
                        else: 
                            print("excluded query is found")    
        return  (float(snm_total)/ float(number_queries), q_not_valid_snm) 
    
    @staticmethod 
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
                          q_snm, B, L, G=GMC_Search.get_snm_query(df_k_q, groundtruth_dic_q)
                          if(q_snm==-1):
                           q_not_valid_snm.add(q)
                          else:
                            number_queries=number_queries+1
 
                            snm_total=snm_total+q_snm
                        else: 
                            print("excluded query is found")    
        return  (float(snm_total)/ float(number_queries), q_not_valid_snm)               

    
    @staticmethod 
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
                          q_snm, B, L, G=GMC_Search.get_snm_query(df_k_q, groundtruth_dic_q)
                          results_k.append({'k': k, 'query': q,'snm': q_snm, 'B':B, 'L':L, 'G':G})
                        else: 
                            print("excluded query is found")    
        return   results_k           

    @staticmethod 
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
                          q_snm, L, G=GMC_Search.get_ssnm_query(df_k_q, groundtruth_dic_q)
                          results_k.append({'k': k, 'query': q,'snm': q_snm, 'L':L, 'G':G})
                        else: 
                            print("excluded query is found")    
        return   results_k     

    @staticmethod 
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
               df_all = df_all.applymap(GMC_Search.normalize) 
            union_size=union_size+len(df_all.drop_duplicates())

        return (union_size, all_tables_size, query_size)
            
        
        
        
        
        

        
        
        
        
    
 
    @staticmethod
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
                    (union_size, all_tables_size, query_size)=GMC_Search.perform_union_get_size(q, aligned_columns_tbl,alignments_, qs , tbles_, normalized)    
                    
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
        
        
        

    
    @staticmethod 
    def compute_syntactic_novelty_measure(groundtruth_file, search_result, snm_avg_result_path_, snm_whole_result_path_):
        """
        this function only makes sense to be run over diluted dataset
        we assume 2 things: 1- the list of dl_tables for each query is sorted descending bu unionability score
                            2-  search_result file is a csv has atleast  these columns: query_name,	tables,	execution_time,	k  
                            groundtruth: is the gound truth havinf unionable tables for each query
        """                    
        groundtruth = GMC_Search.loadDictionaryFromPickleFile(groundtruth_file)

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
            avg_snm, invalid_snm = GMC_Search.get_snm_average(subset_df, groundtruth)
            temp_whole=GMC_Search.get_snm_whole(subset_df, groundtruth, k)
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
            
   
    @staticmethod 
    def compute_syntactic_novelty_measure_simplified(groundtruth_file, search_result, snm_avg_result_path_, snm_whole_result_path_):
        """
        this function only makes sense to be run over diluted dataset
        we assume 2 things: 1- the list of dl_tables for each query is sorted descending bu unionability score
                            2-  search_result file is a csv has atleast  these columns: query_name,	tables,	execution_time,	k  
                            groundtruth: is the gound truth havinf unionable tables for each query
        """                    
        groundtruth = GMC_Search.loadDictionaryFromPickleFile(groundtruth_file)

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
            avg_snm, invalid_snm = GMC_Search.get_ssnm_average(subset_df, groundtruth)
            temp_whole=GMC_Search.get_ssnm_whole(subset_df, groundtruth, k)
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

   
    
    #analyse the results 
    @staticmethod

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
    
    
    def compute_metric_old(self, result, dl_embeddings:dict, query_embeddings:dict, lmda:float, k:float, print_results = False, normalize = False, metric = "", max_metric = True):
        computed_metrics = [] # list of dictionaries, each dict is a row in the evaluation dataframe. 
        q = np.mean(list(query_embeddings.values()), axis=0)
        ranking_without_query = {}
        for key in result:
            ranking_without_query[key] = dl_embeddings[key]
        final_ranking_with_query = ranking_without_query.copy()
        for key in query_embeddings:
            final_ranking_with_query[key] = query_embeddings[key]
            dl_embeddings[key] = query_embeddings[key] # we do not need separate data lake embeddings anymore, so merging with query.
        
        R_without_query = set(result)
        R_with_query = set(result).union(set(query_embeddings.keys()))
        if max_metric == True:
            if metric == "" or metric == "cosine":
                # Evaluating max diversity using cosine distance:
                sim_dict = d_sim(dl_embeddings, q, metric="cosine", normalize=normalize) # index the similarity between each item in S and q for once so that we can re-use them.
                div_dict = d_div(dl_embeddings, metric = "cosine", normalize=normalize) # index the diversity between each pair of items in S so that we can re-use them.
                cosine_with_query_max_scores = self.f_prime(R_with_query, lmda, k, sim_dict, div_dict)
                cosine_wo_query_max_scores =  self.f_prime(R_without_query, lmda, k, sim_dict, div_dict)
                
                if print_results == True:
                    print("Evaluating max diversity using cosine distance:")
                    print("max score with query: ", cosine_with_query_max_scores)
                    print("max score without query: ", cosine_wo_query_max_scores)
                    print("\n=================================================\n")

            if metric == "" or metric == "l1":
                # Evaluating max diversity using l1 distance:
                sim_dict = d_sim(dl_embeddings, q, metric="l1", normalize=normalize) # index the similarity between each item in S and q for once so that we can re-use them.
                div_dict = d_div(dl_embeddings, metric = "l1", normalize=normalize) # index the diversity between each pair of items in S so that we can re-use them.
                l1_with_query_max_scores = f_prime(R_with_query, lmda, k, sim_dict, div_dict)
                l1_wo_query_max_scores = f_prime(R_without_query, lmda, k, sim_dict, div_dict)
                
                if print_results == True:
                    print("Evaluating max diversity using l1 distance:")
                    print("max score with query: ", l1_with_query_max_scores)
                    print("max score without query: ", l1_wo_query_max_scores)
                    print("\n=================================================\n")
            if metric == "" or metric == "l2":
                # Evaluating max diversity using l2 distance:
                sim_dict = d_sim(dl_embeddings, q, metric="l2", normalize=normalize) # index the similarity between each item in S and q for once so that we can re-use them.
                div_dict = d_div(dl_embeddings, metric = "l2", normalize=normalize) # index the diversity between each pair of items in S so that we can re-use them.
                l2_with_query_max_scores = f_prime(R_with_query, lmda, k, sim_dict, div_dict)
                l2_wo_query_max_scores = f_prime(R_without_query, lmda, k, sim_dict, div_dict)            
            if print_results == True:
                print("Evaluating max diversity using l2 distance:")
                print("max score with query: ", l2_with_query_max_scores)
                print("max score without query: ", l2_wo_query_max_scores)
                print("\n=================================================\n")
        else:
            cosine_with_query_max_scores = np.nan
            cosine_wo_query_max_scores = np.nan
            l1_with_query_max_scores = np.nan
            l1_wo_query_max_scores = np.nan
            l2_with_query_max_scores = np.nan
            l2_wo_query_max_scores = np.nan
        if metric == "" or metric == "cosine":    
            # Evaluating max-min diversity and average distance using cosine distance:
            cosine_with_query_min_scores = min_div_score(final_ranking_with_query, metric="cosine", normalize=normalize)
            cosine_wo_query_min_scores = min_div_score(ranking_without_query, metric = "cosine", normalize=normalize)
            cosine_with_query_avg_scores =  sum(cosine_with_query_min_scores) / len(cosine_with_query_min_scores)
            cosine_wo_query_avg_scores = sum(cosine_wo_query_min_scores) / len(cosine_wo_query_min_scores)
            
            # The function below computes distance between pairs of data lake and data lake points, data lake and query points but not query and query points. 
            cosine_w_mix_query_min_scores = min_mix_div_score(final_ranking_with_query, set(query_embeddings.keys()), metric="cosine", normalize=normalize)
            cosine_w_mix_query_avg_scores =  sum(cosine_w_mix_query_min_scores) / len(cosine_w_mix_query_min_scores)

            if print_results == True:
                print ("Evaluating max-min diversity and average distance using cosine distance:")
                print("max-min score with query: ", min(cosine_with_query_min_scores))
                print("max-min score without query: ", min(cosine_wo_query_min_scores))
                print("max-min score with mix query: ", min(cosine_w_mix_query_min_scores))
                print("average distance with query: ", cosine_with_query_avg_scores)
                print("average distance without query: ", cosine_wo_query_avg_scores)
                print("average distance with mix query: ", cosine_w_mix_query_avg_scores)
                print("\n=================================================\n")
        if metric == "" or metric == "l1":
            # Evaluating max-min diversity and average distance using l1 distance:
            l1_with_query_min_scores = min_div_score(final_ranking_with_query, metric="l1", normalize=normalize)
            l1_wo_query_min_scores = min_div_score(ranking_without_query, metric = "l1", normalize=normalize)
            l1_with_query_avg_scores = sum(l1_with_query_min_scores) / len(l1_with_query_min_scores)
            l1_wo_query_avg_scores = sum(l1_wo_query_min_scores) / len(l1_wo_query_min_scores)
            
            # The function below computes distance between pairs of data lake and data lake points, data lake and query points but not query and query points. 
            l1_w_mix_query_min_scores = min_mix_div_score(final_ranking_with_query, set(query_embeddings.keys()), metric="l1", normalize=normalize)
            l1_w_mix_query_avg_scores =  sum(l1_w_mix_query_min_scores) / len(l1_w_mix_query_min_scores)

            if print_results == True:
                print ("Evaluating max-min diversity and average distance using l1 distance:")
                print("max-min score with query: ", min(l1_with_query_min_scores))
                print("max-min score without query: ", min(l1_wo_query_min_scores))
                print("max-min score with mix query: ", min(l1_w_mix_query_min_scores))
                print("average distance with query: ", l1_with_query_avg_scores)
                print("average distance without query: ", l1_wo_query_avg_scores)
                print("average distance with mix query: ", l1_w_mix_query_avg_scores)
                print("\n=================================================\n")
        
        if metric == "" or metric == "l2":
            # Evaluating max-min diversity and average distance using l2 distance:
            l2_with_query_min_scores = min_div_score(final_ranking_with_query, metric= "l2", normalize=normalize)
            l2_wo_query_min_scores = min_div_score(ranking_without_query, metric = "l2", normalize=normalize)
            l2_with_query_avg_scores = sum(l2_with_query_min_scores) / len(l2_with_query_min_scores)
            l2_wo_query_avg_scores = sum(l2_wo_query_min_scores) / len(l2_wo_query_min_scores)
            
            # The function below computes distance between pairs of data lake and data lake points, data lake and query points but not query and query points. 
            l2_w_mix_query_min_scores = min_mix_div_score(final_ranking_with_query, set(query_embeddings.keys()), metric="l2", normalize=normalize)
            l2_w_mix_query_avg_scores =  sum(l2_w_mix_query_min_scores) / len(l2_w_mix_query_min_scores)

            if print_results == True:
                print("Evaluating max-min diversity and average distance using l2 distance:")
                print("score with query: ", min(l2_with_query_min_scores))
                print("score without query: ", min(l2_with_query_min_scores))
                print("max-min score with mix query: ", min(l2_w_mix_query_min_scores))
                print("average distance with query: ", l2_with_query_avg_scores)
                print("average distance without query: ", l2_wo_query_avg_scores)
                print("average distance with mix query: ", l2_w_mix_query_avg_scores)
                print("\n=================================================\n")
        
        # create 6 dictionaries to store all the calculations
        if metric == "" or metric == "cosine": 
            computed_metrics.append({"metric": "cosine", "with_query" : "yes", "max_score": cosine_with_query_max_scores, "max-min_score": min(cosine_with_query_min_scores), "avg_score": cosine_with_query_avg_scores})
            computed_metrics.append({"metric": "cosine", "with_query": "no", "max_score": cosine_wo_query_max_scores, "max-min_score": min(cosine_wo_query_min_scores), "avg_score": cosine_wo_query_avg_scores})
            computed_metrics.append({"metric": "cosine", "with_query": "mix", "max_score": np.nan, "max-min_score": min(cosine_w_mix_query_min_scores), "avg_score": cosine_w_mix_query_avg_scores})
            

        if metric == "" or metric == "l1":
            computed_metrics.append({"metric": "l1", "with_query" : "yes", "max_score": l1_with_query_max_scores, "max-min_score": min(l1_with_query_min_scores), "avg_score": l1_with_query_avg_scores})
            computed_metrics.append({"metric": "l1", "with_query": "no", "max_score": l1_wo_query_max_scores, "max-min_score": min(l1_wo_query_min_scores), "avg_score": l1_wo_query_avg_scores})
            computed_metrics.append({"metric": "l1", "with_query": "mix", "max_score": np.nan, "max-min_score": min(l1_w_mix_query_min_scores), "avg_score": l1_w_mix_query_avg_scores})

        if metric == "" or metric == "l2":
            computed_metrics.append({"metric": "l2", "with_query" : "yes", "max_score": l2_with_query_max_scores, "max-min_score": min(l2_with_query_min_scores), "avg_score": l2_with_query_avg_scores})
            computed_metrics.append({"metric": "l2", "with_query": "no", "max_score": l2_wo_query_max_scores, "max-min_score": min(l2_wo_query_min_scores), "avg_score": l2_wo_query_avg_scores})
            computed_metrics.append({"metric": "l2", "with_query": "mix", "max_score": np.nan, "max-min_score": min(l2_w_mix_query_min_scores), "avg_score": l2_w_mix_query_avg_scores})

        return computed_metrics, embedding_plot
    

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
        total_time = round(int(end_time - start_time) / 10 ** 9, 2)
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
            avg col sims to get table similarity and this is unionability  or similaryty score and then retun the unionbility for the pair
        '''
        all_vectors=self.starmie_vectors
        #all_vectors=self.load_starmie_vectors()
        queries_dict = all_vectors[0]
        dl_dict=all_vectors[1]
        q_vectors=queries_dict[query_name]
        query_rows = self.q_dl_alignemnt[self.q_dl_alignemnt['query_table_name'] == query_name]
        similarity_score=0

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

        similarity_score=similarity_score/num_rows
       
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
           diversity between two tables are the average of diversity between their columns for now 
         """
        DSize=self.Dsize

        div_score=0
        # alignment between query  and s_i
        query_rows_si = self.q_dl_alignemnt[(self.q_dl_alignemnt['query_table_name'] == query_name) & (self.q_dl_alignemnt['dl_table_name'] == s_i)] 
        
        aligned_col_q_si = set(query_rows_si['query_column#'])

        # alignment between query  and s_j
        query_rows_sj = self.q_dl_alignemnt[(self.q_dl_alignemnt['query_table_name'] == query_name) & (self.q_dl_alignemnt['dl_table_name'] == s_j)] 
        aligned_col_q_sj = set(query_rows_sj['query_column#'])

  # for columns from i , j that are alignemed to identical query colmn we calculate the dicersity 
  # for those they are aligned with different columns of q  based on the null assumption we assign 1 for the score 

        alignemd_same_qcolumns=aligned_col_q_si.intersection(aligned_col_q_sj)
        aligned_col_q_si_solo=aligned_col_q_si.difference(alignemd_same_qcolumns)
        aligned_col_q_sj_solo=aligned_col_q_sj.difference(alignemd_same_qcolumns)
        
           
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
                            
                    # add one for each null columns 
                    div_score =div_score+len(aligned_col_q_si_solo) 
                                # add one for each null columns 
                    div_score =div_score+len(aligned_col_q_sj_solo) 
                    num_rows=len(aligned_col_q_sj_solo) +len(aligned_col_q_si_solo) +len(alignemd_same_qcolumns)
        
                    div_score=div_score/num_rows        
            
        return div_score
 
    
      
                   
                                  
       

    
def d_sim(self,s_dict : dict, q_embedding: np.ndarray, metric = "cosine", normalize = False) -> dict:
    sim_dict = dict() # key: s_dict key i.e. s_id; value : similarity score
    for current_s, current_s_embedding in s_dict.items():
        if metric == "l1":
            if normalize == True:
                max_possible_l1 = 2 * len(current_s_embedding)
            else:
                max_possible_l1 = 1
            current_sim = np.linalg.norm(current_s_embedding - q_embedding, ord = 1) / max_possible_l1
        elif metric == "l2":
            if normalize == True:
                max_possible_l2 = np.sqrt(2 * len(current_s_embedding))
            else:
                max_possible_l2 = 1
            current_sim = np.linalg.norm(current_s_embedding - q_embedding, ord = 2) / max_possible_l2
        else: # cosine
            if normalize == True:
                current_sim = 1 - ((utl.CosineSimilarity(current_s_embedding, q_embedding) + 1 ) / 2)
            else:
                current_sim = 1 - utl.CosineSimilarity(current_s_embedding, q_embedding)
        sim_dict[current_s] = current_sim
    return sim_dict

def d_div(self,s_dict : dict, metric = "cosine", normalize = False) -> dict:
    div_dict = dict() # key: s_dict key i.e. s_id; value : similarity score
    for current_s1 in s_dict:
        for current_s2 in s_dict:
            if metric == "l1":
                max_possible_l1 = 2 * len(s_dict[current_s1])
                if normalize == True:
                    current_div = 1 - (np.linalg.norm(s_dict[current_s1] - s_dict[current_s2], ord = 1) / max_possible_l1)
                else:
                    current_div = max_possible_l1 - (np.linalg.norm(s_dict[current_s1] - s_dict[current_s2], ord = 1) / 1)                
            elif metric == "l2":
                max_possible_l2 = np.sqrt(2 * len(s_dict[current_s1]))
                if normalize == True:
                    current_div = 1 - (np.linalg.norm(s_dict[current_s1] - s_dict[current_s2], ord = 2) / max_possible_l2)
                else:
                    current_div = max_possible_l2 - (np.linalg.norm(s_dict[current_s1] - s_dict[current_s2], ord = 2) / 1)
            else: #cosine
                if normalize == True:
                    current_div = (utl.CosineSimilarity(s_dict[current_s1], s_dict[current_s2]) + 1) / 2 #normalized score between 0 and 1
                else:
                    current_div = utl.CosineSimilarity(s_dict[current_s1], s_dict[current_s2])
            div_dict[(current_s1, current_s2)] = current_div
    return div_dict


      
if __name__ == "__main__":
    # benchmark_path="/u6/bkassaie/NAUS/data/ugen_v2/"
    benchmark_path="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/"
    
    # alignment_Dust="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_CL_KMEANS_cosine_alignment_diluted.csv"
    alignment_Dust="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/ugenv2_small_manual_alignment_all.csv"

    #alignment_Dust="data/santos/Santos_dlt_CL_KMEANS_cosine_alignment.csv"
    

    # top_k_starmie=benchmark_path+"diveristy_data/search_results/Starmie/top_20_Starmie_output_04diluted_restricted_noscore.pkl"     
    top_k_starmie=benchmark_path+"diveristy_data/search_results/Starmie/top_20_Starmie_output_04diluted_restricted_noscore.pkl"     

    search_params = {"keyword": "example", "max_results": 10}
    dl_table_vectors_path = "data/ugen_v2/ugenv2_small/vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl"
    query_table_vectors_path = "data/ugen_v2/ugenv2_small/vectors/cl_query_drop_col_tfidf_entity_column_0.pkl"
   # domainsize=20
    domainsize=2  #only for ugenv2_small, which has few rows per table
    
    gmc_search = GMC_Search(alignment_Dust, domainsize, dl_table_vectors_path,query_table_vectors_path,benchmark_path,search_params)

    # we leaod the data corresponindt to acolumn alignment generated by DUST for the output of
    # Starmie on Santos returning maximum 50 unionable dl tables  for each query
    gmc_search.load_unionable_tables(top_k_starmie) 
        
    #generate and persist alignments for diversity function 

    output_csv_file = benchmark_path+'diveristy_data/search_results/GMC/gmc_results_diluted04_restricted.csv'
    
    
    for i in range(2, 11): 
            gmc_search.k=i

            results = gmc_search.execute_search()
            # Define the output CSV file path
            

        # Write the dictionary to a CSV file
            if os.path.exists(output_csv_file):
                with open(output_csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    # Write the data
                    for query_name, (result, secs) in results.items():
                        # Join the list of results into a string, if needed
                        result_str = ', '.join(result) if isinstance(result, list) else str(result)
                        writer.writerow([query_name, result_str,secs,gmc_search.k])
            else: 
                with open(output_csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['query_name', 'tables','gmc_execution_time', 'k'])

                    # Write the data
                    for query_name, (result, secs) in results.items():
                        # Join the list of results into a string, if needed
                        result_str = ', '.join(result) if isinstance(result, list) else str(result)
                        writer.writerow([query_name, result_str,secs,gmc_search.k])
                    # Write the header

        



    
    
   