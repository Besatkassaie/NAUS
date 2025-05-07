import pandas as pd
from naive_search_Novelty import NaiveSearcherNovelty
import test_naive_search_Novelty
from SetSimilaritySearch import SearchIndex
from  process_column import TextProcessor
import pickle
import os
from preprocess_align import gmc_alignmnet_by_query
from preprocess_align import initialize_globally
import csv
import numpy as np
import time
from numpy.linalg import norm
from scipy.spatial import distance
from collections import Counter
import utilities as utl
from SetSimilaritySearch import all_pairs
from GMC_search import GMC_Search
import numpy as np



class Semantic_Novelty:
    """
    Pe: Penalized_Search is a class for performing penalized re_ranking of unionable tables for Novelty based  unionable  table search.
    """

    def __init__(self, table_path):
            self.alignment_data=None
            self.unionable_tables=None       
            self.table_path=table_path
        
        
        
        
        
   
    def _cosine_sim(self, vec1, vec2):
        ''' Get the cosine similarity of two input vectors: vec1 and vec2
        '''
        assert vec1.ndim == vec2.ndim
        return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))
    
    
        
    def load_starmie_vectors(self, dl_table_vectors, query_table_vectors):

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
    
            
    def load_Tabbie_Table_vectors(self, dl_table_vectors, query_table_vectors):

        qfile = open(query_table_vectors,"rb")
            # queries is a list of tuples ; tuple of (str(filename), numpy.ndarray(vectors(numpy.ndarray) for columns) 
        queries_dict = pickle.load(qfile)
        # make as dictnary from first item to secon item 

        tfile = open(dl_table_vectors,"rb")
        dl_dict = pickle.load(tfile)
        # tables is a list of tuples ; tuple of (str(filename), numpy.ndarray(vectors(each verstor is a numpy.ndarray) for columns) 

        
        return (queries_dict,dl_dict)     
    
    
    def get_table_based_similarity(self, query_name, dl_table_name, all_vectors_tabbie):
        '''fro every column pairs from query and datalake table calculate the unionability and write in a file'''
        '''columns to be out: q_table, q_col, dl_table, dl_col, similarity'''
        # check whethter file exists load it other wise generate and write 
        
        if query_name.endswith('.csv'):
             query_name_= query_name[:-4]
        else: 
            query_name_=query_name
                
        if dl_table_name.endswith('.csv'):
             dl_table_name_= dl_table_name[:-4]  
        else: 
             dl_table_name_= dl_table_name       
        queries_dict = all_vectors_tabbie[0]
        dl_dict=all_vectors_tabbie[1]

        sim_data = pd.DataFrame(columns=["q_table","dl_table","table_similarity"])
        try:
            q_vector=queries_dict[query_name_]
            dl_t_vector=dl_dict[dl_table_name_]
            similarity_table = self._cosine_sim(q_vector,dl_t_vector)
        

            # Add a row with the current q_table, dl_table, and similarity_score
            sim_data =pd.DataFrame({"q_table": [query_name], "dl_table": [dl_table_name], "table_similarity": [similarity_table]})
        except Exception as e:
            RED   = "\033[31m"
            RESET = "\033[0m"
            print(f"{RED}could not find {query_name} or {dl_table_name}{RESET}")
            return   sim_data 
            
        return   sim_data   
        
    def get_column_based_similarity(self, query_name, dl_table_name, all_vectors):
        '''fro every column pairs from query and datalake table calculate the unionability and write in a file'''
        '''columns to be out: q_table, q_col, dl_table, dl_col, similarity'''
        # check whethter file exists load it other wise generate and write 

        queries_dict = all_vectors[0]
        dl_dict=all_vectors[1]

        sim_data = pd.DataFrame(columns=["q_table", "q_col", "dl_table","dl_col","similarity"])
          
        
        q_vectors=queries_dict[query_name]
        query_rows = self.alignment_data[self.alignment_data['query_table_name'] == query_name]

        # Filter rows for the current query_name and dl_table_name
        specific_rows = query_rows[query_rows['dl_table_name'] == dl_table_name]
        dl_t_vectors=dl_dict[dl_table_name]
        # Retrieve the relevant columns

        for _, row in specific_rows.iterrows():
            # get their vectors 
            query_column = row['query_column#']
            dl_column = row['dl_column']
            # Call the similarity function
            similarity_col = self._cosine_sim(q_vectors[query_column],dl_t_vectors[dl_column])
    

            # Add a row with the current q_table, dl_table, and similarity_score
            sim_data = pd.concat([
                sim_data,
                pd.DataFrame({"q_table": [query_name], "dl_table": [dl_table_name],"q_col": [query_column] , "dl_col": [dl_column] ,"similarity": [similarity_col]})
            ], ignore_index=True) 
  
        return   sim_data       
        
        
            
    def get_column_based_lexical_distance(self,query_name, dl):
            '''fro every column pairs from query and datalake table calculate the lexical distance and write in a file'''
            '''columns to be out: q_table, q_col, dl_table, dl_col, lexical_distance'''


            lexdis_data = pd.DataFrame(columns=["q_table", "q_col", "dl_table","dl_col","lex_distance"])

            DSize=self.dsize
        
            query_rows = self.alignment_data[self.alignment_data['query_table_name'] == query_name]
                        # Iterate over each dl_table_name
            dl_rows= query_rows[(query_rows['query_table_name'] == query_name) & (query_rows['dl_table_name'] == dl)]
            distance=0
            for _, row in dl_rows.iterrows():
                                dl_column_number = int(row['dl_column'])
                                q_column_number = int(row['query_column#'])
                                
                                # compute the diversity  for columns
                                
                                #get the comlumn from data lake table
                                dl_column_dl=self.table_raw_lol_proccessed.get(dl)[dl_column_number]
                                dl_column_set_dl=self.table_raw_proccessed_los.get(dl)[dl_column_number]
                                
                                #get the comlumn from data lake table
                                q_column_q=self.table_raw_lol_proccessed.get(query_name)[q_column_number]
                                q_column_set=self.table_raw_proccessed_los.get(query_name)[q_column_number]

                            
                                #see what is the number of unique values in the query+ dl columns that are list of list 
                                # we have a threshold to determine the smallness of domain called DS(domain size)
                                # Besat to change: here we do not merge tokens from all cell to 
                                # gether for each column maybe this will change later ?
                                domain_estimate=set.union(set(q_column_q),set(dl_column_dl) )
                                if(len(domain_estimate)<DSize):
                                    distance=self.Jensen_Shannon_distances(q_column_q,dl_column_dl,domain_estimate)
                                    # log the domian infomrmation
                                else: 
                                    # jaccard distance
                                    distance=1-self._lexicalsim_Pair(dl_column_set_dl,q_column_set)
                              
                                            
                                                                
  
                                new_row = {
                                    "q_table":query_name ,
                                    "q_col": q_column_number,
                                    "dl_table": dl,
                                    "dl_col": dl_column_number, 
                                    "lex_distance":distance
                                }


                            # Convert the new row to a DataFrame
                                new_row_df = pd.DataFrame([new_row])

                            # Concatenate the new row with the existing DataFrame
                                lexdis_data = pd.concat([lexdis_data, new_row_df], ignore_index=True)
                                                    # Append the new row to the DataFrame
                        
            return lexdis_data
      
      
           
    def load_unionable_tables(self, path):
        #load the mapping between query and its unionnable tables generated by a system like Starmie
         print("loading the first round ranked resus produced by starmie")
         self.unionable_tables= utl.loadDictionaryFromPickleFile(path) 
         
    def load_column_alignment_data(self, alignment_Dust):
        
        """
        Load data from the specified source.

        The schema for the data is expected as:
        ['query_table_name', 'query_column', 'query_column#', 
        'dl_table_name', 'dl_column#', 'dl_column']

        :return: None
            """
        print("trying to load alignemnt produced buy DUST")
        try:
            # Load the CSV file into a pandas DataFrame
            self.alignment_data = pd.read_csv(alignment_Dust)

            # Verify that the required columns are present
            required_columns = ['query_table_name', 'query_column', 'query_column#',
                                'dl_table_name', 'dl_column#', 'dl_column']
            if not all(column in self.alignment_data.columns for column in required_columns):
                missing_columns = [col for col in required_columns if col not in self.alignment_data.columns]
                raise ValueError(f"Missing required columns in data: {missing_columns}")

            print("Data loaded successfully")
        
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_source}")
        
        except ValueError as e:
            print(f"Error: {e}")
        
        except Exception as e:
            print(f"An unexpected error occurred: {e}")     
            
   
    
    
    def perform_search_optimized(self, p_degree, k, all_vectors, all_table_vectors_Tabbie):
        # Load the required data

      # we are going to first compute the unionability score column wised and aggragate them
      # but this time we are penalizing the aggregated  unionablity score by thae same formula as before but minusing the cosine similarity between the table embedings 
        all_ranked_result = {}

                  
        q_table_names = self.alignment_data['query_table_name'].unique()
            # for every query now compute the similarity scores with dt tables 
        for query_name in q_table_names:
                   
                    start_time = time.time_ns()
                    # get q columns vectors 
                    # Get all rows corresponding to the current query_name
                    query_rows = self.alignment_data[self.alignment_data['query_table_name'] == query_name]
                    # Get all unique dl_table_names for the current query_name
                    dl_table_names = query_rows['dl_table_name'].unique()
                        # Iterate over each dl_table_name
                    # 2) Prepare empty container
                    mergeddata = pd.DataFrame()
                    for dl_table_name in dl_table_names:
                            sim_data_columnwise = self.get_column_based_similarity(query_name,dl_table_name , all_vectors)
                            
                            # aggregate
                            data_tableunionability = (sim_data_columnwise.groupby(["q_table", "dl_table"], as_index=False)["similarity"].sum().rename(columns={"similarity": "table_uninability"}))
                            data_table_sim_tabbie = self.get_table_based_similarity(query_name,dl_table_name , all_table_vectors_Tabbie)
                            
                            
                            
                                                        # b) merge on the two key columns
                            tmp = pd.merge(
                                data_tableunionability,
                                data_table_sim_tabbie,
                                on=["q_table", "dl_table"],
                                how="inner"
                            )

                            # c) compute (table_uninability - 1)**p_degree * table_uninability
                            tmp["penalized_unionability_score"] = (
                                (1- tmp["table_similarity"]) ** p_degree
                                * tmp["table_uninability"]
                            )

                            # d) append to your rolling result
                            mergeddata = pd.concat([mergeddata, tmp], ignore_index=True)
                            
                

                             
                                       
                    # Filter scores for the current query and sort by score
                    top_k_result = (
                        mergeddata[mergeddata['q_table'] == query_name]
                        .sort_values(by="penalized_unionability_score", ascending=False)
                        .head(k)
                    )

            # Store the results
                    all_ranked_result[(query_name, k, p_degree)] = (
                        list(top_k_result[["dl_table",'penalized_unionability_score']].to_records(index=False)),
                        round((time.time_ns() - start_time) / 10 ** 9, 2)
                    )

        return all_ranked_result

          
if __name__ == "__main__":
    # Example usage:
    # alignment_Dust="/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/tus_CL_KMEANS_cosine_alignment_all.csv"
    # first_50_starmie="/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/diveristy_data/search_results/Starmie/top_20_Starmie_output_04diluted_restricted_noscore.pkl"    
    # search_results_file="/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/diveristy_data/search_results/Penalized/search_result_new_penalize_04diluted_restricted_pdeg1.csv"
    # # alignment_Dust="/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/tus_CL_KMEANS_cosine_alignment_all.csv"
    # first_50_starmie="/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/diveristy_data/search_results/Starmie/top_20_Starmie_output_04diluted_restricted_noscore.pkl"    
    # search_results_file="/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/diveristy_data/search_results/Penalized/search_result_new_penalize_04diluted_restricted_pdeg1.csv"
    
    
    # alignment_Dust="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/ugenv2_small_manual_alignment_all.csv"
    # first_50_starmie="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/diveristy_data/search_results/Starmie/top_20_Starmie_output_04diluted_restricted_noscore.pkl"    
    # search_results_file="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/diveristy_data/search_results/Penalized/search_result_penalize_04diluted_restricted_pdeg1.csv"

    # '''load starmie vectors for query and data lake and retrun as dictionaries'''
    # dl_table_vectors = "/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl"
    # query_table_vectors = "/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/vectors/cl_query_drop_col_tfidf_entity_column_0.pkl"
    # #dsize=20
    # #only for ugenv2-small that hae few ros we set domnain size 2
    #dsize=2
    
    #santos
    # dl_table_vectors = "/u6/bkassaie/NAUS/data/santos/vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl"
    # query_table_vectors = "/u6/bkassaie/NAUS/data/santos/vectors/cl_query_drop_col_tfidf_entity_column_0.pkl"
    # dl_table_vectors_tabbie="data/santos/TABBIE_vectors/datalake/embeddings.pkl"
    # q_table_vectors_tabbie="data/santos/TABBIE_vectors/query/embeddings.pkl"
    # alignment_Dust="data/santos/Santos_dlt_CL_KMEANS_cosine_alignment.csv"
    # first_50_starmie="data/santos/diveristy_data/search_results/Starmie/top_20_Starmie_output_04diluted_restricted_noscore.pkl"
    # semantic_novlety_search = Semantic_Novelty(dl_table_vectors)
    # semantic_novlety_search.load_column_alignment_data(alignment_Dust)
    # semantic_novlety_search.load_unionable_tables(first_50_starmie)   
    # all_vectors=semantic_novlety_search.load_starmie_vectors(dl_table_vectors, query_table_vectors)
    # all_table_vectors_Tabbie=semantic_novlety_search.load_Tabbie_Table_vectors(dl_table_vectors_tabbie,q_table_vectors_tabbie)
    # search_results_file="data/santos/diveristy_data/search_results/semanticNovelty/search_result_semNovelty_04diluted_restricted_pdeg1.csv"

    #tus
    # bpath="data/table-union-search-benchmark/small/"
    # dl_table_vectors = f"{bpath}vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl"
    # query_table_vectors = f"{bpath}vectors/cl_query_drop_col_tfidf_entity_column_0.pkl"
    # dl_table_vectors_tabbie=f"{bpath}TABBIE_vectors/datalake/embeddings.pkl"
    # q_table_vectors_tabbie=f"{bpath}TABBIE_vectors/query/embeddings.pkl"
    # alignment_Dust=f"{bpath}tus_CL_KMEANS_cosine_alignment_all.csv"
    # first_50_starmie=f"{bpath}diveristy_data/search_results/Starmie/top_20_Starmie_output_04diluted_restricted_noscore.pkl"
    # semantic_novlety_search = Semantic_Novelty(dl_table_vectors)
    # semantic_novlety_search.load_column_alignment_data(alignment_Dust)
    # semantic_novlety_search.load_unionable_tables(first_50_starmie)   
    # all_vectors=semantic_novlety_search.load_starmie_vectors(dl_table_vectors, query_table_vectors)
    # all_table_vectors_Tabbie=semantic_novlety_search.load_Tabbie_Table_vectors(dl_table_vectors_tabbie,q_table_vectors_tabbie)
    # search_results_file=f"{bpath}diveristy_data/search_results/semanticNovelty/search_result_semNovelty_04diluted_restricted_pdeg1.csv"



    #ugen_v2
    # bpath="data/ugen_v2/"
    # dl_table_vectors = f"{bpath}vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl"
    # query_table_vectors = f"{bpath}vectors/cl_query_drop_col_tfidf_entity_column_0.pkl"
    # dl_table_vectors_tabbie=f"{bpath}TABBIE_vectors/datalake/embeddings.pkl"
    # q_table_vectors_tabbie=f"{bpath}TABBIE_vectors/query/embeddings.pkl"
    # alignment_Dust=f"{bpath}ugenv2_CL_KMEANS_cosine_alignment_diluted.csv"
    # first_50_starmie=f"{bpath}diveristy_data/search_results/Starmie/top_20_Starmie_output_04diluted_restricted_noscore.pkl"
    # semantic_novlety_search = Semantic_Novelty(dl_table_vectors)
    # semantic_novlety_search.load_column_alignment_data(alignment_Dust)
    # semantic_novlety_search.load_unionable_tables(first_50_starmie)   
    # all_vectors=semantic_novlety_search.load_starmie_vectors(dl_table_vectors, query_table_vectors)
    # all_table_vectors_Tabbie=semantic_novlety_search.load_Tabbie_Table_vectors(dl_table_vectors_tabbie,q_table_vectors_tabbie)
    # search_results_file=f"{bpath}diveristy_data/search_results/semanticNovelty/search_result_semNovelty_04diluted_restricted_pdeg1.csv"

   #ugen_v2_small
    bpath="data/ugen_v2/ugenv2_small/"
    dl_table_vectors = f"{bpath}vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl"
    query_table_vectors = f"{bpath}vectors/cl_query_drop_col_tfidf_entity_column_0.pkl"
    dl_table_vectors_tabbie=f"{bpath}TABBIE_vectors/datalake/embeddings.pkl"
    q_table_vectors_tabbie=f"{bpath}TABBIE_vectors/query/embeddings.pkl"
    alignment_Dust=f"{bpath}ugenv2_small_manual_alignment_all.csv"
    first_50_starmie=f"{bpath}diveristy_data/search_results/Starmie/top_20_Starmie_output_04diluted_restricted_noscore.pkl"
    semantic_novlety_search = Semantic_Novelty(dl_table_vectors)
    semantic_novlety_search.load_column_alignment_data(alignment_Dust)
    semantic_novlety_search.load_unionable_tables(first_50_starmie)   
    all_vectors=semantic_novlety_search.load_starmie_vectors(dl_table_vectors, query_table_vectors)
    all_table_vectors_Tabbie=semantic_novlety_search.load_Tabbie_Table_vectors(dl_table_vectors_tabbie,q_table_vectors_tabbie)
    search_results_file=f"{bpath}diveristy_data/search_results/semanticNovelty/search_result_semNovelty_04diluted_restricted_pdeg1.csv"



    for i in range(2,11):   

        k=i     
        p_degree=1          
        relsutls=semantic_novlety_search.perform_search_optimized(p_degree,k, all_vectors,all_table_vectors_Tabbie )
        result_dic={}

        if os.path.exists(search_results_file):
                with open(search_results_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    # Write the data
                    for key_, (result, secs) in relsutls.items():
                        # Join the list of results into a string, if needed
                        result=[r[0] for r in result ]
                        result_str = ', '.join(result) if isinstance(result, list) else str(result)
                        writer.writerow([key_[0], result_str,secs,key_[1],key_[2] ])
                        result_dic[key_[0]]=(result,secs)
        else: 
                print("result file does not exist. Writing results to file.")

                with open(search_results_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['query_name', 'tables','semanticnovelty_execution_time', 'k', 'pdegree'])

                    # Write the data
                    for key_, (result, secs) in relsutls.items():
                        # Join the list of results into a string, if needed
                        result=[r[0] for r in result ]
                        result_str = ', '.join(result) if isinstance(result, list) else str(result)
                        writer.writerow([key_[0], result_str,secs,key_[1],key_[2] ])
                        result_dic[key_[0]]=(result,secs)
       