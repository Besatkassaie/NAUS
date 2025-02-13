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



class Penalized_Search:
    """
    Pe: Penalized_Search is a class for performing penalized re_ranking of unionable tables for Novelty based  unionable  table search.
    """

    def __init__(self):
        self.column_based_lexical_distance_file_=None
        self.column_based_similarity_file_=None
        self.alignment_data=None
        self.unionable_tables=None
        print("search object is created....")
   
    def _cosine_sim(self, vec1, vec2):
        ''' Get the cosine similarity of two input vectors: vec1 and vec2
        '''
        assert vec1.ndim == vec2.ndim
        return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))
    
    
        
    def load_starmie_vectors(self):
        '''load starmie vectors for query and data lake and retrun as dictionaries'''
        dl_table_vectors = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl"
        query_table_vectors = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/vectors/cl_query_drop_col_tfidf_entity_column_0.pkl"
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
        
    def load_column_based_similarity(self,column_based_similarity_file):
        '''fro every column pairs from query and datalake table calculate the unionability and write in a file'''
        '''columns to be out: q_table, q_col, dl_table, dl_col, similarity'''
        # check whethter file exists load it other wise generate and write 

        if os.path.exists(column_based_similarity_file):
        # Load the file into a DataFrame
            sim_data = pd.read_csv(column_based_similarity_file, header=0)  # Treat the first row as column names
        else:
            all_vectors=self.load_starmie_vectors()
            queries_dict = all_vectors[0]
            dl_dict=all_vectors[1]

            sim_data = pd.DataFrame(columns=["q_table", "q_col", "dl_table","dl_col","similarity"])
            # self.alignment_data is a datafram with columns ['query_table_name', 'query_column', 'query_column#','dl_table_name', 'dl_column#', 'dl_column']
          
            q_table_names = self.alignment_data['query_table_name'].unique()
            # for every query now compute the similarity scores with dt tables 
            for query_name in q_table_names:
                    # get q columns vectors 
                    q_vectors=queries_dict[query_name]
                    # Get all rows corresponding to the current query_name
                    query_rows = self.alignment_data[self.alignment_data['query_table_name'] == query_name]

                    # Get all unique dl_table_names for the current query_name
                    dl_table_names = query_rows['dl_table_name'].unique()
                        # Iterate over each dl_table_name
                    for dl_table_name in dl_table_names:
                        similarity_score=0
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
            sim_data.to_csv(column_based_similarity_file, index=False)        
        
        return   sim_data       
        
        
        print("added here")
            
    def load_column_based_lexical_distance(self,column_based_lexical_distance_file ):
        '''fro every column pairs from query and datalake table calculate the lexical distance and write in a file'''
        '''columns to be out: q_table, q_col, dl_table, dl_col, lexical_distance'''
        if os.path.exists(column_based_lexical_distance_file):
        # Load the file into a DataFrame
            print("loading column based lexiacal distance file")
            lexdis_data = pd.read_csv(column_based_lexical_distance_file, header=0)  # Treat the first row as column names
            return lexdis_data
        else:
            print("creating column based lexiacal distance file")

            lexdis_data = pd.DataFrame(columns=["q_table", "q_col", "dl_table","dl_col","lex_distance"])

            dataFolder="santos"
            table_path = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl"
            query_path_raw = "data/"+dataFolder+"/"+"query"
            table_path_raw = "data/"+dataFolder+"/"+"datalake"
            processed_path="data/processed/"+dataFolder+"/"
            index_file_path="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/indices/Joise_Index_DL_santos_tokenized_bot.pkl"
            lex_data = pd.DataFrame(columns=["q_table", "q_col", "dl_table","dl_col","lexical_distance"])

            
            text_processor = TextProcessor()

            # we preprocess the values in tables both query and data lake tables
            list_of_lists=[]
            
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
                print("building Joise Index......")

                for key, value in table_raw_proccessed_los.items():
                    
                    index = SearchIndex(value, similarity_func_name="jaccard", similarity_threshold=0.0)
                    table_raw_index[key]= index   
                    
                # write in a pickle file  
                with open(index_file_path, 'wb') as file:
                        pickle.dump(table_raw_index, file)   
            
            
            self.table_raw_index=table_raw_index
            self.table_path=table_path
            #DSize is a hyper parameter
            DSize=20
        
               
            q_table_names = self.alignment_data['query_table_name'].unique()            # for every query now compute the similarity scores with dt tables 
            for query_name in q_table_names:
                    query_rows = self.alignment_data[self.alignment_data['query_table_name'] == query_name]

                    # Get all unique s_i_names for the current query_name
                    dl_names = query_rows['dl_table_name'].unique()
                        # Iterate over each dl_table_name
                    for dl in dl_names:
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
                              
                                            
                                                                
                            # now compute the average  lexdis_data = pd.DataFrame(columns=["q_table", "q_col", "dl_table","dl_col","lex_distance"])
  
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
                        
            lexdis_data.to_csv(column_based_lexical_distance_file, index=False)        

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
            
    def perform_search(self, p_degree, k):
        lexdis=self.load_column_based_lexical_distance(self.column_based_lexical_distance_file_)
        sim_data=self.load_column_based_similarity(self.column_based_similarity_file_)
        all_ranked_result = {}
        

        # compute  penalized unionability score for query, table pairs  based on column data
        #lexdis_data = pd.DataFrame(columns=["q_table", "q_col", "dl_table","dl_col","lex_distance"])
        # Select the 'q_table' and 'dl_table' columns and drop duplicate pairs
       

        unique_queries = lexdis['q_table'].unique()

        # Reset the index for a cleaner output
        for q in unique_queries:
            _result={}
            
            by_query=lexdis[(lexdis['q_table'] == q)]
            dl_tables = by_query['dl_table'].unique()
            start_time = time.time_ns()
            for dl_table in dl_tables:
                penalized_unionability_score=0
                # Retrieve rows corresponding to the unique pair from lexdis
                lexdis_rows = by_query[(by_query['q_table'] == q) & (by_query['dl_table'] == dl_table)]
                
                # Iterate through these rows to get 'q_col' and 'dl_col'
                for _, lexdis_row in lexdis_rows.iterrows():
                    q_col = lexdis_row['q_col']
                    dl_col = lexdis_row['dl_col']
                    lex_dis=float(lexdis_row['lex_distance'])
                    # Retrieve the corresponding row from sim_data
                    sim_row = sim_data[
                        (sim_data['q_table'] == q) &
                        (sim_data['dl_table'] == dl_table) &
                        (sim_data['q_col'] == q_col) &
                        (sim_data['dl_col'] == dl_col)
                    ]

                    # Ensure there is a match in sim_data
                    if not sim_row.empty:
                        sim_value = float(sim_row.iloc[0]['similarity'])  # Get the first matching row
                        # Call the function 'g' with the relevant values                   
                    else:
                        print(f"No matching row in sim_data for q_table={q}, dl_table={dl_table}, q_col={q_col}, dl_col={dl_col}")
                    
                    penval_col=((lex_dis)**p_degree)*sim_value
                    penalized_unionability_score=penalized_unionability_score+penval_col
                _result[dl_table]= penalized_unionability_score   
                
        # get first k
        
        # Sort the dictionary by value in descending order
            sorted_result = sorted(_result.items(), key=lambda item: item[1], reverse=True)

            # Get the first k items as a dictionary
            # Replace this with your desired value of k
            top_k_result = (sorted_result[:k])
                
            end_time = time.time_ns()    
            total_time = round(int(end_time - start_time) / 10 ** 9, 2)
            print("Total time taken: ", total_time, " seconds.")
            
            
            all_ranked_result[(q,k, p_degree)]=(top_k_result, total_time)
                
        return all_ranked_result
    
    
    def perform_search_optimized(self, p_degree, k):
        # Load the required data
        lexdis = self.load_column_based_lexical_distance(self.column_based_lexical_distance_file_)
        sim_data = self.load_column_based_similarity(self.column_based_similarity_file_)

        all_ranked_result = {}

        # Merge lexdis and sim_data for efficient computation
        merged_data = pd.merge(
            lexdis,
            sim_data,
            on=["q_table", "dl_table", "q_col", "dl_col"],
            how="inner",
            suffixes=("_lex", "_sim")
        )

        # Calculate penalized unionability scores for each row
        merged_data['penalized_unionability_score'] = (
            (merged_data['lex_distance'] ** p_degree) * merged_data['similarity']
        )

        # Group by query table and data lake table to aggregate scores
        grouped_scores = merged_data.groupby(["q_table", "dl_table"])['penalized_unionability_score'].sum().reset_index()

        # Get unique queries
        unique_queries = grouped_scores['q_table'].unique()

        for q in unique_queries:
            start_time = time.time_ns()

            # Filter scores for the current query and sort by score
            top_k_result = (
                grouped_scores[grouped_scores['q_table'] == q]
                .sort_values(by="penalized_unionability_score", ascending=False)
                .head(k)
            )

            # Store the results
            all_ranked_result[(q, k, p_degree)] = (
                list(top_k_result[["dl_table",'penalized_unionability_score']].to_records(index=False)),
                round((time.time_ns() - start_time) / 10 ** 9, 2)
            )

        return all_ranked_result

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
    def item_frequency(self, lst):
            return dict(Counter(lst))
        
   
    def  Jensen_Shannon_distances(self,query_column,dl_column,domain_estimate):
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
            

          
if __name__ == "__main__":
    # Example usage:
    alignment_Dust="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/DUST_Alignment_Diluted04_restricted.csv"
    first_50_starmie="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/top_20_Starmie_output_04diluted_restricted_noscore.pkl"    
    search_results_file="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/search_result_penalize_04diluted_restricted_pdeg1.csv"

    
    
    for i in range(2,11):   
        penalize_search = Penalized_Search()
        penalize_search.column_based_lexical_distance_file_="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/column_based_lexical_distance_restricted_dilut04.csv"
        penalize_search.column_based_similarity_file_="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/column_based_similarity_restricted_dilut04.csv"
        penalize_search.load_column_alignment_data(alignment_Dust)
        penalize_search.load_unionable_tables(first_50_starmie)   
        k=i     
        p_degree=1          
        relsutls=penalize_search.perform_search_optimized(p_degree,k)
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
                    writer.writerow(['query_name', 'tables','penalized_execution_time', 'k', 'pdegree'])

                    # Write the data
                    for key_, (result, secs) in relsutls.items():
                        # Join the list of results into a string, if needed
                        result=[r[0] for r in result ]
                        result_str = ', '.join(result) if isinstance(result, list) else str(result)
                        writer.writerow([key_[0], result_str,secs,key_[1],key_[2] ])
                        result_dic[key_[0]]=(result,secs)
       