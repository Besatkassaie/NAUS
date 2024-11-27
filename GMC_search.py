import pandas as pd
from naive_search_Novelty import NaiveSearcherNovelty
import test_naive_search_Novelty
from SetSimilaritySearch import SearchIndex
from  process_column import TextProcessor
import pickle
import os


import time
class GMC_Search:
    """
    GMC_Search: A class for performing Greedy  Marginal  Contribution (GMC) for Novelty based  unionable  table search.
    """

    def __init__(self, data_source, search_parameters=None):
        """
        Initializes the GMC_Search instance.

        :param data_source: The source of the data (e.g., file path, database, etc.).
        :param search_parameters: Dictionary containing parameters for the search (optional).
        """
        self.data_source = data_source
        self.search_parameters = search_parameters or {}
        self.results = []
        self.data = None
        self.unionability = None
        self.diversity = None

    def load_data(self):
        """
        Load data from the specified source.
        Assume that we have exported a file from Starmie havinf the following Columns; 
        ['DL_table', 'DL_column',  'Q_table', 'Q_column','entropy','dSize','distance','Lexsim', 'SemSim', 'final_value']


        :return: None
        """
        try:
            # Load the CSV file into a pandas DataFrame
            self.data = pd.read_csv(self.data_source)

            # Verify that the required columns are present
            required_columns = ['DL_table', 'DL_column', 'Q_table', 'Q_column', 
                                'entropy', 'dSize', 'distance', 'Lexsim', 'SemSim', 'final_value']
            if not all(column in self.data.columns for column in required_columns):
                missing_columns = [col for col in required_columns if col not in self.data.columns]
                raise ValueError(f"Missing required columns in data: {missing_columns}")

            print("Data loaded successfully.")
        
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_source}")
        
        except ValueError as e:
            print(f"Error: {e}")
        
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        

    def preprocess_data(self):
        """
        Since the input file has column name repeated throught the file we first remove them
        Preprocess the loaded data to make it suitable for search operations.

        :return: None
        """
        """
        Preprocess the loaded data to make it suitable for search operations.
        Removes rows where the values match the column names. 

        :return: None
        """
        if self.data is not None:
            # Get the list of column names
            column_names = self.data.columns.tolist()

            # Remove rows where all values in a row match the column names
            self.data = self.data[~self.data.apply(lambda row: row.tolist() == column_names, axis=1)]

            print("Preprocessing complete. Data cleaned.")
        else:
            print("No data loaded. Please load the data before preprocessing.")
            
    
    

    def execute_search(self):
        """
        Perform the GMC search based on the parameters.

        :return: List of search results
        """
        # TODO: Implement the search algorithm
        pass

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
    def mmc_compute_div_sum(self, s_i: str, div_dict: dict, R_p: set) -> float:
        total_score = 0
        for s_j in R_p:
            if (s_i , s_j) in div_dict:
                total_score += div_dict[(s_i, s_j)]
            else:
                total_score += div_dict[(s_j, s_i)]
        return total_score

    def mmc_compute_div_large(self, s_i: str, div_dict: dict, remaining_s_set : set, max_l : int) -> float: #max_l should be used as: < max_l ; not <= max_l
        div_l_list = [] # we will use the first l values after sorting this.
        for s_j in remaining_s_set: # the items that are not inserted in R_p yet
            if (s_i, s_j) in div_dict:
                div_l_list.append(div_dict[(s_i, s_j)])
            else:
                div_l_list.append(div_dict[(s_j, s_i)])
        # print("Div l: ", div_l_list)
        # print("MAX L", max_l - 1)
        div_l_list = sorted(div_l_list, reverse=True)[:max_l - 1]
        return div_l_list
    def mmc(self, s_set: set, lmda : float, k: int, sim_dict: dict, div_dict : dict, R_p: set) -> dict: # s_dict contains query id as key and its embeddings as values.
        all_mmc = dict()
        # print("R_P:", R_p)
        p = len(R_p) - 1
        div_coefficient = lmda / (k - 1)
        for s_i in s_set:
            sim_term = (1 - lmda) * sim_dict[s_i]
            div_term1 = div_coefficient * self.mmc_compute_div_sum(s_i, div_dict, R_p)
            div_term2 = div_coefficient * sum(self.mmc_compute_div_large(s_i, div_dict, s_set - R_p - {s_i}, k - p))
            current_mmc = sim_term + div_term1 + div_term2
            all_mmc[s_i] = current_mmc
        # print("current mmc:", current_mmc)
        # print("all_mmc:", all_mmc)
        return all_mmc
    
    
    # def d_sim(self,s_dict : dict, q_embedding: np.ndarray, metric = "cosine", normalize = False) -> dict:
    def extract_d_sim(self, query_name: str) -> dict:
        """
        Extract the similarity dictionary (sim_dict) based on the unionability DataFrame for a given query name.

        :param query_name: Name of the query (Q_table) to filter the unionability DataFrame.
        :return: A dictionary with DL_table as the key and unionability score as the value.
        """
        if self.unionability is not None:
            # Filter the unionability DataFrame for the given query name
            filtered_df = self.unionability[self.unionability['Q_table'] == query_name]
            
            # Convert the filtered DataFrame to a dictionary with DL_table as the key and unionability as the value
            sim_dict = dict(zip(filtered_df['DL_table'], filtered_df['unionability']))
            
            print(f"Similarity dictionary created for query: {query_name}")
            return sim_dict
        else:
            print("Unionability data is not available. Please calculate unionability first.")
            return {}

    def extract_d_div(self, metric = "cosine", normalize = False) -> dict:
        div_dict = dict() # key: s_dict key i.e. s_id; value : similarity score
        # for current_s1 in s_dict:
        #     for current_s2 in s_dict:
        #         if metric == "l1":
        #             max_possible_l1 = 2 * len(s_dict[current_s1])
        #             if normalize == True:
        #                 current_div = 1 - (np.linalg.norm(s_dict[current_s1] - s_dict[current_s2], ord = 1) / max_possible_l1)
        #             else:
        #                 current_div = max_possible_l1 - (np.linalg.norm(s_dict[current_s1] - s_dict[current_s2], ord = 1) / 1)                
        #         elif metric == "l2":
        #             max_possible_l2 = np.sqrt(2 * len(s_dict[current_s1]))
        #             if normalize == True:
        #                 current_div = 1 - (np.linalg.norm(s_dict[current_s1] - s_dict[current_s2], ord = 2) / max_possible_l2)
        #             else:
        #                 current_div = max_possible_l2 - (np.linalg.norm(s_dict[current_s1] - s_dict[current_s2], ord = 2) / 1)
        #         else: #cosine
        #             if normalize == True:
        #                 current_div = (utl.CosineSimilarity(s_dict[current_s1], s_dict[current_s2]) + 1) / 2 #normalized score between 0 and 1
        #             else:
        #                 current_div = utl.CosineSimilarity(s_dict[current_s1], s_dict[current_s2])
        #         div_dict[(current_s1, current_s2)] = current_div
        return div_dict

    # def gmc(self, S_dict: dict, q_dict:dict, k: int, lmda: float = 0.7, metric = "cosine", print_results = False, normalize = False, max_metric = True, compute_metric = True) -> set: #S_dict is a dictionary with tuple id as key and its embeddings as value. 
    def gmc(self, S_names: set, query_name:str, k: int, lmda: float = 0.7, metric = "cosine", print_results = False, normalize = False, max_metric = True, compute_metric = True) -> set: #S_dict is a dictionary with tuple id as key and its embeddings as value. 

        '''adopted from https://anonymous.4open.science/r/dust-B79B/diversity_algorithms/div_utilities.py'''
    #the metric is for sim dict and div dict, and is independent of evaluation. we evaluate using all three metrics and in compute_metric() function, we again compute sim_dict and div_dict.
        start_time = time.time_ns()
        # q = np.mean(list(q_dict.values()), axis=0)
        R = set()
        ranked_div_result = []
        sim_dict = self.extract_d_sim( query_name, metric = metric, normalize=normalize) # index the similarity between each item in S and q for once so that we can re-use them.
        div_dict = self.extract_d_div( metric = metric, normalize=normalize) # index the diversity between each pair of items in S so that we can re-use them.
        # debug_dict(sim_dict, 5, "sim dict")
        # debug_dict(div_dict, 5, "div dict")
        S_set = S_names
        for p in range(0, k):
            if len(S_set) == 0:
                break
            mmc_dict = self.mmc(S_set, lmda, k, sim_dict, div_dict, R) # send S to mmc and compute MMC for each si
            s_i  = max(mmc_dict, key=lambda k: mmc_dict[k])
            R.add(s_i)
            ranked_div_result.append(s_i)
            S_set = S_set - {s_i}
        # print("GMC f score:", f_prime(R, lmda, k, sim_dict, div_dict))
        end_time = time.time_ns()
        total_time = round(int(end_time - start_time) / 10 ** 9, 2)
        print("Total time taken: ", total_time, " seconds.")
        # if compute_metric == True:
        #     computed_metrics, embedding_plot = compute_metrics(R, S_dict, q_dict, lmda, k, print_results = print_results, normalize=normalize, metric= metric, max_metric= max_metric)
        #     for each in computed_metrics:
        #         each['time_taken'] = total_time
        # else:
        #     computed_metrics = [{"metric": "n/a", "with_query" : "n/a", "max_score": np.nan, "max-min_score": np.nan, "avg_score": np.nan, 'time_taken' : total_time}]
        #     embedding_plot = ""
        return ranked_div_result#, computed_metrics, embedding_plot
        
    def calculate_unionability(self):
        """
        Calculate the unionability for each DL_table and Q_table pair by summing up the SemSim values.

        :return: A pandas DataFrame with the unionability scores.
        """
        if self.data is not None:
            # Group by DL_table and Q_table and sum the SemSim values
            self.unionability = (
                self.data.groupby(['DL_table', 'Q_table'])['SemSim']
                .sum()
                .reset_index()
                .rename(columns={'SemSim': 'unionability'})
            )
            print("Unionability calculation complete.")
            return self.unionability
        else:
            print("No data loaded. Please load and preprocess the data first.")
            return None

    def calculate_diversity(self):
        
        """
    Calculate the diversity by grouping self.data by Q_table and for each DL_table,
    return a set of DL_columns. The result is a list of triples (Q_table, DL_table, set of DL_columns).

    :return: A list of triples (Q_table, DL_table, set of DL_columns).
    """
        #load required data here 
        dataFolder="santos"
        table_path = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl"
        query_path_raw = "data/"+dataFolder+"/"+"query"
        table_path_raw = "data/"+dataFolder+"/"+"datalake"
        index_path="data/indices/"
        processed_path="data/processed/"+dataFolder+"/"
        index_file_path="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/indices/Joise_Index_DL_santos_tokenized_bot.pkl"
        
        text_processor = TextProcessor()

        
     
        #I assume that the order  of columns are comming from the original csv files 
    
    
        # we preprocess the values in tables both query and data lake tables
        list_of_lists=[]
        
        tables_raw=NaiveSearcherNovelty.read_csv_files_to_dict(table_path_raw)
        
        #the normalized/tokenized/original with no duplicates  dl(data lake) tables are stored in table_raw_proccessed_los
        table_raw_proccessed_los={}
        
            # write the proccessed result having columns as set to a pickle file 
        dl_tbls_processed_set_file_name="dl_tbls_processed_set.pkl"
        
        table_raw_proccessed_los=test_naive_search_Novelty.getProcessedTables(text_processor, dl_tbls_processed_set_file_name, processed_path, tables_raw,"los", 1, 1)
        # process dl tables and save as list of lists 
        self.table_raw_proccessed_los=table_raw_proccessed_los
        table_raw_lol_proccessed={}
            # write the proccessed result having columns as set to a pickle file 
        dl_tbls_processed_lol_file_name="dl_tbls_processed_lol.pkl"
        self.dl_tbls_processed_lol_file_name=dl_tbls_processed_lol_file_name
        table_raw_lol_proccessed=test_naive_search_Novelty.getProcessedTables(text_processor,  dl_tbls_processed_lol_file_name,processed_path, tables_raw,"lol", 1, 1)
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
        
        if self.data is not None:
            try:
                # Initialize an empty list to store the results
                diversity_pairs = []


                # Group the data by Q_table
                grouped = self.data.groupby('Q_table')

                # Iterate over each group
                for q_table, group in grouped:
                    # Further group by DL_table within each Q_table group
                    dl_grouped = group.groupby('DL_table')
                    
                    for dl_table, dl_group in dl_grouped:
                        # Get the set of DL_columns for this Q_table and DL_table
                        dl_columns_set = set(dl_group['DL_column'])

                        # Append the triple to the results
                        diversity_pairs.append((q_table, dl_table, dl_columns_set))
                
                print("Diversity pairing complete.")

            except KeyError as e:
                print(f"Error: Missing required column in data - {e}")
                return None
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return None
           
           
           
           
           
        
            # in the following "q_main_table", "s_i" is comming from unionability calculation and we
            # load the corresponidng s_i, s_j for the alignment determined by unionability
            diversity_scores_df = pd.DataFrame(columns=["q_main_table", "s_i", "s_j", "diversity_score"])
            file_path='diversity_scores.csv'
            if os.path.exists(file_path):
                    # Load the CSV file into a DataFrame
                    diversity_scores_df = pd.read_csv(file_path)
                    print(f"Loaded existing diversity scores from {file_path}")
            else:    # Assuming `diversity_pairs` is a list of triples (Q_table, DL_table, set of DL_columns)
                    if diversity_pairs:
                        for q_main_table, s_i, dl_columns in diversity_pairs:
                            print(f"Processing Q_table: {q_main_table}, s_i: {s_i}, Columns: {dl_columns}")

                            # Call get_diversity_by_alignment for each row and get  a  data frames  with multiple rows 
                            # corresponding to alignments 
                            diversity_data = self.get_diversity_by_alignment(s_i, dl_columns)
                            # Print or process the returned data as needed
                            if diversity_data is not None:
                                print(f"Diversity data for {q_main_table} and s_i {s_i}:\n", diversity_data)
                                # process  diversity_data which is a dataframe where Q_si_table is our s_i
                                #'columns = ['DL_table', 'DL_column', 'Q_si_table', 'Qsi_column', 'dSize', 'distance', 'Lexsim']

                                # Initialize diversity_score for this combination
                                res=self.compute_diversity_scores(q_main_table,diversity_data)
                
                                diversity_scores_df = pd.concat(
                                [
                                    diversity_scores_df,res
                                ],
                                ignore_index=True
                            )
                                
                                
                            else:
                                print(f"No diversity data available for {q_table} and {dl_table}.")
                                                # Write the DataFrame to the CSV file
                        
            diversity_scores_df.to_csv(file_path, index=False)
            print(f"Wrote new diversity scores to {file_path}")
            self.diversity=diversity_scores_df
       
       
       
       
       
       
        else:
            print("No data loaded. Please load and preprocess the data first.")
            return None
   
    
    def compute_diversity_scores(self, q_main_table, diversity_data):
            """
            Compute diversity scores for each DL_table group in the diversity_data DataFrame.

            Args:
                q_main_table (str): The main query table associated with the computation.
                diversity_data (pd.DataFrame): A DataFrame with columns 
                                            ['DL_table', 'DL_column', 'Q_si_table', 
                                            'Qsi_column', 'dSize', 'distance', 'Lexsim'].

            Returns:
                pd.DataFrame: A DataFrame with columns ["q_main_table", "s_i", "s_j", "diversity_score"].
            """
            # Initialize an empty list to store results
            diversity_scores_list = []

            # Group by DL_table
            grouped = diversity_data.groupby('DL_table')

            for dl_table, group in grouped:
                # Initialize diversity_score for the current group
                diversity_score = 0

                # Compute the diversity_score based on the given conditions
                for _, row in group.iterrows():
                    if row['dSize'] < 20:
                        diversity_score += row['distance']
                    else:
                        diversity_score += (1 - row['Lexsim'])

                # Append the result as a dictionary to the list
                diversity_scores_list.append({
                    "q_main_table": q_main_table,
                    "s_i": group['Q_si_table'].iloc[0],  # s_i is the same for all rows in the group
                    "s_j": dl_table,
                    "diversity_score": diversity_score
                })

            # Convert the list to a DataFrame
            diversity_scores_df = pd.DataFrame(diversity_scores_list)
            return diversity_scores_df
    
    def get_diversity_by_alignment(self, s_i, columns_):
        """
        get  diversity data for each DL_table and s_i pair 
        however  limit s_i to its columns_ then find alignment with each DL_table 
             

        :return: A pandas DataFrame with the diversity data for s_i and columns_ 
        """
        # first generate data for diversity between s_i on columns_ with all dl_tables
        current_diversity=self.generateDataForDiversity(s_i, columns_)
        return current_diversity  

    
    
    def generateDataForDiversity(self, s_i, columns_):
        '''s_i is the name of the file contaning table rows
           columns_ is the set of column indices that we consider
           returns a dataframe '''
        
        


        
    
        # Call NaiveSearcher, which has linear search and bounds search, from naive_search.py
        searcher = NaiveSearcherNovelty(self.table_raw_lol_proccessed,self.table_path, 1.00,self.table_raw_index, self.table_raw_proccessed_los, 1)
        
        
        query_times = []
    
        query_start_time = time.time()  
          
        diversity_data = searcher.top_all_for_GMC(s_i,columns_,threshold=0.7)
    
        query_times.append(time.time() - query_start_time)
         
        
        return diversity_data
      
if __name__ == "__main__":
    # Example usage:
    data_source = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/full_small_domain_domain_size_Threshold20_kept.csv"
    search_params = {"keyword": "example", "max_results": 10}

    gmc_search = GMC_Search(data_source, search_params)
    gmc_search.load_data()
    gmc_search.preprocess_data()
        # Calculate unionability
    unionability_scores = gmc_search.calculate_unionability()
    
    
    
    diversity_scores = gmc_search.calculate_diversity()
    
    

    results = gmc_search.execute_search()
    filtered_results = gmc_search.filter_results({"category": "example_category"})
    gmc_search.export_results("path/to/output")
    summary = gmc_search.summarize_results()

    print("Search Summary:", summary)