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
import argparse
import logging
from pathlib import Path
import sys



class Penalized_Search:
    """
    Pe: Penalized_Search is a class for performing penalized re_ranking of unionable tables for Novelty based  unionable  table search.
    """

    def __init__(self, dsize, dataFolder, table_path, query_path_raw, table_path_raw,processed_path ):
            self.alignment_data=None
            self.unionable_tables=None
            lex_data = pd.DataFrame(columns=["q_table", "q_col", "dl_table","dl_col","lexical_distance"])

            
            text_processor = TextProcessor()

            # we preprocess the values in tables both query and data lake tables
            
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
            table_raw_lol_proccessed=test_naive_search_Novelty.getProcessedTables(text_processor, 
                                                                                  dl_tbls_processed_lol_file_name,
                                                                                  processed_path,self. tables_raw,"lol", 1, 1)
            self.table_raw_lol_proccessed=table_raw_lol_proccessed
        

            self.table_path=table_path
            #DSize is a hyper parameter
            self.dsize=dsize
        
        
        
        
        
   
    def _cosine_sim(self, vec1, vec2):
        ''' Get the cosine similarity of two input vectors: vec1 and vec2
        '''
        assert vec1.ndim == vec2.ndim
        return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))
    
    
        
    def load_starmie_vectors(self,  dl_table_vectors ,query_table_vectors):
        '''load starmie vectors for query and data lake and retrun as dictionaries'''

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
            
   
    
    
    def perform_search_optimized(self, p_degree, k, all_vectors):
        # Load the required data

        
        all_ranked_result = {}

                  
        q_table_names = self.alignment_data['query_table_name'].unique()
            # for every query now compute the similarity scores with dt tables 
          
        for query_name in q_table_names:
                    start_time = time.time_ns()  
                    
                    grouped_scores_q_total = pd.DataFrame(columns=["q_table", "dl_table",'penalized_unionability_score'])

                    # get q columns vectors 
                    # Get all rows corresponding to the current query_name
                    query_rows = self.alignment_data[self.alignment_data['query_table_name'] == query_name]
                    # Get all unique dl_table_names for the current query_name
                    dl_table_names = query_rows['dl_table_name'].unique()
                        # Iterate over each dl_table_name
                    for dl_table_name in dl_table_names:
                                    lexdis = self.get_column_based_lexical_distance(query_name,dl_table_name )
                                    sim_data = self.get_column_based_similarity(query_name,dl_table_name , all_vectors)


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

                                    grouped_scores_q_total=pd.concat([grouped_scores,grouped_scores_q_total])

                             
                                       
                    # Filter scores for the current query and sort by score
                    top_k_result = (
                        grouped_scores_q_total[grouped_scores_q_total['q_table'] == query_name]
                        .sort_values(by="penalized_unionability_score", ascending=False)
                        .head(k)
                    )

            # Store the results
                    all_ranked_result[(query_name, k, p_degree)] = (
                        list(top_k_result[["dl_table",'penalized_unionability_score']].to_records(index=False)),
                        (time.time_ns() - start_time) / 10 ** 9
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
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    def parse_args():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description='Penalized Search: Re-ranking of unionable tables for Novelty-based search',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Required arguments
        parser.add_argument('--data_folder', type=str, required=True,
                          help='Path to the data folder containing benchmark data')
        
        parser.add_argument('--alignment_file', type=str, required=True,
                          help='Name of the alignment file (e.g., Manual_Alignment_4gtruth_santos_small_all.csv)')
        
        # Optional arguments
        parser.add_argument('--k_range', type=str, default='2-11',
                          help='Range of k values to evaluate in format "start-end" (default: 2-11)')
        
        parser.add_argument('--p_degree', type=float, default=1.0,
                          help='Penalty degree for re-ranking (default: 1.0)')
        
        parser.add_argument('--domain_size', type=int, default=20,
                          help='Domain size threshold for diversity calculations (default: 20)')
        
        parser.add_argument('--output_file', type=str,
                          help='Path to output file (default: data_folder/diveristy_data/search_results/Penalized/search_result_penalize_04diluted_restricted_pdeg{p_degree}.csv)')
        
        parser.add_argument('--verbose', action='store_true',
                          help='Enable verbose logging')
        
        return parser.parse_args()

    def validate_paths(args):
        """Validate that all required paths exist."""
        data_folder = Path(args.data_folder)
        required_paths = {
            'data_folder': data_folder,
            'query_folder': data_folder / 'query',
            'datalake_folder': data_folder / 'datalake',
            'processed_folder': data_folder / 'proccessed',
            'vectors_folder': data_folder / 'vectors',
            'alignment_file': data_folder / args.alignment_file
        }
        
        for name, path in required_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{name} path does not exist: {path}")

    def setup_paths(args):
        """Setup all required paths for the search."""
        data_folder = Path(args.data_folder)
        
        paths = {
            'alignment_file': data_folder / args.alignment_file,
            'starmie_results': data_folder / 'diveristy_data/search_results/Starmie/top_20_Starmie_output_04diluted_restricted_noscore.pkl',
            'dl_vectors': data_folder / 'vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl',
            'query_vectors': data_folder / 'vectors/cl_query_drop_col_tfidf_entity_column_0.pkl',
            'query_raw': data_folder / 'query',
            'table_raw': data_folder / 'datalake',
            'processed': data_folder / 'proccessed'
        }
        
        # Set default output file if not specified
        if not args.output_file:
            paths['output_file'] = data_folder / f'diveristy_data/search_results/Penalized/search_result_penalize_04diluted_restricted_pdeg{args.p_degree}.csv'
        else:
            paths['output_file'] = Path(args.output_file)
            
        return paths

    def run_search(penalize_search, k, p_degree, all_vectors, output_file):
        """Run the penalized search for a specific k value."""
        try:
            logger.info(f"Running search for k={k}, p_degree={p_degree}")
            results = penalize_search.perform_search_optimized(p_degree, k, all_vectors)
            
            # Prepare results for writing
            mode = 'a' if output_file.exists() else 'w'
            with open(output_file, mode, newline='') as file:
                writer = csv.writer(file)
                
                # Write header if new file
                if mode == 'w':
                    writer.writerow(['query_name', 'tables', 'penalized_execution_time', 'k', 'pdegree'])
                
                # Write results
                for key, (result, secs) in results.items():
                    result = [r[0] for r in result]  # Extract table names
                    result_str = ', '.join(result) if isinstance(result, list) else str(result)
                    writer.writerow([key[0], result_str, secs, key[1], key[2]])
            
            logger.info(f"Results for k={k} written to {output_file}")
            return results
            
        except Exception as e:
            logger.error(f"Error running search for k={k}: {str(e)}")
            raise

    try:
        # Parse and validate arguments
        args = parse_args()
        if args.verbose:
            logger.setLevel(logging.DEBUG)
            
        # Validate paths
        validate_paths(args)
        
        # Setup paths
        paths = setup_paths(args)
        
        # Initialize search
        logger.info("Initializing penalized search")
        penalize_search = Penalized_Search(
            args.domain_size,
            str(paths['data_folder']),
            str(paths['dl_vectors']),
            str(paths['query_raw']),
            str(paths['table_raw']),
            str(paths['processed'])
        )
        
        # Load required data
        logger.info("Loading alignment data")
        penalize_search.load_column_alignment_data(str(paths['alignment_file']))
        
        logger.info("Loading unionable tables")
        penalize_search.load_unionable_tables(str(paths['starmie_results']))
        
        logger.info("Loading vector representations")
        all_vectors = penalize_search.load_starmie_vectors(
            str(paths['dl_vectors']),
            str(paths['query_vectors'])
        )
        
        # Run search for each k value
        start_k, end_k = map(int, args.k_range.split('-'))
        all_results = {}
        
        for k in range(start_k, end_k + 1):
            results = run_search(penalize_search, k, args.p_degree, all_vectors, paths['output_file'])
            all_results.update(results)
            
        logger.info("Search completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)