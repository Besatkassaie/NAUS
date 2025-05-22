import pandas as pd
import numpy as np
import pickle
import os
import csv
import time
import argparse
from pathlib import Path
from numpy.linalg import norm
from scipy.spatial import distance
from collections import Counter
from SetSimilaritySearch import all_pairs
from naive_search_Novelty import NaiveSearcherNovelty
import test_naive_search_Novelty
from process_column import TextProcessor
import utilities as utl


class Stamie0_Search:
    """
    Stamie0_Search is a class for performing re-ranking of unionable tables using Starmie scoring 
    with manual alignment and getting sum of column similarity scores.
    """

    def __init__(self, dsize, data_folder, table_path, query_path_raw, table_path_raw, processed_path):
        """
        Initialize the Starmie0 Search class.
        
        Args:
            dsize (int): Domain size parameter
            data_folder (str): Path to the data folder
            table_path (str): Path to the table vectors
            query_path_raw (str): Path to raw query data
            table_path_raw (str): Path to raw table data
            processed_path (str): Path to processed data
        """
        self.alignment_data = None
        self.unionable_tables = None
        self.dsize = dsize
        self.table_path = table_path
        
        # Initialize text processor and load raw tables
        text_processor = TextProcessor()
        self.tables_raw = NaiveSearcherNovelty.read_csv_files_to_dict(table_path_raw)
        
        # Process tables and save results
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

    def _cosine_sim(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        assert vec1.ndim == vec2.ndim
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    def load_starmie_vectors(self, dl_table_vectors, query_table_vectors):
        """Load Starmie vectors for query and data lake tables."""
        with open(query_table_vectors, "rb") as qfile:
            queries = pickle.load(qfile)
        queries_dict = {item[0]: item[1] for item in queries}

        with open(dl_table_vectors, "rb") as tfile:
            tables = pickle.load(tfile)
        dl_dict = {item[0]: item[1] for item in tables}
        
        return queries_dict, dl_dict

    def get_column_based_similarity(self, query_name, dl_table_name, all_vectors):
        """Calculate similarity scores for column pairs."""
        queries_dict, dl_dict = all_vectors
        sim_data = pd.DataFrame(columns=["q_table", "q_col", "dl_table", "dl_col", "similarity"])
        
        q_vectors = queries_dict[query_name]
        query_rows = self.alignment_data[self.alignment_data['query_table_name'] == query_name]
        specific_rows = query_rows[query_rows['dl_table_name'] == dl_table_name]
        dl_t_vectors = dl_dict[dl_table_name]

        for _, row in specific_rows.iterrows():
            query_column = row['query_column#']
            dl_column = row['dl_column']
            similarity_col = self._cosine_sim(q_vectors[query_column], dl_t_vectors[dl_column])
            
            sim_data = pd.concat([
                sim_data,
                pd.DataFrame({
                    "q_table": [query_name],
                    "dl_table": [dl_table_name],
                    "q_col": [query_column],
                    "dl_col": [dl_column],
                    "similarity": [similarity_col]
                })
            ], ignore_index=True)
        
        return sim_data

    def get_column_based_lexical_distance(self, query_name, dl):
        """Calculate lexical distance for column pairs."""
        lexdis_data = pd.DataFrame(columns=["q_table", "q_col", "dl_table", "dl_col", "lex_distance"])
        
        query_rows = self.alignment_data[self.alignment_data['query_table_name'] == query_name]
        dl_rows = query_rows[(query_rows['query_table_name'] == query_name) & 
                           (query_rows['dl_table_name'] == dl)]
        
        for _, row in dl_rows.iterrows():
            dl_column_number = int(row['dl_column'])
            q_column_number = int(row['query_column#'])
            
            dl_column_dl = self.table_raw_lol_proccessed.get(dl)[dl_column_number]
            dl_column_set_dl = self.table_raw_proccessed_los.get(dl)[dl_column_number]
            q_column_q = self.table_raw_lol_proccessed.get(query_name)[q_column_number]
            q_column_set = self.table_raw_proccessed_los.get(query_name)[q_column_number]
            
            domain_estimate = set.union(set(q_column_q), set(dl_column_dl))
            
            if len(domain_estimate) < self.dsize:
                distance = self.Jensen_Shannon_distances(q_column_q, dl_column_dl, domain_estimate)
            else:
                distance = 1 - self._lexicalsim_Pair(dl_column_set_dl, q_column_set)
            
            lexdis_data = pd.concat([
                lexdis_data,
                pd.DataFrame([{
                    "q_table": query_name,
                    "q_col": q_column_number,
                    "dl_table": dl,
                    "dl_col": dl_column_number,
                    "lex_distance": distance
                }])
            ], ignore_index=True)
        
        return lexdis_data

    def load_unionable_tables(self, path):
        """Load unionable tables from pickle file."""
        print("Loading the first round ranked results produced by Starmie")
        self.unionable_tables = utl.loadDictionaryFromPickleFile(path)

    def load_column_alignment_data(self, alignment_file):
        """Load column alignment data from CSV file."""
        print("Loading alignment produced by DUST")
        try:
            self.alignment_data = pd.read_csv(alignment_file)
            required_columns = ['query_table_name', 'query_column', 'query_column#',
                              'dl_table_name', 'dl_column#', 'dl_column']
            
            if not all(column in self.alignment_data.columns for column in required_columns):
                missing_columns = [col for col in required_columns if col not in self.alignment_data.columns]
                raise ValueError(f"Missing required columns in data: {missing_columns}")
            
            print("Alignment data loaded successfully")
        
        except Exception as e:
            print(f"Error loading alignment data: {str(e)}")
            raise

    def perform_search_optimized(self, p_degree, k, all_vectors):
        """Perform optimized search with given parameters."""
        all_ranked_result = {}
        q_table_names = self.alignment_data['query_table_name'].unique()
        
        for query_name in q_table_names:
            start_time = time.time_ns()
            grouped_scores_q_total = pd.DataFrame(columns=["q_table", "dl_table", 'starmie0_unionability_score'])
            
            query_rows = self.alignment_data[self.alignment_data['query_table_name'] == query_name]
            dl_table_names = query_rows['dl_table_name'].unique()
            
            for dl_table_name in dl_table_names:
                lexdis = self.get_column_based_lexical_distance(query_name, dl_table_name)
                sim_data = self.get_column_based_similarity(query_name, dl_table_name, all_vectors)
                
                merged_data = pd.merge(
                    lexdis, sim_data,
                    on=["q_table", "dl_table", "q_col", "dl_col"],
                    how="inner",
                    suffixes=("_lex", "_sim")
                )
                
                merged_data['starmie0_unionability_score'] = merged_data['similarity']
                grouped_scores = merged_data.groupby(["q_table", "dl_table"])['starmie0_unionability_score'].sum().reset_index()
                grouped_scores_q_total = pd.concat([grouped_scores, grouped_scores_q_total])
            
            top_k_result = (
                grouped_scores_q_total[grouped_scores_q_total['q_table'] == query_name]
                .sort_values(by="starmie0_unionability_score", ascending=False)
                .head(k)
            )
            
            all_ranked_result[(query_name, k, p_degree)] = (
                list(top_k_result[["dl_table", 'starmie0_unionability_score']].to_records(index=False)),
                (time.time_ns() - start_time) / 10 ** 9
            )
        
        return all_ranked_result

    def _lexicalsim_Pair(self, query_column, table_column):
        """Calculate lexical similarity between two columns."""
        sets = [query_column, set(table_column)]
        pairs = list(all_pairs(sets, similarity_func_name="jaccard", similarity_threshold=0.0))
        return pairs[0][2] if pairs else 0

    def item_frequency(self, lst):
        """Calculate frequency of items in a list."""
        return dict(Counter(lst))

    def Jensen_Shannon_distances(self, query_column, dl_column, domain_estimate):
        """Calculate Jensen-Shannon distance between two columns."""
        x_axis = {item: i for i, item in enumerate(domain_estimate)}
        
        frequency_q = self.item_frequency(query_column)
        frequency_dl = self.item_frequency(dl_column)
        
        list_length_q = len(query_column)
        list_length_dl = len(dl_column)
        
        array_q = np.zeros(len(domain_estimate))
        array_dl = np.zeros(len(domain_estimate))
        
        for item in domain_estimate:
            index_ = x_axis[item]
            array_q[index_] = frequency_q.get(item, 0) / float(list_length_q)
            array_dl[index_] = frequency_dl.get(item, 0) / float(list_length_dl)
        
        dis_qdl = distance.jensenshannon(array_q, array_dl)
        dis_dlq = distance.jensenshannon(array_dl, array_q)
        
        if dis_qdl != dis_dlq:
            raise ValueError('The distance metric is asymmetric')
        
        return dis_dlq


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Starmie0 Search with manual alignment')
    
    parser.add_argument('--data_folder', type=str, required=True,
                      help='Path to the data folder (e.g., data/santos/small)')
    
    parser.add_argument('--alignment_file', type=str, required=True,
                      help='Name of the alignment file (e.g., Manual_Alignment_4gtruth_santos_small_all.csv)')
    
    parser.add_argument('--starmie_results', type=str, required=True,
                      help='Path to Starmie results pickle file')
    
    parser.add_argument('--output_file', type=str, required=True,
                      help='Path to output CSV file')
    
    parser.add_argument('--k_start', type=int, default=2,
                      help='Starting value of k (default: 2)')
    
    parser.add_argument('--k_end', type=int, default=10,
                      help='Ending value of k (default: 10)')
    
    parser.add_argument('--p_degree', type=float, default=1.0,
                      help='Penalty degree (default: 1.0)')
    
    parser.add_argument('--domain_size', type=int, default=20,
                      help='Domain size parameter (default: 20)')
    
    return parser.parse_args()


def main():
    """Main function to run Starmie0 search."""
    args = parse_args()
    
    # Set up paths
    data_folder = Path(args.data_folder)
    alignment_file = data_folder / args.alignment_file
    starmie_results = data_folder / args.starmie_results
    output_file = data_folder / args.output_file
    
    # Set up vector paths
    dl_table_vectors = data_folder / "vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl"
    query_table_vectors = data_folder / "vectors/cl_query_drop_col_tfidf_entity_column_0.pkl"
    
    # Set up raw data paths
    query_path_raw = data_folder / "query"
    table_path_raw = data_folder / "datalake"
    processed_path = data_folder / "proccessed"
    
    # Initialize search
    print("Initializing Starmie0 Search...")
    search = Stamie0_Search(
        args.domain_size,
        str(data_folder),
        str(dl_table_vectors),
        str(query_path_raw),
        str(table_path_raw),
        str(processed_path)
    )
    
    # Load data
    print("Loading alignment data...")
    search.load_column_alignment_data(str(alignment_file))
    
    print("Loading unionable tables...")
    search.load_unionable_tables(str(starmie_results))
    
    print("Loading Starmie vectors...")
    all_vectors = search.load_starmie_vectors(str(dl_table_vectors), str(query_table_vectors))
    
    # Process results for each k value
    print(f"Processing results for k={args.k_start} to {args.k_end}...")
    for k in range(args.k_start, args.k_end + 1):
        print(f"Processing k={k}...")
        results = search.perform_search_optimized(args.p_degree, k, all_vectors)
        
        # Write results
        file_exists = os.path.exists(output_file)
        mode = 'a' if file_exists else 'w'
        
        with open(output_file, mode=mode, newline='') as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow(['query_name', 'tables', 'starmie0_execution_time', 'k', 'pdegree'])
            
            for key, (result, secs) in results.items():
                result = [r[0] for r in result]
                result_str = ', '.join(result)
                writer.writerow([key[0], result_str, secs, key[1], key[2]])
    
    print(f"Results written to {output_file}")


if __name__ == "__main__":
    main()
   