# %%
import random
import glob
import os
import torch
import time
import math
import re
import json, shutil
import string
import numpy as np
import tqdm as tqdm
import pandas as pd
import _pickle as cPickle
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import utilities as utl
from transformers import BertTokenizer, BertModel, RobertaTokenizerFast, RobertaModel
import torch, sys
import torch.nn as nn
from torch.nn.parallel import DataParallel
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import itertools
from sklearn import metrics
#from glove_embeddings import GloveTransformer
#import fasttext_embeddings as ft
import matplotlib.pyplot as plt
from model_classes import BertClassifierPretrained, BertClassifier
import csv
from scipy.sparse.csgraph import connected_components
import argparse

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import difflib


# %%
def init():
    print("init() is called .....")
    available_embeddings = ["bert", "bert_serialized", "roberta", "roberta_serialized", "sentence_bert", "sentence_bert_serialized", "glove", "fasttext", "dust", "dust_serialized", "starmie"]

    embedding_type = available_embeddings[3] #change it to fasttext or bert for using the respective embeddings.
    use_numeric_columns = True
#benchmark_name = "tus_benchmark" 
    benchmark_name = "santos" 
    clustering_metric = "cosine" # "cosine"
    single_col = 0

    dl_table_folder = r"data" + os.sep + benchmark_name + os.sep + "datalake"
    query_table_folder = r"data" + os.sep + benchmark_name + os.sep + "query"
#groundtruth_file = r"groundtruth" + os.sep + benchmark_name + "_union_groundtruth.pickle"
    groundtruth_file = r"groundtruth" + os.sep + benchmark_name + "_union_groundtruth.pickle"

    query_tables = glob.glob(query_table_folder + os.sep + "*.csv")
    groundtruth = utl.loadDictionaryFromPickleFile(groundtruth_file)
# limit the ground truth to only one query and 10 unionables 
# Limit the ground truth to one query and 10 unionables
    limited_groundtruth = {}
    if groundtruth:
    # Pick a single query (the first one in this case)
        single_query = next(iter(groundtruth))

    # Extract the query name without the file extension
        query_name = single_query.split(".")[0]

    # Split the query name into individual substrings
        query_substrings = set(query_name.split("_"))

    # Filter unionables based on the condition
        unionables = [
        table for table in groundtruth[single_query]
        if sum(1 for substr in table.split("_") if substr in query_substrings) <= 1
    ][:10]  # Limit to 10 unionables

    # Only include the filtered unionables
        limited_groundtruth[single_query] = unionables

    groundtruth = limited_groundtruth
 
    align_plot_folder = r"plots_align"

# %%
    print("Embedding type: ", embedding_type)
    if embedding_type == "bert" or embedding_type == "bert_serialized":
        model = BertModel.from_pretrained('bert-base-uncased') 
        model = BertClassifierPretrained(model)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        vec_length = 768
    elif embedding_type == "roberta" or embedding_type == "roberta_serialized":
        model = RobertaModel.from_pretrained("roberta-base")
        model = BertClassifierPretrained(model)
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        vec_length = 768
    elif embedding_type == "sentence_bert" or embedding_type == "sentence_bert_serialized":
        model = SentenceTransformer('bert-base-uncased') #case insensitive model. BOSTON and boston have the same embedding.
        tokenizer = ""
        vec_length = 768
    elif embedding_type == "glove":
        model = GloveTransformer()
        tokenizer = ""
        vec_length = 300
    elif embedding_type == "fasttext":
        model = ft.get_embedding_model()
        tokenizer = ""
        vec_length = 300
    elif embedding_type == "dust" or embedding_type == "dust_serialized":
        model_path = r'./out_model/tus_finetune_roberta/checkpoints/best-checkpoint.pt'
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        model = RobertaModel.from_pretrained('roberta-base')
        model = BertClassifier(model, num_labels = 2, hidden_size = 768, output_size = 768)
        model = DataParallel(model, device_ids=[0, 1, 2, 3]) 
    #print(model)   
        model.load_state_dict(torch.load(model_path))
    elif embedding_type == "starmie":
        if single_col == 0:
            dl_column_embeddings = utl.loadDictionaryFromPickleFile(r"starmie_embeddings" + os.sep + benchmark_name + "_vectors" + os.sep + "cl_datalake_drop_col_tfidf_entity_column_0.pkl")
            query_column_embeddings = utl.loadDictionaryFromPickleFile(r"starmie_embeddings" + os.sep + benchmark_name + "_vectors" + os.sep + "cl_query_drop_col_tfidf_entity_column_0.pkl")
        else:
            dl_column_embeddings = utl.loadDictionaryFromPickleFile(r"starmie_embeddings" + os.sep + benchmark_name + "_vectors" + os.sep + "cl_datalake_drop_col_tfidf_entity_column_0_singleCol.pkl")
            query_column_embeddings = utl.loadDictionaryFromPickleFile(r"starmie_embeddings" + os.sep + benchmark_name + "_vectors" + os.sep + "cl_query_drop_col_tfidf_entity_column_0_singleCol.pkl")
    
        dl_column_embeddings = {key: value for key, value in dl_column_embeddings}
        query_column_embeddings = {key: value for key, value in query_column_embeddings}
        starmie_embeddings = {}
        for table in query_column_embeddings:
            if os.path.exists(query_table_folder + os.sep + table):
                table_df = utl.read_csv_file(query_table_folder + os.sep + table)
                col_headers = [str(col).strip() for col in table_df.columns] 
                for idx, item in enumerate(col_headers):
                    starmie_embeddings[(table, item)] =  query_column_embeddings[table][idx]
        for table in dl_column_embeddings:
            table_df = utl.read_csv_file(dl_table_folder + os.sep + table)
            col_headers = [str(col).strip() for col in table_df.columns] 
            for idx, item in enumerate(col_headers):
                starmie_embeddings[(table, item)] =  dl_column_embeddings[table][idx]

    else:
        print("invalid embedding type")
        sys.exit()
    return embedding_type,use_numeric_columns,benchmark_name,clustering_metric,dl_table_folder,query_tables,groundtruth,model,tokenizer,starmie_embeddings

 # %%   
 
def export_alignment_to_csv(final_alignment, track_columns_reverse, output_file):
    """
    Export the final alignment to a CSV file.

    Parameters:
        final_alignment (set): A set of tuples representing aligned column pairs.
        track_columns_reverse (dict): Reverse mapping from indices to (table_name, column_name).
        output_file (str): Path to the output CSV file.
    """
    file_exists = os.path.exists(output_file)
    

    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header
        if not file_exists:
                # If file doesn't exist, write column names
            writer.writerow(["query_table_name", "query_column", "query_column#", "dl_table_name", "dl_column#","dl_column"])
        
        for col_pair in final_alignment:
            # Get the table and column names for both columns in the pair
            query_col = track_columns_reverse[col_pair[0]]
            dl_col = track_columns_reverse[col_pair[1]]
            
            # Write the row to the CSV file
            writer.writerow([query_col[0], query_col[1],query_col[2], dl_col[0], dl_col[1],dl_col[2]])
    
    print(f"Alignment exported to {output_file}") 


def gmc_alignmnet_by_query( query_table,query_columns,dl_tables, embedding_type='roberta_serialized'):   
    
    
    use_numeric_columns = True
    #benchmark_name = "tus_benchmark" 
    benchmark_name = "santos" 
    clustering_metric = "cosine" # "cosine"

    dl_table_folder = r"data" + os.sep + benchmark_name + os.sep + "datalake"

    groundtruth={query_table:dl_tables}
  

    
    start_time = time.time_ns()
    used_queries = 0
    # evaluation of align phase.
    query_table_name = query_table.rsplit(os.sep, 1)[-1] #should be fixed if query table name doesn;t have csv in its name
    print(query_table_name)
    if query_table_name == "workforce_management_information_a.csv" or query_table_name == "workforce_management_information_b.csv":
        return None
    column_embeddings = []
    track_tables = {}
    track_columns = {}
    #mapping from universal comlumn id to (table, column#)
    track_columns_reverse={}
    record_same_cluster = {}
    query_column_ids = set()
    i = 0
    if query_table_name not in groundtruth:
        return None
    else:
        unionable_tables = groundtruth[query_table_name]
        # get embeddings for query table columns.
        query_embeddings = compute_embeddings_single_table(dl_table_folder+"/"+query_table, embedding_type, use_numeric_columns=use_numeric_columns)
        if len(query_embeddings) == 0:
            print("Not enough rows. Ignoring this query table.")
            return None
        print(query_table)
        # break
        used_queries += 1
        unionable_table_path = [dl_table_folder + os.sep + tab for tab in unionable_tables if tab != query_table_name]
        unionable_table_path = [path for path in unionable_table_path if os.path.exists(path)]
        
        if benchmark_name == "tus_benchmark":
            unionable_table_path = random.sample(unionable_table_path, min(10, len(unionable_table_path))) 
            # for e in unionable_table_path:
            #     f_name = os.path.basename(e)
            #     destination_path = os.path.join(destination_folder, f_name)
            #     shutil.copyfile(e, destination_path)
            # continue
        print("starting dl embedings------")
        dl_embeddings = compute_embeddings(unionable_table_path, embedding_type, use_numeric_columns= use_numeric_columns)
        print("end dl embedings----")
        
        print("project query embedings----")
        
        #Filter query_embeddings to only include columns that are part of query_columns
        filtered_query_embeddings = {
            key: value
            for key, value in query_embeddings.items()
            if key[2] in query_columns
        }

        # Replace the original query_embeddings with the filtered version if needed
        query_embeddings = filtered_query_embeddings
        
        if len(dl_embeddings) == 0:
            print("Not enough rows in any data lake tables. Ignoring this cluster.")
        for column in query_embeddings:
            column_embeddings.append(query_embeddings[column])
            track_columns[column] = i
            track_columns_reverse[i]=column
            if column[0] not in track_tables:
                track_tables[column[0]] = {i}
            else:
                track_tables[column[0]].add(i)
            if column[1] not in record_same_cluster:
                record_same_cluster[column[1]] =  {i}
            else:
                record_same_cluster[column[1]].add(i)
            query_column_ids.add(i)
            i += 1
        for column in dl_embeddings:
            column_embeddings.append(dl_embeddings[column])
            track_columns[column] = i
            track_columns_reverse[i]=column
            if column[0] not in track_tables:
                track_tables[column[0]] = {i}
            else:
                track_tables[column[0]].add(i)
            if column[1] not in record_same_cluster:
                record_same_cluster[column[1]] =  {i}
            else:
                record_same_cluster[column[1]].add(i)
            i += 1
            
        all_true_edges = set() 
        all_true_query_edges = set()
        for col_index_set in record_same_cluster:
            set1 = record_same_cluster[col_index_set]
            set2 = record_same_cluster[col_index_set]
            current_true_edges = set()
            current_true_query_edges = set()
            for s1 in set1:
                for s2 in set2:
                    current_true_edges.add(tuple(sorted((s1,s2))))
                    if s1 in query_column_ids or s2 in query_column_ids:
                        current_true_query_edges.add(tuple(sorted((s1,s2))))
            all_true_edges = all_true_edges.union(current_true_edges)  
            all_true_query_edges = all_true_query_edges.union(current_true_query_edges) 
        column_embeddings = list(column_embeddings)
        x = np.array(column_embeddings)
        
        zero_positions = set()
        for table in track_tables:
            indices = track_tables[table]
            all_combinations = findsubsets(indices, 2)
            for each in all_combinations:
                zero_positions.add(each)
        
        arr = np.zeros((len(track_columns),len(track_columns)))
        for i in range(0, len(track_columns)-1):
            for j in range(i+1, len(track_columns)):
                #print(i, j)
                if (i, j) not in zero_positions and (j, i) not in zero_positions and i !=j:
                    arr[i][j] = 1
                    arr[j][i] = 1
        # convert to sparse matrix representation 
        s = csr_matrix(arr)  
# with  s he is taking care of not having columns from the same table to appear in the same cluster 
        all_distance = {}
        all_labels = {}
        record_current_precision = {}
        record_current_recall = {}
        record_current_f_measure = {}
        record_current_query_precision = {}
        record_current_query_recall = {}
        record_current_query_f_measure = {}
        min_k = len(query_embeddings)
        max_k = 0
        record_result_edges = {}
        record_result_query_edges = {}
        
        for item in track_tables:
            #print(item, len(track_tables[item]))
            if len(track_tables[item])> min_k:
                min_k = len(track_tables[item])
            max_k += len(track_tables[item])
        
        clusterAlg_2_all_result_query_edges={}
        print("cluster numbers "+str(min_k)+" to "+str(max_k))
        for i in range(min_k, min(max_k, max_k)):
            print("started "+str(i)+" cluster for "+ query_table_name)
            #clusters = KMeans(n_clusters=14).fit(x)
            clusters = AgglomerativeClustering(n_clusters=i, metric=clustering_metric,
                        compute_distances = True , linkage='average', connectivity = s)
            clusters.fit_predict(x)
            labels = (clusters.labels_) #.tolist()
            all_labels[i]= labels.tolist()
            #The silhouette ranges from −1 to +1, where a high value indicates that the object is
            # well matched to its own cluster and poorly matched to neighboring clusters. If most objects
            # have a high value, then the clustering configuration is appropriate. If many points have a low or negative value, 
            # then the clustering configuration may have too many or too few clusters
            # . A clustering with an average silhouette width of over 0.7 is considered to
            # be "strong", a value over 0.5 "reasonable" and over 0.25 "weak",
            all_distance[i] = metrics.silhouette_score(x, labels)
            result_dict = {}
            wrong_results = set()
            for (col_index, label) in enumerate(all_labels[i]):
                if label in result_dict:
                    result_dict[label].add(col_index)
                else:
                    result_dict[label] = {col_index}
            #result_dict is a mapping from cluster label to columns in the cluster 
            all_result_edges = set() 
            all_result_query_edges = set()
            for col_index_set in result_dict:
                set1 = result_dict[col_index_set]
                set2 = result_dict[col_index_set]
                current_result_edges = set()
                current_result_query_edges = set()
                for s1 in set1:
                    for s2 in set2:
                        current_result_edges.add(tuple(sorted((s1,s2))))
                        if s1 in query_column_ids or s2 in query_column_ids:
                            current_result_query_edges.add(tuple(sorted((s1,s2))))
                all_result_edges = all_result_edges.union(current_result_edges)
                all_result_query_edges = all_result_query_edges.union(current_result_query_edges)
            current_true_positive = len(all_true_edges.intersection(all_result_edges))
            current_precision = current_true_positive/len(all_result_edges)
            current_recall = current_true_positive/len(all_true_edges)

            current_query_true_positive = len(all_true_query_edges.intersection(all_result_query_edges))
            current_query_precision = current_query_true_positive/len(all_result_query_edges)
            current_query_recall = current_query_true_positive/len(all_true_query_edges)

            
            clusterAlg_2_all_result_query_edges[i]=(all_result_query_edges,result_dict)


            record_current_precision[i] = current_precision
            record_current_recall[i] = current_recall
            record_current_f_measure[i] = 0
            if (current_precision + current_recall) > 0:
                record_current_f_measure[i] = (2 * current_precision * current_recall)/ (current_precision + current_recall)              
            record_result_edges[i] = all_result_edges

            record_current_query_precision[i] = current_query_precision
            record_current_query_recall[i] = current_query_recall
            record_current_query_f_measure[i] = 0
            if (current_query_precision + current_query_recall) > 0:
                record_current_query_f_measure[i] = (2 * current_query_precision * current_query_recall)/ (current_query_precision + current_query_recall)              
            record_result_query_edges[i] = all_result_query_edges
            print("Ended "+str(i)+" cluster for "+ query_table_name)

            
            distance_list = all_distance.items()
            distance_list = sorted(distance_list) 
            x, y = zip(*distance_list)
            algorithm_k = max(all_distance, key=all_distance. get) 
            # get the best clustering algorithm as algorithm_k then retrive the result_dict
            final_alignment_4_query=clusterAlg_2_all_result_query_edges[algorithm_k][0]
            
            alignment_list=[]
            for col_pair in final_alignment_4_query:
            # Get the table and column names for both columns in the pair
                query_col = track_columns_reverse[col_pair[0]]
                dl_col = track_columns_reverse[col_pair[1]]
                
                alignment_list.append([query_col[0], query_col[1],query_col[2], dl_col[0], dl_col[1],dl_col[2]])
            #export_alignment_to_csv(final_alignment_4_query, track_columns_reverse, "DUST_Alignment.csv")
            #query_col[0] == "311_calls_historic_data_0.csv" && dl_col[0] == "311_calls_historic_data_3.csv" 
            return alignment_list


def remove_problematic_alignments(final_alignment_4_query, queries_with_problematic_alignments, track_columns_reverse):
    output=set()
    seen=set()
    # Group second elements by their first element
    for col_pair in final_alignment_4_query:
        dl_col = track_columns_reverse[col_pair[1]]
        query_col= track_columns_reverse[col_pair[0]]
        query_table_name=query_col[0]
        query_column=query_col[1]
        query_columnnumber=query_col[2] 
        dl_table_name=dl_col[0]
        dl_column=dl_col[1]
        dl_columnnumber=dl_col[2]
        temp=(query_columnnumber, dl_table_name) 
        if temp  in   seen: 
                if query_table_name in queries_with_problematic_alignments:
                      queries_with_problematic_alignments[query_table_name].add(col_pair)
                else:
                      queries_with_problematic_alignments[query_table_name] = {col_pair}
             
        else:
          seen.add(temp)  
          output.add(col_pair)  
    
    return  output, queries_with_problematic_alignments
    
    

                       
                        
 
                 
    
    




 
def DUST_alignmnet_export(output_file, dl_table_folder,query_table_folder, benchmark_name,groundtruth_file, excluded_queries):   
    
   if os.path.exists(output_file):
        print(f"{output_file} exists.")
   else:
        use_numeric_columns = True
        #benchmark_name = "tus_benchmark" 
    
        clustering_metric = "cosine" # "cosine"
        #clustering_metric = "cityblock" # "cosine"

        query_tables = glob.glob(query_table_folder + os.sep + "*.csv")
        if benchmark_name=='santos':
            groundtruth = utl.loadDictionaryFromPickleFile(groundtruth_file)
        if   benchmark_name=='ugen_v2':
            groundtruth = utl.loadDictionaryFromCSV_ugenv2(groundtruth_file)
        if benchmark_name=='tus_benchmark': 
            groundtruth=utl.create_query_to_datalake_dict(groundtruth_file)
            

        # # limit the ground truth to only one query and 10 unionables 
        # # Limit the ground truth to one query and 10 unionables
        # limited_groundtruth = {}
        # if groundtruth:
        #     # Pick a single query (the first one in this case)
        #     #single_query = next(iter(groundtruth))

        #     # Extract the query name without the file extension
        #     #query_name = single_query.split(".")[0]
        #     query_name =  "World-History_GW90JPKL_ugen_v2.csv"
        #     # Split the query name into individual substrings
        #     query_substrings = set(query_name.split("_"))

        #     # Filter unionables based on the condition
        #     # unionables = [
        #     #     table for table in groundtruth[single_query]
        #     #     if sum(1 for substr in table.split("_") if substr in query_substrings) <= 1
        #     # ][:10]  # Limit to 10 unionables

        #     # Only include the filtered unionables
        #     #limited_groundtruth[single_query] = unionables
        #     limited_groundtruth[query_name] = ["World-History_VZK3I8BZ_ugen_v2.csv"]

        #groundtruth = limited_groundtruth


        start_time = time.time_ns()
        used_queries = 0
        queries_with_problematic_alignments={}
        # evaluation of align phase.
        for query_table in query_tables:
            query_table_name = query_table.rsplit(os.sep, 1)[-1]
            print("query name: "+query_table_name)
            if query_table_name in excluded_queries:
                continue
            column_embeddings = []
            track_tables = {}
            track_columns = {}
            #mapping from universal comlumn id to (table, column#)
            track_columns_reverse={}
            record_same_cluster = {}
            query_column_ids = set()
            i = 0
            if query_table_name not in groundtruth:
                print("query is not considered")
                continue
            else:
                unionable_tables = groundtruth[query_table_name]
                # get embeddings for query table columns.
                query_embeddings = compute_embeddings([query_table], embedding_type, use_numeric_columns=use_numeric_columns)
                if len(query_embeddings) == 0:
                    print("Not enough rows. Ignoring this query table.")
                    continue
                # print(query_embeddings)
                # break
                used_queries += 1
                print("working on query number: "+ str(used_queries))
                unionable_table_path = [dl_table_folder + os.sep + tab for tab in unionable_tables if tab != query_table_name]
                unionable_table_path = [path for path in unionable_table_path if os.path.exists(path)]
                
                if benchmark_name == "tus_benchmark":
                    unionable_table_path = random.sample(unionable_table_path, min(10, len(unionable_table_path))) 
                    # for e in unionable_table_path:
                    #     f_name = os.path.basename(e)
                    #     destination_path = os.path.join(destination_folder, f_name)
                    #     shutil.copyfile(e, destination_path)
                    # continue
            #   print("starting dl embedings------")
                dl_embeddings = compute_embeddings(unionable_table_path, embedding_type, use_numeric_columns= use_numeric_columns)
            #  print("end dl embedings----")
                if len(dl_embeddings) == 0:
                    print("Not enough rows in any data lake tables. Ignoring this cluster.")
                for column in query_embeddings:
                    column_embeddings.append(query_embeddings[column])
                    track_columns[column] = i
                    track_columns_reverse[i]=column
                    if column[0] not in track_tables:
                        track_tables[column[0]] = {i}
                    else:
                        track_tables[column[0]].add(i)
                    if column[1] not in record_same_cluster:
                        record_same_cluster[column[1]] =  {i}
                    else:
                        record_same_cluster[column[1]].add(i)
                    query_column_ids.add(i)
                    i += 1
                for column in dl_embeddings:
                    column_embeddings.append(dl_embeddings[column])
                    track_columns[column] = i
                    track_columns_reverse[i]=column
                    if column[0] not in track_tables:
                        track_tables[column[0]] = {i}
                    else:
                        track_tables[column[0]].add(i)
                    if column[1] not in record_same_cluster:
                        record_same_cluster[column[1]] =  {i}
                    else:
                        record_same_cluster[column[1]].add(i)
                    i += 1
                    
                all_true_edges = set() 
                all_true_query_edges = set()
                for col_index_set in record_same_cluster:
                    set1 = record_same_cluster[col_index_set]
                    set2 = record_same_cluster[col_index_set]
                    current_true_edges = set()
                    current_true_query_edges = set()
                    for s1 in set1:
                        for s2 in set2:
                            current_true_edges.add(tuple(sorted((s1,s2))))
                            if s1 in query_column_ids or s2 in query_column_ids:
                                current_true_query_edges.add(tuple(sorted((s1,s2))))
                    all_true_edges = all_true_edges.union(current_true_edges)  
                    all_true_query_edges = all_true_query_edges.union(current_true_query_edges) 
                column_embeddings = list(column_embeddings)
                x = np.array(column_embeddings)
                
                zero_positions = set()
                for table in track_tables:
                    indices = track_tables[table]
                    all_combinations = findsubsets(indices, 2)
                    for each in all_combinations:
                        zero_positions.add(each)
                
                arr = np.zeros((len(track_columns),len(track_columns)))
                for i in range(0, len(track_columns)-1):
                    for j in range(i+1, len(track_columns)):
                        #print(i, j)
                        if (i, j) not in zero_positions and (j, i) not in zero_positions and i !=j:
                            arr[i][j] = 1
                            arr[j][i] = 1
                # convert to sparse matrix representation 
                s = csr_matrix(arr)  
                
                ## checking the number of connected components in the connectivity graph 
                n_components, labels = connected_components(csgraph=s, directed=False, connection='weak')
                if n_components>1:
                    print (f"{query_table} has a connectivity graph with more than 1 connected component")
        # with  s he is taking care of not having columns from the same table to appear in the same cluster 
                all_distance = {}
                all_labels = {}
                record_current_precision = {}
                record_current_recall = {}
                record_current_f_measure = {}
                record_current_query_precision = {}
                record_current_query_recall = {}
                record_current_query_f_measure = {}
                min_k = len(query_embeddings)
                max_k = 0
                record_result_edges = {}
                record_result_query_edges = {}
                
                for item in track_tables:
                    #print(item, len(track_tables[item]))
                    if len(track_tables[item])> min_k:
                        min_k = len(track_tables[item])
                    max_k += len(track_tables[item])
                
                clusterAlg_2_all_result_query_edges={}
                

                #print("cluster numbers "+str(min_k)+" to "+str(max_k))
                for i in range(min_k, min(max_k, max_k)):
                    #print("started "+str(i)+" cluster for "+ query_table_name)
                    #clusters = KMeans(n_clusters=14).fit(x)
                    clusters = AgglomerativeClustering(n_clusters=i, metric=clustering_metric,
                                compute_distances = True , linkage='average', connectivity = s)
                    clusters.fit_predict(x)
            

                    labels = (clusters.labels_) #.tolist()
                    all_labels[i]= labels.tolist()
                    #The silhouette ranges from −1 to +1, where a high value indicates that the object is
                    # well matched to its own cluster and poorly matched to neighboring clusters. If most objects
                    # have a high value, then the clustering configuration is appropriate. If many points have a low or negative value, 
                    # then the clustering configuration may have too many or too few clusters
                    # . A clustering with an average silhouette width of over 0.7 is considered to
                    # be "strong", a value over 0.5 "reasonable" and over 0.25 "weak",
                    all_distance[i] = metrics.silhouette_score(x, labels)
                    result_dict = {}
                    wrong_results = set()
                    for (col_index, label) in enumerate(all_labels[i]):
                        if label in result_dict:
                            result_dict[label].add(col_index)
                        else:
                            result_dict[label] = {col_index}
                    #result_dict is a mapping from cluster label to columns in the cluster 
                    all_result_edges = set() 
                    all_result_query_edges = set()
                    for col_index_set in result_dict:
                        set1 = result_dict[col_index_set]
                        set2 = result_dict[col_index_set]
                        current_result_edges = set()
                        current_result_query_edges = set()
                        for s1 in set1:
                            for s2 in set2:
                                current_result_edges.add(tuple(sorted((s1,s2))))
                                if s1 in query_column_ids or s2 in query_column_ids:
                                    current_result_query_edges.add(tuple(sorted((s1,s2))))
                        all_result_edges = all_result_edges.union(current_result_edges)
                        all_result_query_edges = all_result_query_edges.union(current_result_query_edges)
                    current_true_positive = len(all_true_edges.intersection(all_result_edges))
                    current_precision = current_true_positive/len(all_result_edges)
                    current_recall = current_true_positive/len(all_true_edges)

                    current_query_true_positive = len(all_true_query_edges.intersection(all_result_query_edges))
                    current_query_precision = current_query_true_positive/len(all_result_query_edges)
                    current_query_recall = current_query_true_positive/len(all_true_query_edges)

                    
                    clusterAlg_2_all_result_query_edges[i]=(all_result_query_edges,result_dict)
    
    
                    record_current_precision[i] = current_precision
                    record_current_recall[i] = current_recall
                    record_current_f_measure[i] = 0
                    if (current_precision + current_recall) > 0:
                        record_current_f_measure[i] = (2 * current_precision * current_recall)/ (current_precision + current_recall)              
                    record_result_edges[i] = all_result_edges

                    record_current_query_precision[i] = current_query_precision
                    record_current_query_recall[i] = current_query_recall
                    record_current_query_f_measure[i] = 0
                    if (current_query_precision + current_query_recall) > 0:
                        record_current_query_f_measure[i] = (2 * current_query_precision * current_query_recall)/ (current_query_precision + current_query_recall)              
                    record_result_query_edges[i] = all_result_query_edges
                    #print("Ended "+str(i)+" cluster for "+ query_table_name)

                
                distance_list = all_distance.items()
                distance_list = sorted(distance_list) 
                x, y = zip(*distance_list)
                algorithm_k = max(all_distance, key=all_distance. get) 
                # get the best clustering algorithm as algorithm_k then retrive the result_dict
                final_alignment_4_query=clusterAlg_2_all_result_query_edges[algorithm_k][0]
                # we have obsereved that some times one column of query is aligned with more than one column of a datalake table;
                # here before writing the alignment I would delete the extra alignments with this logic: 
                # just keep the fist one  "randomly" 
                final_alignment_4_query,queries_with_problematic_alignments =remove_problematic_alignments(final_alignment_4_query,queries_with_problematic_alignments, track_columns_reverse)                    
                export_alignment_to_csv(final_alignment_4_query, track_columns_reverse,output_file )
                print("-------------------------------------")



        
                
        end_time = time.time_ns()
        total_time = int(end_time - start_time)/ 10 **9

        print("Total time for exporting Dust Alignments: ", str(total_time))
        print("-------------------------------------")
        print("----------probmeletic alignments for queries-------------------")
        print("number of queries with probmeletic alignments whihc are corrected"+str(len(queries_with_problematic_alignments)))
        total_1 = 0
        for key, value in queries_with_problematic_alignments.items():
          total_1 +=  len(value)
        print("number of all columns with problematic alignmnets: which are corrected"+str(total_1))


# fasttext_model = ft.get_embedding_model()        

# %%
# sbert_model = SentenceTransformer('bert-base-uncased') #case insensitive model. BOSTON and boston have the same embedding.
# sbert_tokenizer = ""

# %%
# bert_model = BertModel.from_pretrained('bert-base-uncased')
# bert_model = BertClassifierPretrained(bert_model) 
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



# %%
tfidf_vectorizer = TfidfVectorizer()

# %%
def getColumnType(attribute, column_threshold=0.5, entity_threshold=0.5):
    strAttribute = [item for item in attribute if type(item) == str]
    strAtt = [item for item in strAttribute if not item.isdigit()]
    for i in range(len(strAtt)-1, -1, -1):
        entity = strAtt[i]
        num_count = 0
        for char in entity:
            if char.isdigit():
                num_count += 1
        if num_count/len(entity) > entity_threshold:
            del strAtt[i]            
    if len(strAtt)/len(attribute) > column_threshold:
        return 1
    else:
        return 0

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def findsubsets(s, n):
    return list(itertools.combinations(s, n))

# Function to select tokens based on TF-IDF for each column
def select_values_by_tfidf(values, num_tokens_to_select = 512):
    # tfidf_vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(values)
        tfidf_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        sorted_indices = np.argsort(tfidf_scores)[::-1]
        sorted_values = [values[i] for i in sorted_indices]
        return sorted_values[:num_tokens_to_select]
    except ValueError:
        # Handle the case where TF-IDF scores cannot be computed
        print("Returning the same values.")
        if len(values) <= num_tokens_to_select:
            return values
        else:
            return random.sample(values, num_tokens_to_select)
        
def get_glove_embeddings(column_data, sample_size = 50000, sim_threshold = 0.05):     
    sample1_list = random.sample(column_data, min(sample_size, len(column_data)))
    sample1_embeddings = model.transform(sample1_list).reshape(1,-1).flatten()
    column_data = list(set(column_data) - set(sample1_list))
    while(len(column_data) > 0):
        sample2_list = random.sample(column_data, min(sample_size, len(column_data)))
        sample2_embeddings = model.transform(sample2_list).reshape(1,-1).flatten()    
        column_data = list(set(column_data) - set(sample2_list))
        if sample2_embeddings.size == 0:
            continue
        elif sample1_embeddings.size == 0:
            sample1_embeddings = sample2_embeddings
            continue
        else:
            cosine = utl.CosineSimilarity(sample1_embeddings, sample2_embeddings)
            sample1_embeddings = (sample1_embeddings + sample2_embeddings) / 2
            if cosine >= (1 - sim_threshold):
                break
    if sample1_embeddings.size == 0:
        sample1_embeddings = np.random.uniform(-1, 1, 300).astype(np.float32) #glove embedding is 300 vector long             
    return sample1_embeddings

def get_fasttext_embeddings(column_data, sample_size = 50000, sim_threshold = 0.05):     
    sample1_list = random.sample(column_data, min(sample_size, len(column_data)))
    sample1_embeddings = ft.get_fasttext_embeddings(model, sample1_list).reshape(1,-1).flatten() # glove_model.transform(sample1_list).reshape(1,-1)
    column_data = list(set(column_data) - set(sample1_list))
    while(len(column_data) > 0):
        sample2_list = random.sample(column_data, min(sample_size, len(column_data)))
        sample2_embeddings = ft.get_fasttext_embeddings(model, sample2_list).reshape(1,-1).flatten()   
        column_data = list(set(column_data) - set(sample2_list))
        if sample2_embeddings.size == 0:
            continue
        elif sample1_embeddings.size == 0:
            sample1_embeddings = sample2_embeddings
            continue
        else:
            cosine = utl.CosineSimilarity(sample1_embeddings, sample2_embeddings)
            sample1_embeddings = (sample1_embeddings + sample2_embeddings) / 2
            if cosine >= (1 - sim_threshold):
                break
    if sample1_embeddings.size == 0:
        sample1_embeddings = np.random.uniform(-1, 1, 300).astype(np.float32) #glove embedding is 300 vector long     
        # print(sample1_embeddings)        
    return sample1_embeddings
# ft.get_fasttext_embeddings(model, column_data).reshape(1,-1)

def get_sentence_bert_embeddings(column_data, sample_size = 50000, sim_threshold = 0.05): 
    # embedding each column as a table.  
    # print("Column data:", column_data)   
    sample1_embeddings = utl.EmbedTable(column_data, embedding_type="sentence_bert", model=model, tokenizer=tokenizer)[0]
    
    # print("type: ", type(sample1_embeddings)) #.shape()) #.shape())
    # print(sample1_embeddings)
    if np.isnan(sample1_embeddings).any():
        # print("sample embedding is nan")
        sample1_embeddings = np.random.uniform(-1, 1, 768).astype(np.float32) #sbert embedding is 768 vector long     
    # print("len:", len(sample1_embeddings))
    return sample1_embeddings

def get_bert_embeddings(column_data, sample_size = 50000, sim_threshold = 0.05, embedding_type = "bert"): 
    # embedding each column as a table.  
    # print("Column data:", column_data)   
    sample1_embeddings = utl.EmbedTable(column_data, embedding_type=embedding_type, model=model, tokenizer=tokenizer)[0]
    
    # print("type: ", type(sample1_embeddings)) #.shape()) #.shape())
    # print(sample1_embeddings)
    if np.isnan(sample1_embeddings).any():
        # print("sample embedding is nan")
        sample1_embeddings = np.random.uniform(-1, 1, 768).astype(np.float32) #sbert embedding is 768 vector long     
    # print("len:", len(sample1_embeddings))
    return sample1_embeddings


def get_sentence_bert_embeddings_serialize(column_data, sample_size = 512, sim_threshold = 0.05): 
    selected_tokens = select_values_by_tfidf(column_data, num_tokens_to_select = sample_size)
    selected_tokens = ' '.join(selected_tokens)
    sample1_embeddings = utl.EmbedTable([selected_tokens], embedding_type="sentence_bert", model=model, tokenizer=tokenizer)[0]
    # print("sample1 embeddings: ", sample1_embeddings)
    if np.isnan(sample1_embeddings).any():
        sample1_embeddings = np.random.uniform(-1, 1, 768).astype(np.float32) #sbert embedding is 768 vector long     
    return sample1_embeddings

def get_bert_embeddings_serialize(column_data, sample_size = 512, sim_threshold = 0.05, embedding_type = "bert"): 
    selected_tokens = select_values_by_tfidf(column_data, num_tokens_to_select = sample_size)
    selected_tokens = ' '.join(selected_tokens)

    
    sample1_embeddings = utl.EmbedTable([selected_tokens], embedding_type=embedding_type, model=model, tokenizer=tokenizer)[0]
    # print("sample1 embeddings: ", sample1_embeddings)
    if np.isnan(sample1_embeddings).any():
        sample1_embeddings = np.random.uniform(-1, 1, 768).astype(np.float32) #sbert embedding is 768 vector long     
    return sample1_embeddings



def get_starmie_embeddings(column_key):
    return starmie_embeddings[column_key]

# def get_dust_embeddings(column_data, sample_size = 50000, sim_threshold = 0.05, embedding_type = "dust"): 
#     # embedding each column as a table.  
#     # print("Column data:", column_data)   
#     sample1_embeddings = utl.EmbedTable(column_data, embedding_type=embedding_type, model=model, tokenizer=tokenizer)[0]
    
#     # print("type: ", type(sample1_embeddings)) #.shape()) #.shape())
#     # print(sample1_embeddings)
#     if np.isnan(sample1_embeddings).any():
#         # print("sample embedding is nan")
#         sample1_embeddings = np.random.uniform(-1, 1, 768).astype(np.float32) #sbert embedding is 768 vector long     
#     # print("len:", len(sample1_embeddings))
#     return sample1_embeddings

# collect the columns within the whole data lake for lsh ensemble.
def compute_embeddings(table_path_list, embedding_type, max_col_size = 10000, use_numeric_columns = True):
    computed_embeddings = {}
    using_tables = 0
    zero_columns = 0
    numeric_columns = 0
    # print(table_path_list)
    for file in table_path_list:
        #try:
        df = utl.read_csv_file(file)
        if len(df) < 3:
            continue
        else:
            using_tables += 1
        # df = pd.read_csv(file, encoding = "latin-1", on_bad_lines="skip", lineterminator="\n")
        table_name = file.rsplit(os.sep,1)[-1]
        for idx, column in enumerate(df.columns):
            column_data = list(set(df[column].map(str)))
            if use_numeric_columns == False and getColumnType(column_data) == 0:
                numeric_columns += 1
                continue
            column_data = random.sample(column_data, min(len(column_data), max_col_size))
            all_text = ' '.join(column_data)
            # .join(map(str.lower, original_list))
            all_text = re.sub(r'\([^)]*\)', '', all_text)
            column_data = list(set(re.sub(r"[^a-z0-9]+", " ", all_text.lower()).split()))
            if len(column_data) == 0:
                zero_columns += 1
                continue
            # print("Total column values:", len(column_data))                
            if embedding_type == "glove":
                this_embedding = get_glove_embeddings(column_data)
            elif embedding_type == "fasttext":
                this_embedding = get_fasttext_embeddings(column_data)
            elif embedding_type == "sentence_bert":
                this_embedding = get_sentence_bert_embeddings(column_data)
            elif embedding_type == "sentence_bert_serialized":
                this_embedding = get_sentence_bert_embeddings_serialize(column_data)
            elif embedding_type == "bert" or embedding_type == "roberta" or embedding_type == "dust":
                this_embedding = get_bert_embeddings(column_data, embedding_type = embedding_type)
            elif embedding_type == "bert_serialized" or embedding_type == "roberta_serialized" or embedding_type == "dust_serialized":
                this_embedding = get_bert_embeddings_serialize(column_data, embedding_type = embedding_type)
            elif embedding_type == "starmie":
                this_embedding = get_starmie_embeddings((table_name, str(column).strip()))
            computed_embeddings[(table_name, column,idx)] = this_embedding
    # print(computed_embeddings)
            # this_fasttext_embedding = ft.get_fasttext_embeddings(fasttext_model, column_data).reshape(1,-1)
            # fasttext_embedding_dict[(table_name, column)] = this_fasttext_embedding
        # except Exception as e:
        #     print("Error in table:", table_name)
        #     print(e)
    # print(f"Computed embeddings. Total cols: {len(glove_embedding_dict)}")
    print(f"Embedded {using_tables} table(s), {len(computed_embeddings)} column(s). Zero column(s): {zero_columns}. Numeric Column(s): {numeric_columns}")
    return computed_embeddings



def compute_embeddings_single_table(table_path, embedding_type, max_col_size = 10000, use_numeric_columns = True):
    

    computed_embeddings = {}
    using_tables = 0
    zero_columns = 0
    numeric_columns = 0

    df = utl.read_csv_file(table_path)
    using_tables += 1
    # df = pd.read_csv(file, encoding = "latin-1", on_bad_lines="skip", lineterminator="\n")
    table_name = table_path.rsplit(os.sep,1)[-1]
    for idx, column in enumerate(df.columns):
        column_data = list(set(df[column].map(str)))
        if use_numeric_columns == False and getColumnType(column_data) == 0:
            numeric_columns += 1
            continue
        column_data = random.sample(column_data, min(len(column_data), max_col_size))
        all_text = ' '.join(column_data)
        # .join(map(str.lower, original_list))
        all_text = re.sub(r'\([^)]*\)', '', all_text)
        column_data = list(set(re.sub(r"[^a-z0-9]+", " ", all_text.lower()).split()))
        if len(column_data) == 0:
            zero_columns += 1
            continue
        # print("Total column values:", len(column_data))                
        if embedding_type == "glove":
            this_embedding = get_glove_embeddings(column_data)
        elif embedding_type == "fasttext":
            this_embedding = get_fasttext_embeddings(column_data)
        elif embedding_type == "sentence_bert":
            this_embedding = get_sentence_bert_embeddings(column_data)
        elif embedding_type == "sentence_bert_serialized":
            this_embedding = get_sentence_bert_embeddings_serialize(column_data)
        elif embedding_type == "bert" or embedding_type == "roberta" or embedding_type == "dust":
            this_embedding = get_bert_embeddings(column_data, embedding_type = embedding_type)
        elif embedding_type == "bert_serialized" or embedding_type == "roberta_serialized" or embedding_type == "dust_serialized":
            this_embedding = get_bert_embeddings_serialize(column_data, embedding_type = embedding_type)
        elif embedding_type == "starmie":
            this_embedding = get_starmie_embeddings((table_name, str(column).strip()))
        computed_embeddings[(table_name, column,idx)] = this_embedding
    # print(computed_embeddings)
            # this_fasttext_embedding = ft.get_fasttext_embeddings(fasttext_model, column_data).reshape(1,-1)
            # fasttext_embedding_dict[(table_name, column)] = this_fasttext_embedding
        # except Exception as e:
        #     print("Error in table:", table_name)
        #     print(e)
    # print(f"Computed embeddings. Total cols: {len(glove_embedding_dict)}")
    print(f"Embedded {using_tables} table(s), {len(computed_embeddings)} column(s). Zero column(s): {zero_columns}. Numeric Column(s): {numeric_columns}")
    return computed_embeddings

def print_metrics(precision, recall, f_measure):
    total_precision = 0
    total_recall = 0
    total_f_measure = 0
    average_precision = 0
    average_recall = 0
    average_f_measure = 0

    for item in precision:
        total_precision += precision[item]
    for item in recall:
        total_recall += recall[item]
    for item in f_measure:
        total_f_measure += f_measure[item]

    average_precision = total_precision / len(precision)
    average_recall = total_recall / len(recall)
    average_f_measure = total_f_measure / len(f_measure)
    print("Average precision:", average_precision)
    print("Average recall", average_recall)
    print("Average f measure", average_f_measure)

def plot_accuracy_range(original_dict, embedding_type, benchmark_name, metric = "F1-score", range_val = 0.1, save = False, save_folder = r"plots_align"):
    # Initialize a new dictionary to store ranges and count
    # range_count_dict = {'0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0}
    # range_count_dict = {'0.0-0.1': 0, '0.1-0.2': 0, '0.2-0.3': 0, '0.3-0.4': 0, '0.4-0.5': 0, '0.5-0.6': 0, '0.6-0.7': 0, '0.7-0.8': 0, '0.8-0.9': 0, '0.9-1.0': 0}
    # Initialize an empty dictionary
    save_name = f"{save_folder + os.sep}_{metric}_{benchmark_name}_{embedding_type}"
    range_count_dict = {}

    # Create keys with ranges from 0 to 1 with 0.1 difference and set values to 0
    for i in range(0,10):
        start_range = i / 10.0
        end_range = (i + 1) / 10.0
        key = f'{start_range:.1f}-{end_range:.1f}'
        range_count_dict[key] = 0
    # Iterate through the original dictionary values
    for value in original_dict.values():
        # Determine the range for the current value
        for key_range in range_count_dict:
            range_start, range_end = map(float, key_range.split('-'))
            if range_start <= value < range_end:
                # Increment the count for the corresponding range
                range_count_dict[key_range] += 1
                break  # Exit the loop once the range is found

    # Extract data for plotting
    ranges = list(range_count_dict.keys())
    counts = list(range_count_dict.values())

    plt.figure(figsize=(10, 6))  # Adjust the width and height as needed


    # Plotting the bar graph
    plt.bar(ranges, counts, color='blue')

    # Adding labels and title
    plt.xlabel('Ranges')
    plt.ylabel('Number of Tables')
    plt.title(f'{embedding_type} F1-score in {benchmark_name}')

    # Adding text on top of each bar
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    if save == True:
        plt.savefig(save_name)
    else:
        plt.show()
    plt.clf()




import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_tsne_subset_labels_columns_only(
    x, 
    track_columns_reverse, 
    random_seed=42, 
    perplexity=30,
    num_labels=50
):
    """
    Visualize column embeddings in 2D using t-SNE.
    - Colors each point by the table name (distinct hue per table).
    - Labels each point with ONLY the column name (no table prefix).
    - Labels only a random subset of points to reduce clutter.

    Parameters
    ----------
    x : np.ndarray
        (n_columns, embedding_dim) array of column embeddings.
    track_columns_reverse : dict
        Maps each row index i -> (table_name, column_name, col_idx).
    random_seed : int
        Random seed for reproducibility in t-SNE and subset sampling.
    perplexity : int
        The perplexity parameter for t-SNE (typical range [5..50]).
    num_labels : int
        How many points to label with text (randomly chosen).
        For many columns, labeling all can lead to excessive overlap.
    """

    # 1. Fit t-SNE to get 2D embeddings
    tsne = TSNE(
        n_components=2, 
        random_state=random_seed, 
        perplexity=perplexity, 
        init='pca'
    )
    x_2d = tsne.fit_transform(x)  # shape: (n_columns, 2)

    # 2. Collect table info & column names
    table_names = []
    short_labels = []
    for i in range(x.shape[0]):
        table_name, col_name, _ = track_columns_reverse[i]
        table_names.append(table_name)
        # Label with only the column name (omit table)
        short_labels.append(col_name)

    unique_tables = sorted(set(table_names))
    n_tables = len(unique_tables)

    # Create a list of distinct colors spaced around the HSV color wheel
    # endpoint=False to avoid wrapping hue=1 (which is same as hue=0).
    colors = plt.cm.hsv(np.linspace(0, 1, n_tables, endpoint=False))

    # Map each table name to a color
    color_map = {}
    for i, tname in enumerate(unique_tables):
        color_map[tname] = colors[i]

    # 3. Create the figure
    plt.figure(figsize=(16, 12))
    np.random.seed(random_seed)

    # 4. Plot all points, colored by table
    for i in range(x.shape[0]):
        tname = table_names[i]
        plt.scatter(
            x_2d[i, 0],
            x_2d[i, 1],
            c=[color_map[tname]],
            alpha=0.7,
            s=20,
            edgecolors='k',
            zorder=2
        )

    # 5. Randomly pick which points to label
    n_points = x.shape[0]
    if num_labels > n_points:
        num_labels = n_points
    indices_to_label = np.random.choice(range(n_points), size=num_labels, replace=False)

    # 6. Annotate ONLY the chosen subset (column name only)
    for i in indices_to_label:
        plt.annotate(
            short_labels[i],
            xy=(x_2d[i, 0], x_2d[i, 1]),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=8,
            color='black',
            zorder=3
        )

    # 7. Build a legend (one entry per table)
    handles = []
    for tname in unique_tables:
        handles.append(
            plt.Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=color_map[tname],
                markeredgecolor='k',
                label=tname,
                markersize=8
            )
        )
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("t-SNE of Column Embeddings (Column Name Labels Only)")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    plt.show()

import numpy as np
from scipy.spatial.distance import cosine

def build_distance_matrix(column_embeddings, track_columns_reverse):
    """
    Build a precomputed distance matrix for hierarchical clustering.
    If two columns come from the same table, their distance is set to a huge value (1e9).
    Otherwise, use 1 - cosine_similarity as the distance.

    Parameters
    ----------
    column_embeddings : list or np.ndarray
        shape (n_columns, embedding_dim). The embedding for each column.
    track_columns_reverse : dict
        Maps each integer column index i -> (table_name, column_name, col_idx).
        E.g. track_columns_reverse[0] = ("tableA.csv", "colA", 0)

    Returns
    -------
    dist_matrix : np.ndarray
        shape (n_columns, n_columns), storing pairwise distances.
    """
    n_cols = len(column_embeddings)
    dist_matrix = np.zeros((n_cols, n_cols), dtype=np.float32)

    for i in range(n_cols):
        for j in range(n_cols):
            if i == j:
                dist_matrix[i, j] = 0.0
            else:
                # Check if both columns are from the same table
                table_i = track_columns_reverse[i][0]
                table_j = track_columns_reverse[j][0]
                if table_i == table_j:
                    # "Infinite" distance => cannot merge
                    dist_matrix[i, j] = 1e9
                else:
                    # Actual distance; e.g., 1 - cosine_similarity
                    v1 = column_embeddings[i]
                    v2 = column_embeddings[j]
                    cos_sim = 1 - cosine(v1, v2)   # This is the cosine similarity
                    dist_matrix[i, j] = 1 - cos_sim  # so distance = 1 - similarity
    return dist_matrix


def main():

    dataset="tus_benchmark"
    if dataset==  "santos small":   
        excluded_queries={"workforce_management_information_a.csv", "workforce_management_information_b.csv" }
        #benchmark_name = "tus_benchmark" 
        benchmark_name = "santos" 
        dl_table_folder = r"data" + os.sep + benchmark_name + os.sep + "datalake"
        query_table_folder = r"data" + os.sep + benchmark_name + os.sep + "query"
        #groundtruth_file = r"groundtruth" + os.sep + benchmark_name + "_union_groundtruth.pickle"
        groundtruth_file="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/top_20_Starmie_output_diluted_restricted_noscore.pkl"
        outputfile= f"DUST_Alignment_4gtruth_{benchmark_name}.csv"

            
        
    
    elif  dataset==  "ugen_v2":   
        excluded_queries={ }
        #benchmark_name = "tus_benchmark" 
        benchmark_name = "ugen_v2" 
        dl_table_folder = r"data" + os.sep + benchmark_name + os.sep + "datalake"
        query_table_folder = r"data" + os.sep + benchmark_name + os.sep + "query"
        groundtruth_file="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/ugen_v2/manual_benchmark_validation_results/ugen_v2_eval_groundtruth.csv"
        outputfile= f"DUST_Alignment_4gtruth_{benchmark_name}.csv"
  
           
    elif dataset==  "tus_benchmark":   
        excluded_queries={ }
        benchmark_name = "tus_benchmark" 
        dl_table_folder = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/table-union-search-benchmark/small/datalake"
        query_table_folder = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/table-union-search-benchmark/small/query"
        groundtruth_file="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/table-union-search-benchmark/small/tus_small_noverlap_groundtruth.csv"
        outputfile= f"/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/table-union-search-benchmark/small/DUST_Alignment_4gtruth_{benchmark_name}.csv"

    DUST_alignmnet_export(outputfile,dl_table_folder,query_table_folder, benchmark_name,groundtruth_file, excluded_queries)
    print(f"DUST Alignment Export for {dataset} completed.")
    
        
def initialize_globally():
    global random_seed, embedding_type, model, tokenizer, vec_length, tfidf_vectorizer

    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    embedding_type = 'roberta_serialized'
    
    model = RobertaModel.from_pretrained("roberta-base")
    model = BertClassifierPretrained(model)
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    vec_length = 768
    tfidf_vectorizer = TfidfVectorizer()
    
if __name__ == "__main__":
        
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    embedding_type = 'roberta_serialized'
    
    model = RobertaModel.from_pretrained("roberta-base")
    model = BertClassifierPretrained(model)
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    vec_length = 768
    tfidf_vectorizer = TfidfVectorizer()
    main()