#!/usr/bin/env python
"""
preprocess_align.py

This script processes query tables to compute column alignments between a query and a data lake.
The per‐query alignment phase is parallelized using Python’s multiprocessing.
All helper functions (such as compute_embeddings, findsubsets, cop_kmeans, export_alignment_to_csv, etc.)
are assumed to exist as in your current code.
"""

import os
import glob
import csv
import time
import random
import itertools
import re
import json, shutil
import string
import numpy as np
import torch
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
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import itertools
from sklearn import metrics
# from glove_embeddings import GloveTransformer
# import fasttext_embeddings as ft
import matplotlib.pyplot as plt
from model_classes import BertClassifierPretrained, BertClassifier
import csv
from copkmeans.cop_kmeans2 import cop_kmeans
from multiprocessing import Pool, cpu_count

# ---------------------------- Global Parameters ----------------------------

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

available_embeddings = ["bert", "bert_serialized", "roberta", "roberta_serialized",
                        "sentence_bert", "sentence_bert_serialized", "glove", "fasttext",
                        "dust", "dust_serialized", "starmie"]

embedding_type = available_embeddings[3]  # e.g., "roberta_serialized"
use_numeric_columns = True
benchmark_name = "table-union-search-benchmark" 
clustering_metric = "cosine"  # for silhouette scoring
single_col = 0

# dl_table_folder = os.path.join("data", benchmark_name, "datalake")
# query_table_folder = os.path.join("data", benchmark_name, "query")
dl_table_folder = "/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/datalake"
query_table_folder = "/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/query"
groundtruth_file="/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/tus_small_noverlap_groundtruth_notdilute.csv"
#groundtruth_file = os.path.join("data", benchmark_name,benchmark_name + "_union_groundtruth_diluted.pickle")
output_file = "/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/tus_CL_KMEANS_cosine_alignment.csv"
align_plot_folder = os.path.join("plots_align")

query_tables = glob.glob(os.path.join(query_table_folder, "*.csv"))
groundtruth = utl.loadDictionaryFromPickleFile(groundtruth_file)

tfidf_vectorizer = TfidfVectorizer()

# ---------------------------- Embedding Selection ----------------------------
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
    model = SentenceTransformer('bert-base-uncased')  # case-insensitive model; "BOSTON" and "boston" have the same embedding.
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
    model = BertClassifier(model, num_labels=2, hidden_size=768, output_size=768)
    model = DataParallel(model, device_ids=[0, 1, 2, 3]) 
    model.load_state_dict(torch.load(model_path))
elif embedding_type == "starmie":
    if single_col == 0:
        dl_column_embeddings = utl.loadDictionaryFromPickleFile(os.path.join("starmie_embeddings", benchmark_name + "_vectors", "cl_datalake_drop_col_tfidf_entity_column_0.pkl"))
        query_column_embeddings = utl.loadDictionaryFromPickleFile(os.path.join("starmie_embeddings", benchmark_name + "_vectors", "cl_query_drop_col_tfidf_entity_column_0.pkl"))
    else:
        dl_column_embeddings = utl.loadDictionaryFromPickleFile(os.path.join("starmie_embeddings", benchmark_name + "_vectors", "cl_datalake_drop_col_tfidf_entity_column_0_singleCol.pkl"))
        query_column_embeddings = utl.loadDictionaryFromPickleFile(os.path.join("starmie_embeddings", benchmark_name + "_vectors", "cl_query_drop_col_tfidf_entity_column_0_singleCol.pkl"))
    
    dl_column_embeddings = {key: value for key, value in dl_column_embeddings}
    query_column_embeddings = {key: value for key, value in query_column_embeddings}
    starmie_embeddings = {}
    for table in query_column_embeddings:
        if os.path.exists(os.path.join(query_table_folder, table)):
            table_df = utl.read_csv_file(os.path.join(query_table_folder, table))
            col_headers = [str(col).strip() for col in table_df.columns] 
            for idx, item in enumerate(col_headers):
                starmie_embeddings[(table, item)] = query_column_embeddings[table][idx]
    for table in dl_column_embeddings:
        table_df = utl.read_csv_file(os.path.join(dl_table_folder, table))
        col_headers = [str(col).strip() for col in table_df.columns] 
        for idx, item in enumerate(col_headers):
            starmie_embeddings[(table, item)] = dl_column_embeddings[table][idx]
else:
    print("invalid embedding type")
    sys.exit()

# ---------------------------- Helper Functions ----------------------------

def getColumnType(attribute, column_threshold=0.5, entity_threshold=0.5):
    strAttribute = [item for item in attribute if type(item)==str]
    strAtt = [item for item in strAttribute if not item.isdigit()]
    for i in range(len(strAtt)-1, -1, -1):
        entity = strAtt[i]
        num_count = sum(1 for char in entity if char.isdigit())
        if num_count/len(entity) > entity_threshold:
            del strAtt[i]
    return 1 if len(strAtt)/len(attribute) > column_threshold else 0

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

def findsubsets(s, n):
    return list(itertools.combinations(s, n))

def select_values_by_tfidf(values, num_tokens_to_select=512):
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(values)
        tfidf_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        sorted_indices = np.argsort(tfidf_scores)[::-1]
        sorted_values = [values[i] for i in sorted_indices]
        return sorted_values[:num_tokens_to_select]
    except ValueError:
        if len(values) <= num_tokens_to_select:
            return values
        else:
            return random.sample(values, num_tokens_to_select)

def get_glove_embeddings(column_data, sample_size=50000, sim_threshold=0.05):
    sample1_list = random.sample(column_data, min(sample_size, len(column_data)))
    sample1_embeddings = model.transform(sample1_list).reshape(1,-1).flatten()
    column_data = list(set(column_data) - set(sample1_list))
    while len(column_data) > 0:
        sample2_list = random.sample(column_data, min(sample_size, len(column_data)))
        sample2_embeddings = model.transform(sample2_list).reshape(1,-1).flatten()    
        column_data = list(set(column_data) - set(sample2_list))
        if sample2_embeddings.size == 0:
            continue
        elif sample1_embeddings.size == 0:
            sample1_embeddings = sample2_embeddings
        else:
            cosine = utl.CosineSimilarity(sample1_embeddings, sample2_embeddings)
            sample1_embeddings = (sample1_embeddings + sample2_embeddings) / 2
            if cosine >= (1 - sim_threshold):
                break
    if sample1_embeddings.size == 0:
        sample1_embeddings = np.random.uniform(-1, 1, 300).astype(np.float32)
    return sample1_embeddings

def get_fasttext_embeddings(column_data, sample_size=50000, sim_threshold=0.05):
    sample1_list = random.sample(column_data, min(sample_size, len(column_data)))
    sample1_embeddings = ft.get_fasttext_embeddings(model, sample1_list).reshape(1,-1).flatten()
    column_data = list(set(column_data) - set(sample1_list))
    while len(column_data) > 0:
        sample2_list = random.sample(column_data, min(sample_size, len(column_data)))
        sample2_embeddings = ft.get_fasttext_embeddings(model, sample2_list).reshape(1,-1).flatten()   
        column_data = list(set(column_data) - set(sample2_list))
        if sample2_embeddings.size == 0:
            continue
        elif sample1_embeddings.size == 0:
            sample1_embeddings = sample2_embeddings
        else:
            cosine = utl.CosineSimilarity(sample1_embeddings, sample2_embeddings)
            sample1_embeddings = (sample1_embeddings + sample2_embeddings) / 2
            if cosine >= (1 - sim_threshold):
                break
    if sample1_embeddings.size == 0:
        sample1_embeddings = np.random.uniform(-1, 1, 300).astype(np.float32)
    return sample1_embeddings

def get_sentence_bert_embeddings(column_data, sample_size=50000, sim_threshold=0.05):
    sample1_embeddings = utl.EmbedTable(column_data, embedding_type="sentence_bert", model=model, tokenizer=tokenizer)[0]
    if np.isnan(sample1_embeddings).any():
        sample1_embeddings = np.random.uniform(-1, 1, 768).astype(np.float32)
    return sample1_embeddings

def get_bert_embeddings(column_data, sample_size=50000, sim_threshold=0.05, embedding_type="bert"):
    sample1_embeddings = utl.EmbedTable(column_data, embedding_type=embedding_type, model=model, tokenizer=tokenizer)[0]
    if np.isnan(sample1_embeddings).any():
        sample1_embeddings = np.random.uniform(-1, 1, 768).astype(np.float32)
    return sample1_embeddings

def get_sentence_bert_embeddings_serialize(column_data, sample_size=512, sim_threshold=0.05):
    selected_tokens = select_values_by_tfidf(column_data, num_tokens_to_select=sample_size)
    selected_tokens = ' '.join(selected_tokens)
    sample1_embeddings = utl.EmbedTable([selected_tokens], embedding_type="sentence_bert", model=model, tokenizer=tokenizer)[0]
    if np.isnan(sample1_embeddings).any():
        sample1_embeddings = np.random.uniform(-1, 1, 768).astype(np.float32)
    return sample1_embeddings

def get_bert_embeddings_serialize(column_data, sample_size=512, sim_threshold=0.05, embedding_type="bert"):
    selected_tokens = select_values_by_tfidf(column_data, num_tokens_to_select=sample_size)
    selected_tokens = ' '.join(selected_tokens)
    sample1_embeddings = utl.EmbedTable([selected_tokens], embedding_type=embedding_type, model=model, tokenizer=tokenizer)[0]
    if np.isnan(sample1_embeddings).any():
        sample1_embeddings = np.random.uniform(-1, 1, 768).astype(np.float32)
    return sample1_embeddings

def get_starmie_embeddings(column_key):
    return starmie_embeddings[column_key]

def compute_embeddings(table_path_list, embedding_type, max_col_size=10000, use_numeric_columns=True):
    computed_embeddings = {}
    using_tables = 0
    zero_columns = 0
    numeric_columns = 0
    for file in table_path_list:
        df = utl.read_csv_file(file)
        if len(df) < 3:
            continue
        else:
            using_tables += 1
        table_name = file.rsplit(os.sep, 1)[-1]
        for idx, column in enumerate(df.columns):
            column_data = list(set(df[column].map(str)))
            if use_numeric_columns == False and getColumnType(column_data) == 0:
                numeric_columns += 1
                continue
            column_data = random.sample(column_data, min(len(column_data), max_col_size))
            all_text = ' '.join(column_data)
            all_text = re.sub(r'\([^)]*\)', '', all_text)
            column_data = list(set(re.sub(r"[^a-z0-9]+", " ", all_text.lower()).split()))
            if len(column_data) == 0:
                zero_columns += 1
                continue
            if embedding_type == "glove":
                this_embedding = get_glove_embeddings(column_data)
            elif embedding_type == "fasttext":
                this_embedding = get_fasttext_embeddings(column_data)
            elif embedding_type == "sentence_bert":
                this_embedding = get_sentence_bert_embeddings(column_data)
            elif embedding_type == "sentence_bert_serialized":
                this_embedding = get_sentence_bert_embeddings_serialize(column_data)
            elif embedding_type in ["bert", "roberta", "dust"]:
                this_embedding = get_bert_embeddings(column_data, embedding_type=embedding_type)
            elif embedding_type in ["bert_serialized", "roberta_serialized", "dust_serialized"]:
                this_embedding = get_bert_embeddings_serialize(column_data, embedding_type=embedding_type)
            elif embedding_type == "starmie":
                this_embedding = get_starmie_embeddings((table_name, str(column).strip()))
            computed_embeddings[(table_name, column, idx)] = this_embedding
    print(f"Embedded {using_tables} table(s), {len(computed_embeddings)} column(s). Zero column(s): {zero_columns}. Numeric Column(s): {numeric_columns}")
    return computed_embeddings

def print_metrics(precision, recall, f_measure):
    total_precision = sum(precision.values())
    total_recall = sum(recall.values())
    total_f_measure = sum(f_measure.values())
    average_precision = total_precision / len(precision)
    average_recall = total_recall / len(recall)
    average_f_measure = total_f_measure / len(f_measure)
    print("Average precision:", average_precision)
    print("Average recall:", average_recall)
    print("Average f measure:", average_f_measure)

def plot_accuracy_range(original_dict, embedding_type, benchmark_name, metric="F1-score", range_val=0.1, save=False, save_folder=align_plot_folder):
    save_name = os.path.join(save_folder, f"{metric}_{benchmark_name}_{embedding_type}.png")
    range_count_dict = {}
    for i in range(0, 10):
        start_range = i / 10.0
        end_range = (i + 1) / 10.0
        key = f'{start_range:.1f}-{end_range:.1f}'
        range_count_dict[key] = 0
    for value in original_dict.values():
        for key_range in range_count_dict:
            range_start, range_end = map(float, key_range.split('-'))
            if range_start <= value < range_end:
                range_count_dict[key_range] += 1
                break
    ranges = list(range_count_dict.keys())
    counts = list(range_count_dict.values())
    plt.figure(figsize=(10, 6))
    plt.bar(ranges, counts, color='blue')
    plt.xlabel('Ranges')
    plt.ylabel('Number of Tables')
    plt.title(f'{embedding_type} F1-score in {benchmark_name}')
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    if save:
        plt.savefig(save_name)
    else:
        plt.show()
    plt.clf()

def export_alignment_to_csv(final_alignment, track_columns_reverse, output_file):
    file_exists = os.path.exists(output_file)
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["query_table_name", "query_column", "query_column#", "dl_table_name", "dl_column#", "dl_column"])
        for row in final_alignment:
                    writer.writerow(row)
    print(f"Alignment exported to {output_file}")

# ---------------------------- Worker Function ----------------------------
def process_query(query_table):
    """
    Process a single query table.
    Returns a dictionary with keys:
      - "query_table_name": name of the query table,
      - "alignment": the final alignment (list of rows),
      - "track_columns_reverse": mapping from universal column id to (table, column, col_index),
      - "metrics": a dictionary with precision/recall values (optional).
    Returns None if the query is skipped.
    """
    query_table_name = query_table.rsplit(os.sep, 1)[-1]
    print(f"Processing {query_table_name}")
    if query_table_name in ["workforce_management_information_a.csv", "workforce_management_information_b.csv"]:
        return None
    if query_table_name not in groundtruth:
        return None

    # Prepare structures for query and datalake embeddings.
    column_embeddings = []
    track_tables = {}            # mapping: table name -> set of universal column indices
    track_columns = {}           # mapping: (table, column, idx) -> universal index
    track_columns_reverse = {}   # reverse mapping
    record_same_cluster = {}     # mapping: column identifier -> set of indices
    query_column_ids = set()
    idx = 0

    unionable_tables = groundtruth[query_table_name]
    query_embeddings = compute_embeddings([query_table], embedding_type, use_numeric_columns=use_numeric_columns)
    if len(query_embeddings) == 0:
        print(f"Not enough rows. Ignoring {query_table_name}")
        return None

    # Build unionable table paths.
    unionable_table_path = [os.path.join(dl_table_folder, tab) for tab in unionable_tables if tab != query_table_name]
    unionable_table_path = [path for path in unionable_table_path if os.path.exists(path)]
    if benchmark_name == "tus_benchmark":
        unionable_table_path = random.sample(unionable_table_path, min(10, len(unionable_table_path)))
    dl_embeddings = compute_embeddings(unionable_table_path, embedding_type, use_numeric_columns=use_numeric_columns)
    if len(dl_embeddings) == 0:
        print(f"Not enough rows in any data lake tables for {query_table_name}")
        return None

    # Process query embeddings.
    for col in query_embeddings:
        column_embeddings.append(query_embeddings[col])
        track_columns[col] = idx
        track_columns_reverse[idx] = col
        track_tables.setdefault(col[0], set()).add(idx)
        record_same_cluster.setdefault(col[1], set()).add(idx)
        query_column_ids.add(idx)
        idx += 1
    # Process data lake embeddings.
    for col in dl_embeddings:
        column_embeddings.append(dl_embeddings[col])
        track_columns[col] = idx
        track_columns_reverse[idx] = col
        track_tables.setdefault(col[0], set()).add(idx)
        record_same_cluster.setdefault(col[1], set()).add(idx)
        idx += 1

    # Build true edge sets.
    all_true_edges = set()
    all_true_query_edges = set()
    for key, indices in record_same_cluster.items():
        for s1 in indices:
            for s2 in indices:
                edge = tuple(sorted((s1, s2)))
                all_true_edges.add(edge)
                if s1 in query_column_ids or s2 in query_column_ids:
                    all_true_query_edges.add(edge)

    x = np.array(column_embeddings)

    # Build connectivity mask to prevent clustering columns from the same table.
    zero_positions = set()
    for table, indices in track_tables.items():
        for pair in itertools.combinations(indices, 2):
            zero_positions.add(pair)
    ncols = len(track_columns)
    arr = np.zeros((ncols, ncols))
    for i in range(ncols - 1):
        for j in range(i + 1, ncols):
            if ((i, j) not in zero_positions and (j, i) not in zero_positions and i != j):
                arr[i][j] = 1
                arr[j][i] = 1
    connectivity = csr_matrix(arr)

    # Determine candidate cluster counts.
    min_k = len(query_embeddings)
    max_k = 0
    for indices in track_tables.values():
        if len(indices) > min_k:
            min_k = len(indices)
        max_k += len(indices)

    all_distance = {}
    cluster_results = {}
    record_current_precision = {}
    record_current_recall = {}
    record_current_f_measure = {}
    record_current_query_precision = {}
    record_current_query_recall = {}
    record_current_query_f_measure = {}

    for k_candidate in range(min_k, max_k):
        # Run clustering using COP-KMeans (which is assumed to handle constraints internally)
        must_link = []
        cannot_link = list(zero_positions)
        try:
            labels, centers = cop_kmeans(dataset=x, k=k_candidate, ml=must_link, cl=cannot_link,distance_metric='cosine')
        except Exception as err:
            print(f"Error for k={k_candidate}: {err}")
            continue
        all_distance[k_candidate] = metrics.silhouette_score(x, labels)
        result_dict = {}
        for idx_val, lab in enumerate(labels):
            result_dict.setdefault(lab, set()).add(idx_val)
        all_result_edges = set()
        all_result_query_edges = set()
        for group in result_dict.values():
            for s1 in group:
                for s2 in group:
                    all_result_edges.add(tuple(sorted((s1, s2))))
                    if s1 in query_column_ids or s2 in query_column_ids:
                        all_result_query_edges.add(tuple(sorted((s1, s2))))
        cluster_results[k_candidate] = (all_result_query_edges, result_dict)

        # Compute precision and recall based on true edges.
        if all_result_edges:
            current_true_positive = len(all_true_edges.intersection(all_result_edges))
            current_precision = current_true_positive / len(all_result_edges)
            current_recall = current_true_positive / len(all_true_edges)
        else:
            current_precision = current_recall = 0

        if all_result_query_edges:
            current_query_true_positive = len(all_true_query_edges.intersection(all_result_query_edges))
            current_query_precision = current_query_true_positive / len(all_result_query_edges)
            current_query_recall = current_query_true_positive / len(all_true_query_edges)
        else:
            current_query_precision = current_query_recall = 0

        record_current_precision[k_candidate] = current_precision
        record_current_recall[k_candidate] = current_recall
        if (current_precision + current_recall) > 0:
            record_current_f_measure[k_candidate] = 2 * current_precision * current_recall / (current_precision + current_recall)
        else:
            record_current_f_measure[k_candidate] = 0
        record_current_query_precision[k_candidate] = current_query_precision
        record_current_query_recall[k_candidate] = current_query_recall
        if (current_query_precision + current_query_recall) > 0:
            record_current_query_f_measure[k_candidate] = 2 * current_query_precision * current_query_recall / (current_query_precision + current_query_recall)
        else:
            record_current_query_f_measure[k_candidate] = 0

    if not all_distance:
        print(f"No valid clustering found for {query_table_name}")
        return None

    best_k = max(all_distance, key=all_distance.get)
    final_alignment = cluster_results[best_k][0]

    # Build alignment list: each row is [query_table_name, query_column, query_column#, dl_table_name, dl_column#, dl_column]
    alignment_list = []
    for edge in final_alignment:
        col1 = track_columns_reverse[edge[0]]
        col2 = track_columns_reverse[edge[1]]
        alignment_list.append([col1[0], col1[1], col1[2], col2[0], col2[1], col2[2]])
    
    metrics_dict = {
        "precision": record_current_precision[best_k],
        "recall": record_current_recall[best_k],
        "f_measure": record_current_f_measure[best_k],
        "query_precision": record_current_query_precision[best_k],
        "query_recall": record_current_query_recall[best_k],
        "query_f_measure": record_current_query_f_measure[best_k]
    }
    return {"query_table_name": query_table_name,
            "alignment": alignment_list,
            "track_columns_reverse": track_columns_reverse,
            "metrics": metrics_dict}



def all_query_in_datalake(query_folder, datalake_folder):
        # Get a set of file names (base names) from the query folder
    query_files = {os.path.basename(f) for f in glob.glob(os.path.join(query_folder, "*")) if os.path.isfile(f)}
    
    # Get a set of file names (base names) from the data lake folder
    datalake_files = {os.path.basename(f) for f in glob.glob(os.path.join(datalake_folder, "*")) if os.path.isfile(f)}
    
    # Find files that are in the query folder but not in the data lake folder
    missing_files = query_files - datalake_files

    if missing_files:
        print("The following files in the query folder are missing in the data lake folder:")
        for file in sorted(missing_files):
            print(file)
        return False    
    else:
        print("All files in the query folder are present in the data lake folder.")
        return True


def verify_groungtruth(groundtruth_file, query_folder):
    query_to_datalake_dict = utl.loadDictionaryFromPickleFile(groundtruth_file)
        # Get a set of all file names (base names) in the query folder
    query_files = {os.path.basename(f) for f in glob.glob(os.path.join(query_folder, "*")) if os.path.isfile(f)}
    
    # Check 1: Ensure every file in the query folder is a key in the dictionary.
    missing_keys = [fname for fname in query_files if fname not in query_to_datalake_dict]
    if missing_keys:
        print("The following query files are missing as keys in the dictionary:")
        for key in missing_keys:
            print(f"  - {key}")
        return False    
    else:
        print("All query files are present as keys in the dictionary.")
    
    # Check 2: Ensure that for each key, the corresponding list contains that key.
    missing_self = [key for key, table_list in query_to_datalake_dict.items() if key not in table_list]
    if missing_self:
        print("\nThe following dictionary keys do not appear in their associated list:")
        for key in missing_self:
            print(f"  - {key} -> {query_to_datalake_dict[key]}")
            return False 
    else:
        print("All dictionary keys are mapped to lists that contain themselves.")
    return True
# ---------------------------- Main Multiprocessing Block ----------------------------
def main():
    
    queries_are_duplicated=all_query_in_datalake(query_table_folder,datalake_folder=dl_table_folder)
    if queries_are_duplicated:
        # make sure for every query you have mapping beween queries in the ground truth 
        grthrut_has_query=verify_groungtruth(groundtruth_file, query_table_folder)
        if grthrut_has_query:
            start_time = time.time()
            print(f"Found {len(query_tables)} query tables.")
            with Pool(processes=cpu_count()) as pool:
                results = pool.map(process_query, query_tables)
            valid_results = [r for r in results if r is not None]
            print(f"Processed {len(valid_results)} query tables successfully.")

            # Export alignments (serially) to avoid concurrent file writes.
            for res in valid_results:
                export_alignment_to_csv(res["alignment"], res["track_columns_reverse"], output_file)
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total processing time: {total_time:.2f} seconds")
            
            # Optionally, aggregate and print overall metrics here.
            final_query_precision = {}
            final_query_recall = {}
            final_query_f_measure = {}
            for res in valid_results:
                qname = res["query_table_name"]
                final_query_precision[qname] = res["metrics"]["query_precision"]
                final_query_recall[qname] = res["metrics"]["query_recall"]
                final_query_f_measure[qname] = res["metrics"]["query_f_measure"]
            print("Overall metrics (using query columns as ground truth):")
            print_metrics(final_query_precision, final_query_recall, final_query_f_measure)
            #plot_accuracy_range(final_query_f_measure, embedding_type, benchmark_name, metric="F1-score", save=True, save_folder=align_plot_folder)
        else:
             print("correct the groundtruth file")

    else: 
        #first copy the query into the datalake files
                
        print("copy queries to the data lake folder")
if __name__ == '__main__':
    main()