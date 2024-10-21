import numpy as np
import random
import pickle
import argparse
import mlflow
from naive_search_Novelty import NaiveSearcherNovelty
from checkPrecisionRecall import saveDictionaryAsPickleFile, calcMetrics
import time

def generate_random_table(nrow, ncol):
    return np.random.rand(nrow, ncol)

def verify(table1, table2,threshold=0.6):
    score = 0.0
    nrow = len(table1)
    ncol = len(table2)
    graph = np.zeros(shape=(nrow,ncol),dtype=float)
    for i in range(nrow):
        for j in range(ncol):
            sim = cosine_sim(table1[i],table2[j])
            if sim > threshold:
                graph[i,j] = sim
    max_graph = make_cost_matrix(graph, lambda cost: (graph.max() - cost) if (cost != DISALLOWED) else DISALLOWED)
    m = Munkres()
    indexes = m.compute(max_graph)
    for row,col in indexes:
        score += graph[row,col]
    return score,indexes

def generate_test_data(num, ndim):
    # for test only: randomly generate tables and 2 queries
    # num: the number of tables in the dataset; ndim: dimension of column vectors
    tables = []
    queries = []
    for i in range(num):
        ncol = random.randint(2,9)
        tbl = generate_random_table(ncol, ndim)
        tables.append((i,tbl))
    for j in range(2):
        ncol = random.randint(2,9)
        tbl = generate_random_table(ncol, ndim)
        queries.append((j+num,tbl))
    return tables, queries
import os
import csv


def write_dict_to_csv_file(dictionary, file_path):
    """
    Writes a dictionary to a CSV file.

    Assumes that the dictionary keys are column headers and the values are lists of column data.

    Parameters:
        dictionary (dict): The dictionary to write.
        file_path (str): The path to the CSV file where the dictionary will be saved.
    """
    # Get the headers from the dictionary keys
    headers = dictionary.keys()
    # Determine the number of rows
    num_rows = len(next(iter(dictionary.values())))
    
    # Write to CSV file
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for i in range(num_rows):
            row = {key: dictionary[key][i] for key in headers}
            writer.writerow(row)


#if __name__ == '__main__':
def main(args2=None):
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="sato", choices=['sherlock', 'sato', 'cl', 'tapex'])
    parser.add_argument("--benchmark", type=str, default='santos')
    parser.add_argument("--augment_op", type=str, default="drop_col")
    parser.add_argument("--sample_meth", type=str, default="tfidf_entity")
    # matching is the type of matching
    parser.add_argument("--matching", type=str, default='exact') #exact or bounds (or greedy)
    parser.add_argument("--table_order", type=str, default="column")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--single_column", dest="single_column", action="store_true")
    # For error analysis
    parser.add_argument("--bucket", type=int, default=0) # the error analysis has 5 equally-sized buckets
    parser.add_argument("--analysis", type=str, default='col') # 'col', 'row', 'numeric'
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.6)
    # For Scalability experiments
    parser.add_argument("--scal", type=float, default=1.00)
    # mlflow tag
    parser.add_argument("--mlflow_tag", type=str, default=None)
    parser.add_argument("--penalty", type=float, default=0.0) # penalization parameter for novetly a value between [0,1)


    hp = parser.parse_args(args=args2)

    # mlflow logging
    for variable in ["encoder", "benchmark", "augment_op", "sample_meth", "matching", "table_order", "run_id", "single_column", "K", "threshold", "scal", "penalty"]:
        mlflow.log_param(variable, getattr(hp, variable))

    if hp.mlflow_tag:
        mlflow.set_tag("tag", hp.mlflow_tag)

    dataFolder = hp.benchmark

    # If the filepath to the pkl files are different, change here:
    if hp.encoder == 'cl':
        query_path = "data/"+dataFolder+"/vectors/"+hp.encoder+"_query_"+hp.augment_op+"_"+hp.sample_meth+"_"+hp.table_order+"_"+str(hp.run_id)
        table_path = "data/"+dataFolder+"/vectors/"+hp.encoder+"_datalake_"+hp.augment_op+"_"+hp.sample_meth+"_"+hp.table_order+"_"+str(hp.run_id)
        query_path_raw = "data/"+dataFolder+"/"+"query"
        table_path_raw = "data/"+dataFolder+"/"+"datalake"
        

        if hp.single_column:
            query_path += "_singleCol"
            table_path += "_singleCol"
        query_path += ".pkl"
        table_path += ".pkl"
    else:
        query_path = "data/"+dataFolder+"/"+hp.encoder+"_query.pkl"
        table_path = "data/"+dataFolder+"/"+hp.encoder+"_datalake.pkl"

    # Load the query file
    qfile = open(query_path,"rb")
    queries = pickle.load(qfile)
    queries_raw=NaiveSearcherNovelty.read_csv_files_to_dict(query_path_raw)
    # queries is a list of tuples ; tuple of (str(filename), numpy.ndarray(vectors(numpy.ndarray) for columns) 
    #I assume that the order  of columns are comming from the original csv files 

            
       
    print("Number of queries: %d" % (len(queries)))
    
    qfile.close()
    # Call NaiveSearcher, which has linear search and bounds search, from naive_search.py
    searcher = NaiveSearcherNovelty(table_path, hp.scal,table_path_raw)
    returnedResults = {}
    start_time = time.time()
    # For error analysis of tables
    analysis = hp.analysis
    # bucketFile = open("data/"+dataFolder+"/buckets/query_"+analysis+"Bucket_"+str(hp.bucket)+".txt", "r")
    # bucket = bucketFile.read()
    queries.sort(key = lambda x: x[0])
    query_times = []
    qCount = 0
    for query in queries:
            qCount += 1
            if qCount % 10 == 0:
                print("Processing query ",qCount, " of ", len(queries), " total queries.")
        # if query[0] in bucket:
            query_start_time = time.time()
            #get the query raw content 
            raw_query=queries_raw[query[0]]
            if(len(raw_query)!= len(query[1])):
                print("raise an exception stating that the files do not match")
            
            if hp.matching == 'exact':
                penalty=0.2
                qres = searcher.topk(hp.encoder,raw_query, query, hp.K, hp.penalty,threshold=hp.threshold)
            else: # Bounds matching
                qres = searcher.topk_bounds(hp.encoder, query, hp.K, threshold=hp.threshold)
            res = []
            for tpl in qres:
                tmp = (tpl[0],tpl[1])
                res.append(tmp)
            returnedResults[query[0]] = [r[1] for r in res]
            query_times.append(time.time() - query_start_time)
    write_dict_to_csv_file(returnedResults,'penalty'+str(penalty)+'.csv')
    print("Average QUERY TIME: %s seconds " % (sum(query_times)/len(query_times)))
    print("10th percentile: ", np.percentile(query_times, 10), " 90th percentile: ", np.percentile(query_times, 90))
    print("--- Total Query Time: %s seconds ---" % (time.time() - start_time))

    # santosLarge and WDC benchmarks are used for efficiency
    if hp.benchmark == 'santosLarge' or hp.benchmark == 'wdc':
        print("No groundtruth for %s benchmark" % (hp.benchmark))
    else:
        # Calculating effectiveness scores (Change the paths to where the ground truths are stored)
        if 'santos' in hp.benchmark:
            k_range = 1
            groundTruth = "data/santos/santosUnionBenchmark.pickle"
        else:
            k_range = 10
            if hp.benchmark == 'tus':
                groundTruth = 'data/table-union-search-benchmark/small/tus-groundtruth/tusLabeledtusUnionBenchmark'
            elif hp.benchmark == 'tusLarge':
                groundTruth = 'data/table-union-search-benchmark/large/tus-groundtruth/tusLabeledtusLargeUnionBenchmark'

       
        calcMetrics(hp.K, k_range, returnedResults, gtPath=groundTruth)
        
if __name__ == '__main__':
    main()        
 
