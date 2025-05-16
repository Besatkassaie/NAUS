import numpy as np
import random
import pickle
import argparse
import mlflow
from naive_search_export import NaiveSearcher
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


#if __name__ == '__main__':
# this class used to export the result from Starmie in two format one with score and one without score 
# then we 
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
    parser.add_argument("--restrict", type=int, default=0) # if is 1 means that restict your datalake only to unionables based on a ground truth 

    hp = parser.parse_args(args=args2)

    # mlflow logging
    for variable in ["encoder", "benchmark", "augment_op", "sample_meth", "matching", "table_order", "run_id", "single_column", "K", "threshold", "scal", "restrict"]:
        mlflow.log_param(variable, getattr(hp, variable))

    if hp.mlflow_tag:
        mlflow.set_tag("tag", hp.mlflow_tag)

    #dataFolder = hp.benchmark+"/small"
    dataFolder = hp.benchmark
    # If the filepath to the pkl files are different, change here:
    if hp.encoder == 'cl':
        query_path = "data/"+dataFolder+"/vectors/"+hp.encoder+"_query_"+hp.augment_op+"_"+hp.sample_meth+"_"+hp.table_order+"_"+str(hp.run_id)
        table_path = "data/"+dataFolder+"/vectors/"+hp.encoder+"_datalake_"+hp.augment_op+"_"+hp.sample_meth+"_"+hp.table_order+"_"+str(hp.run_id)

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
    print("Number of queries: %d" % (len(queries)))
    qfile.close()
    # Call NaiveSearcher, which has linear search and bounds search, from naive_search.py
    searcher = NaiveSearcher(table_path, hp.scal)
    returnedResults = {}
    returnedResults_noscore = {}
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
            if hp.matching == 'exact':
                if(hp.restrict==0):
                  qres = searcher.topk(hp.encoder, query, hp.K, threshold=hp.threshold)
                else:
                  print("working on restricted data")
                  qres = searcher.topk(hp.encoder, query, hp.K, threshold=hp.threshold, restrict=1 , 
                                       gth="data/"+dataFolder+"/santos_union_groundtruth.pickle_diluted.pickle")
  
            else: # Bounds matching
                qres = searcher.topk_bounds(hp.encoder, query, hp.K, threshold=hp.threshold)
            res = []
            for tpl in qres:
                tmp = (tpl[0],tpl[1])
                res.append(tmp)
                #with score
                returnedResults[query[0]] = [(r[1],r[0]) for r in res]
                returnedResults_noscore[query[0]] = [r[1] for r in res]
            query_times.append(time.time() - query_start_time)
    #print(returnedResults)        
    with open("data/"+dataFolder+"/diveristy_data/search_results/Starmie/top_20_Starmie_output_04diluted_restricted_withscore.pkl", 'wb') as file:
                 pickle.dump(returnedResults, file)   
    with open("data/"+dataFolder+"/diveristy_data/search_results/Starmie/top_20_Starmie_output_04diluted_restricted_noscore.pkl", 'wb') as file:
                 pickle.dump(returnedResults_noscore, file)                
    # print("Average QUERY TIME: %s seconds " % (sum(query_times)/len(query_times)))
    print("10th percentile: ", np.percentile(query_times, 10), " 90th percentile: ", np.percentile(query_times, 90))
    print("--- Total Query Time: %s seconds ---" % (time.time() - start_time))

 
if __name__ == '__main__':
    main()     