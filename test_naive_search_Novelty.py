import numpy as np
import random
import pickle
import argparse
import mlflow
from naive_search_Novelty import NaiveSearcherNovelty
from checkPrecisionRecall import saveDictionaryAsPickleFile, calcMetrics
import time
from datetime import datetime

from SetSimilaritySearch import SearchIndex
from  process_column import TextProcessor
from scipy.stats import entropy




current_time = datetime.now()


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




 # build if does not exist , load if exists
# data type either los(list of set) or lol(list of list)
def getProcessedTables(text_processor, proccessed_file_name,processed_path, raw_data , dataType, tokenize, bot):
        data_prc={}
        processed_set_file_path=processed_path+proccessed_file_name
        proccessed_set_exists = os.path.isfile(processed_set_file_path)
        if proccessed_set_exists:
        #load it  
            print("loading proccessed data......")
            with open(processed_set_file_path, 'rb') as file:
                data_prc = pickle.load(file)
        else: #build it
            if(dataType=="los"):
                list_of_lists=[]
                
                for key, value in raw_data.items():
                    list_of_lists= value
                    if(tokenize==1):
                        list_of_lists=text_processor.processColumns(list_of_lists)
                    #to remove duplicates
                    list_of_sets = [set(inner_list) for inner_list in list_of_lists]
                    if(bot==1):
                        list_of_sets=text_processor.columnsToBagOfTokens(list_of_sets)
                    data_prc[key]=list_of_sets
                    
            
                    
            else:
                 for key, value in raw_data.items():
                
                    # The input sets must be a Python list of iterables (i.e., lists or sets).
                    list_of_lists= []
                    for col in value:
                        list_of_lists.append(text_processor.process(col))
                            
                    data_prc[key]=list_of_lists
       
            with open(processed_set_file_path, 'wb') as file:
                            pickle.dump(data_prc, file)   

                
     
            
        return data_prc


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
    parser.add_argument("--penalty", type=int, default=0) # if 1  t penalize for the sake of novelty if 0 do not 
    parser.add_argument("--tokenize",  type=int, default=0) # if 1   tokenize and normalize cell values  of both query and data lake 
    parser.add_argument("--bot", type=int, default=0) # if 1  get treat each column as bag of tokens
    parser.add_argument("--penalty_degree", type=int, default=1) # the power degree of penalty
    parser.add_argument("--late_penalty", type=int, default=0) # the step that the penalty applied late_penalty=1 mean that it happend after
    #finding max bipartite graph using semantic similarity and when computing the final score for each graph 0 means is happened when computing the semantic similarity for each column pair
    parser.add_argument("--entropy", type=int, default=0) # 1 means take into account jensenshannon distance between  query column and dl column when doing penalization


    hp = parser.parse_args(args=args2)

    # mlflow logging
    for variable in ["encoder", "benchmark", "augment_op", "sample_meth", "matching", 
                     "table_order", "run_id", "single_column", "K", "threshold", "scal", 
                     "penalty","tokenize", "bot", "penalty_degree", "late_penalty", "entropy"]:
        mlflow.log_param(variable, getattr(hp, variable))

    if hp.mlflow_tag:
        mlflow.set_tag("tag", hp.mlflow_tag)

    dataFolder = hp.benchmark

    if hp.tokenize== 1:
        text_processor = TextProcessor()
        


    run = mlflow.active_run()
    # artifact_uri = run.info.artifact_uri
    # print(artifact_uri)

    # If the filepath to the pkl files are different, change here:
    if hp.encoder == 'cl':
        query_path = "data/"+dataFolder+"/vectors/"+hp.encoder+"_query_"+hp.augment_op+"_"+hp.sample_meth+"_"+hp.table_order+"_"+str(hp.run_id)
        table_path = "data/"+dataFolder+"/vectors/"+hp.encoder+"_datalake_"+hp.augment_op+"_"+hp.sample_meth+"_"+hp.table_order+"_"+str(hp.run_id)
        query_path_raw = "data/"+dataFolder+"/"+"query"
        table_path_raw = "data/"+dataFolder+"/"+"datalake"
        index_path="data/indices/"
        processed_path="data/processed/"+dataFolder+"/"
        

        if hp.single_column:
            query_path += "_singleCol"
            table_path += "_singleCol"
        query_path += ".pkl"
        table_path += ".pkl"
    else:
        query_path = "data/"+dataFolder+"/"+hp.encoder+"_query.pkl"
        table_path = "data/"+dataFolder+"/"+hp.encoder+"_datalake.pkl"

    
    #here we construct the index file name and 
    # if it exists we do not create one and only load on 
    #index file name is constructed like: Joise_Index_DL_hp.benchmark_hp.tokenize_hp.bot
    if hp.tokenize==1:
        tk='tokenized'
    else:
        tk='ntokenized'
    if hp.bot==1:
        bot='bot'
    else:
        bot='nbot'  
                  
    index_file_name="Joise_Index_DL_"+hp.benchmark+"_"+tk+"_"+bot+".pkl"
    # Load the query file
   # print("quesry path"+query_path)
    qfile = open(query_path,"rb")
    queries = pickle.load(qfile)
    # queries is a list of tuples ; tuple of (str(filename), numpy.ndarray(vectors(numpy.ndarray) for columns) 
    qfile.close()
    
    queries_raw_org=NaiveSearcherNovelty.read_csv_files_to_dict(query_path_raw)
    #I assume that the order  of columns are comming from the original csv files 
   
   
    # we preprocess the values in tables both query and data lake tables
    list_of_lists=[]
    queries_raw={}
    
    q_tbls_processed_set_file_name="q_tbls_processed_set.pkl"
    queries_raw= getProcessedTables(text_processor, q_tbls_processed_set_file_name, processed_path, queries_raw_org,"los",hp.tokenize, hp.bot )
    tables_raw=NaiveSearcherNovelty.read_csv_files_to_dict(table_path_raw)
    
    #the normalized/tokenized/original with no duplicates  dl(data lake) tables are stored in table_raw_proccessed_los
    table_raw_proccessed_los={}
    
        # write the proccessed result having columns as set to a pickle file 
    dl_tbls_processed_set_file_name="dl_tbls_processed_set.pkl"
    
    table_raw_proccessed_los=getProcessedTables(text_processor, dl_tbls_processed_set_file_name, processed_path, tables_raw,"los", hp.tokenize, hp.bot)
     # process dl tables and save as list of lists 
     
    table_raw_lol_proccessed={}
        # write the proccessed result having columns as set to a pickle file 
    dl_tbls_processed_lol_file_name="dl_tbls_processed_lol.pkl"
    
    table_raw_lol_proccessed=getProcessedTables(text_processor,  dl_tbls_processed_lol_file_name,processed_path, tables_raw,"lol", hp.tokenize, hp.bot)
     
     # process q tables and save as list of lists 
     
    q_table_raw_lol_proccessed={}
        # write the proccessed result having columns as set to a pickle file 
    q_tbls_processed_lol_file_name="q_tbls_processed_lol.pkl"
    q_table_raw_lol_proccessed=getProcessedTables(text_processor, q_tbls_processed_lol_file_name,processed_path, queries_raw_org,"lol", hp.tokenize, hp.bot)
    # lets build the set similarity index here over data lakes and store 
    # the reslut in a dictionary from table name to its column index

    table_raw_index={}

    index_file_path=index_path+index_file_name
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
       
    


    
  
    # Call NaiveSearcher, which has linear search and bounds search, from naive_search.py
    searcher = NaiveSearcherNovelty(table_raw_lol_proccessed,table_path, hp.scal,table_raw_index, table_raw_proccessed_los,  hp.entropy)
    
    returnedResults = {}
    returnedResults_full=[]
    start_time = time.time()
    # For error analysis of tables
    analysis = hp.analysis
    # bucketFile = open("data/"+dataFolder+"/buckets/query_"+analysis+"Bucket_"+str(hp.bucket)+".txt", "r")
    # bucket = bucketFile.read()
    queries.sort(key = lambda x: x[0])
    query_times = []
    qCount = 0
    res_full=[]
    for query in queries:
            qCount += 1
            if qCount % 10 == 0:
                print("Processing query ",qCount, " of ", len(queries), " total queries.")
        # if query[0] in bucket:
            query_start_time = time.time()
            #get the query raw content 
            raw_query_processed_los=queries_raw[query[0]]
            raw_query_asList=queries_raw_org[query[0]]
            if(len(raw_query_processed_los)!= len(query[1])):
                print("raise an exception stating that the files do not match")
            
            if hp.matching == 'exact':
                if(hp.penalty==1):
                    if(hp.late_penalty==0):
                         qres = searcher.topk(hp.penalty_degree,hp.encoder,raw_query_processed_los, query, hp.K, hp.penalty,threshold=hp.threshold)
                    else:     
                    #topk_late_penalty
                     qres = searcher.topk_late_penalty(hp.penalty_degree,hp.encoder,raw_query_asList,q_table_raw_lol_proccessed,
                                                       raw_query_processed_los, query, hp.K, hp.penalty,threshold=hp.threshold)
                else:
                    qres = searcher.topk_2(hp.encoder, query, hp.K, threshold=hp.threshold)

                         
            else: # Bounds matching
                qres = searcher.topk_bounds(hp.encoder, query, hp.K, threshold=hp.threshold)
            res = []
            for tpl in qres:
                tmp = (tpl[0],tpl[1])
                tmp_full=(query[0],tpl[1], tpl[0])
                res.append(tmp)
                res_full.append(tmp_full)
            returnedResults[query[0]] = [r[1] for r in res]
            query_times.append(time.time() - query_start_time)
    
    #write_dict_to_csv_file(returnedResults,'penalty'+str(penalty)+'.csv')
    
    csv_file = 'full_results_penalty_'+str(hp.penalty)+'_tk_'+str(hp.tokenize)+'_bot_'+str(hp.bot)+'_penaldeg_'+str(hp.penalty_degree)+'.csv'
    with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['query', 'table', 'score'])  # Add headers
                               
            writer.writerows(res_full)
            #writer.writerow("date time : {}".format(datetime.now()))

    # Log the CSV file as an artifact
    mlflow.log_artifact(csv_file)
    
    
    
    AverageQUERYTIME=(sum(query_times)/len(query_times))
    mlflow.log_param("AverageQUERYTIME", AverageQUERYTIME)
    print("Average QUERY TIME: %s seconds " % (sum(query_times)/len(query_times)))
    print("10th percentile: ", np.percentile(query_times, 10), " 90th percentile: ", np.percentile(query_times, 90))
    print("--- Total Query Time: %s seconds ---" % (time.time() - start_time))
    TotalQueryTime=time.time() - start_time
    mlflow.log_param("TotalQueryTime", TotalQueryTime)

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
 
