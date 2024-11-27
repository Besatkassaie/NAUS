import time
from datetime import datetime
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import csv
from  process_column import TextProcessor
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import pickle
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances






def table2doc(table_):
    ''' input table_ is a list of list (columns)
        we only merge the lists and retrun it as the document'''
    merged_list = [item for sublist in table_ for item in sublist]
    return  merged_list

def process_file(input_path, output_path):

       # Read the CSV file as a list of lists
    with open(input_path, 'r') as infile:
        reader = csv.reader(infile)
        table_ = list(reader)  # Convert to list of lists

    # Perform processing here (this is just a placeholder)
    doc = table2doc(table_)

    # Save the result to the output file
    with open(output_path, 'w') as outfile:
         outfile.write(" ".join(doc))

def buildCorpus(input_folder, output_folder):
    ''' got through all the tables in the input folder and 
    convert it to documents and wrtie the result as a text file in output folder
    the document name is the same as table name but as .txt '''
        # Ensure the output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Get a list of all files in the input folder
    input_files = list(Path(input_folder).glob('*'))

    # Process each file in parallel
 
    for input_file in input_files:
        output_file = Path(output_folder) / (input_file.stem + '.txt')
        process_file(input_file, output_file) 
    
def processCorpus(input_folder, output_folder):
    text_processor = TextProcessor()
    input_files = list(Path(input_folder).glob('*.txt'))
    for input_file in input_files:
        output_file = Path(output_folder) / (input_file.stem + '.txt')
        with open(input_file, 'r') as file:
            content_input = file.read()
        content_out_list=text_processor.processString(content_input) 
        content_out = " ".join(content_out_list)
           # Save the result to the output file
        with open(output_file, 'w') as outfile:
            outfile.write(content_out)



def data_exist(x_files_path, y_files_path, x_files_type, y_files_type):
    x_files = list(Path(x_files_path).glob(x_files_type))
    y_files = list(Path(y_files_path).glob(y_files_type))
    if len(x_files) != len(y_files):
        # Delete all files in y_folder
            for file in y_files:
                file.unlink()
            return False    
    else: 
            return True
        

def buildVocabDic(paths):
    '''the input is a list of folder path
    the vocab dictionary is build using all the document in all the input folders
    output is a dictionary from tokens to index'''
    corpus=[]
    # read documents 
    for flder in paths:
        folder_path = Path(flder)
        for file_path in folder_path.glob('*.txt'):  # 
          if file_path.is_file():  # Ensure it's a file and not a directory
            with open(file_path, 'r') as file:
              content = file.read()  # Read the content as a string
              corpus.append(content)
              
    print("number of documents in the corpus for building vocab dictionary is "+str(len(corpus)))         
    
    # Initialize CountVectorizer
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(' '), stop_words=None)
    vectorizer.token_pattern
    

# Fit the vectorizer to the documents to build the vocabulary
    vectorizer.fit(corpus)

# Retrieve the vocabulary (term-to-index mapping)
    term_to_index = vectorizer.vocabulary_
    print("the number of vocabulary is "+str(len(term_to_index)))
    return   term_to_index   

def produceSparseVectors(path_, vocabs):
        
        '''the input is a list of folder path and input voacabulary dictionary
        for every file a vector is generated
        and output is a dictionary from file name to vector
        I have tested and made sure that query table has 50 files that are copied
        from data lake so no need to keep two corpus seperated''' 
        print("constructing sparse vectors......")
        sparse_vectors={}
        corpus=[]
        # we keep track of file name and its index in the genrated list of vectors 
        dic_filename2index={}
        indx=0
        folder_path = Path(path_)
        for file_path in folder_path.glob('*.txt'):  # 
                if file_path.is_file():  # Ensure it's a file and not a directory
                    with open(file_path, 'r') as file:
                      content_ = file.read()  # Read the content as a string 
                      corpus.append(content_)
                      dic_filename2index[file_path.name]=indx
                      indx=indx+1
        
                      
        vectorizer = CountVectorizer(vocabulary=vocabs,tokenizer=lambda x: x.split(' '), stop_words=None)
        X = vectorizer.fit_transform(corpus)
        all_vectors=X.toarray()
        #then add the the sparse_vectors
        for file_name in dic_filename2index.keys():
            index_=dic_filename2index[file_name]
            vector=all_vectors[index_]
            sparse_vectors[file_name]=vector 
        
                    
        return sparse_vectors    
        
def buildNoveltyGroundTruth( sparse_vectors, input_gtruth,new_ground_truth, distance_type='Euclidean'):
    
   

# Column names in the new ground truth
    columns = ["query_table", "data_lake_table", "relevance"]
    df = pd.read_csv(input_gtruth, usecols=[1, 3])  # Replace 'your_file.csv' with the path to your CSV file

# Group by the 'query_table' column
    grouped = df.groupby('query_table')

# Iterate through each group and print the rows
    for name, group in grouped:
        q_t= name
       
        dl_tbs=group['data_lake_table'].values
        dl_tbs_count=len(dl_tbs)
        
        q_t_vect=sparse_vectors[q_t.replace(".csv",".txt")]
       # dl_tbs_l=dl_tbs.tolist # data lake tables 
        dl_tbs_l=[t.replace(".csv",".txt") for t in dl_tbs.tolist()]
        #dl_tbs_l_vects=[sparse_vectors[tb] for tb in dl_tbs_l]
        dl_tbs_ranked={}
        i=0
        max_relevancy=dl_tbs_count
        while i<dl_tbs_count:
             # scores=[Euclidean_distance(q_t_vect,sparse_vectors[t]) for t in dl_tbs_l]  
              scores = []
              for t in dl_tbs_l:
                    try:
                        # Calculate the Euclidean distance for the current item
                        distance = Euclidean_distance(q_t_vect, sparse_vectors[t])
                        scores.append(distance)
                    except KeyError:
                        # Key is missing in sparse_vectors; skip this item and continue
                        print(f"Warning: Key '{t}' not found in sparse_vectors. Skipping.")
                        continue
                    
              max_value = max(scores)
              max_indexes = [i for i, x in enumerate(scores) if x == max_value]
              
              for max_index in max_indexes:
                try:
                 dl_tbs_ranked[dl_tbs_l[max_index]]=max_relevancy
                
                 q_t_vect=q_t_vect+sparse_vectors[dl_tbs_l[max_index]]
                except KeyError:
                        # Key is missing in sparse_vectors; skip this item and continue
                        print(f"Warning: Key '{t}' not found in sparse_vectors. Skipping.")
                        continue
              tmp_dl_tbs_l = [item for i, item in enumerate(dl_tbs_l) if i not in max_indexes]
              
              dl_tbs_l=tmp_dl_tbs_l
              i=i+len(max_indexes)
              max_relevancy=max_relevancy-1         
        #adjust relevancy
        min_relevancy=min(dl_tbs_ranked.values())-1

        modified_dict = {key: value -min_relevancy for key, value in dl_tbs_ranked.items()}
        # Column names

        # Check if the file exists
        file_exists = os.path.exists(new_ground_truth)

# Open the file in append mode if it exists, otherwise in write mode
        with open(new_ground_truth, mode='a' if file_exists else 'w', newline='') as file:
                writer = csv.writer(file)
    # Write the header only if the file is being created (write mode)
                if not file_exists:
                    columns = ["query_table", "data_lake_table", "relevance"]
                    writer.writerow(columns)
    # Write the data rows
                for data_lake_table, relevance in modified_dict.items():
                    writer.writerow([q_t, data_lake_table, relevance])

        
        
 


        
    return 9     
def Euclidean_distance(vec_1, vec_2):
    dist = np.linalg.norm(vec_1 - vec_2)
    return dist


def main(args_2=None):
        
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--benchmark", type=str, default='santos')
        # parser.add_argument("--groundtruth", type=str, default='santos_benchmark')
        
        # hp = parser.parse_args(args=args2)
        
        # dataFolder = hp.benchmark
        # benchmark_folder=hp.groundtruth
        dataFolder="santos"
        sparse_vector_path="data/"+dataFolder+"/sparse_vectors/"
        sparse_vector_file_name='dl_q_vectors.pkl'
        
        
        q_tables_raw_path= "data/"+dataFolder+"/"+"query"
        dl_tables_raw_path = "data/"+dataFolder+"/"+"datalake"
        
        q_documents_raw_path= "data/"+dataFolder+"/"+"documents"+"/"+"query"
        dl_documents_raw_path = "data/"+dataFolder+"/"+"documents"+"/"+"datalake"
        
        q_processed_documents_raw_path= "data/"+dataFolder+"/"+"processed_documents"+"/"+"query"
        dl_processed_documents_raw_path = "data/"+dataFolder+"/"+"processed_documents"+"/"+"datalake"
       
   
       
       
        if(not data_exist(q_tables_raw_path,q_documents_raw_path,'*.csv', '*.txt' )): 
            buildCorpus(q_tables_raw_path,q_documents_raw_path)  

        if(not data_exist(dl_tables_raw_path,dl_documents_raw_path,'*.csv', '*.txt' )): 
            buildCorpus(dl_tables_raw_path,dl_documents_raw_path)  
        
        # process every document: tokenize, stem, and casefolding and persist to folder
        # check if data does not exist generate them 
        
        if(not data_exist(q_documents_raw_path,q_processed_documents_raw_path,'*.txt', '*.txt' )): 
             processCorpus (q_documents_raw_path,q_processed_documents_raw_path)  

        if(not data_exist(dl_documents_raw_path,dl_processed_documents_raw_path,'*.txt', '*.txt' )): 
             processCorpus (dl_documents_raw_path,dl_processed_documents_raw_path)  
        
        
        # now build dictionary from both dl and q which is mapping from term to index
        
        # now build/load vectors of term frequency for query and dl documents \
        # and persist them in
        # pkl file which is a dictionary from file name to its sparse vector
        
        vector_dic={}
        sparse_vec_Path=sparse_vector_path+sparse_vector_file_name
        vec_exists = os.path.isfile(sparse_vec_Path)
        if vec_exists:
        #load it  
           print("loading sparse vector dictionary")
           with open(sparse_vec_Path, 'rb') as file:
              vector_dic = pickle.load(file)
        else:    
            vocab_dict=buildVocabDic([dl_processed_documents_raw_path,q_processed_documents_raw_path])

            vector_dic=produceSparseVectors(dl_processed_documents_raw_path, vocab_dict)
            # write in a pickle file  
            with open(sparse_vec_Path, 'wb') as file:
                    pickle.dump(vector_dic, file)   
        
        distance_type="Euclidean"
        ground_truth="data/"+dataFolder+"/santos_small_benchmark_groundtruth.csv"
        new_ground_truth="data/"+dataFolder+"/santos_small_benchmark_groundtruth"+distance_type+".csv"
        
        buildNoveltyGroundTruth(vector_dic,ground_truth,new_ground_truth,distance_type )
      
 
    
      
        



        
if __name__ == '__main__':
        main()        
 

        