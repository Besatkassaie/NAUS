import random
import pickle
import bz2
import os
import sys
import csv
import torch
import numpy as np
import pandas as pd
import _pickle as cPickle
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

# Useful functions copied from SANTOS
# --------------------------------------------------------------------------------
# This function saves dictionaries as pickle files in the storage.
def saveDictionaryAsPickleFile(dictionary, dictionaryPath):
    if dictionaryPath.rsplit(".")[-1] == "pickle":
        filePointer=open(dictionaryPath, 'wb')
        pickle.dump(dictionary,filePointer, protocol=pickle.HIGHEST_PROTOCOL)
        filePointer.close()
    else: #pbz2 format
        with bz2.BZ2File(dictionaryPath, "w") as f: 
            cPickle.dump(dictionary, f)

            
# load the pickle file as a dictionary
def loadDictionaryFromPickleFile(dictionaryPath):
    print("Loading dictionary at:", dictionaryPath)
    if (dictionaryPath.rsplit(".")[-1] == "pickle" or dictionaryPath.rsplit(".")[-1] == "pkl"):
        filePointer=open(dictionaryPath, 'rb')
        dictionary = pickle.load(filePointer)
        filePointer.close()
    else: #pbz2 format
        dictionary = bz2.BZ2File(dictionaryPath, "rb")
        dictionary = cPickle.load(dictionary)
    print("The total number of keys in the dictionary are:", len(dictionary))
    return dictionary



def write_dict_to_csv(data_dict, file_path):
    """
    Write a dictionary to a CSV file, where each row is a key and one element from the corresponding list.

    Parameters:
        data_dict (dict): Dictionary where keys are strings and values are lists.
        file_path (str): Path to the CSV file.
    """
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write each key-value pair as a separate row
        for key, values in data_dict.items():
            for value in values:
                writer.writerow([key, value])




def load_alignment(alignmnet_file):
        """
        Load data from the specified source.

        The schema for the data is expected as:
        ['query_table_name', 'query_column', 'query_column#', 
        'dl_table_name', 'dl_column#', 'dl_column']
        """
        try:
            # Load the CSV file into a pandas DataFrame
            data = pd.read_csv(alignmnet_file)

            # Verify that the required columns are present
            required_columns = ['query_table_name', 'query_column', 'query_column#',
                                'dl_table_name', 'dl_column#', 'dl_column']
            if not all(column in data.columns for column in required_columns):
                missing_columns = [col for col in required_columns if col not in data.columns]
                raise ValueError(f"Missing required columns in data: {missing_columns}")

            print("alignment Data loaded successfully")
            return data
        
        except FileNotFoundError:
            print(f"Error: File not found at {alignmnet_file}")
        
        except ValueError as e:
            print(f"Error: {e}")
        
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            




from collections import defaultdict

def create_query_to_datalake_dict(csv_path):
    """
    Reads a CSV file with columns at least:
      - query_table
      - data_lake_table
    Returns a dictionary mapping each query_table
    to a set of data_lake_table values that appear.
    """
    query_to_datalake = defaultdict(set)
    
    with open(csv_path, mode='r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_table = row['query_table']
            d_table = row['data_lake_table']
            query_to_datalake[q_table].add(d_table)
    
    return dict(query_to_datalake) 

# load csv file as a dictionary. Further preprocessing may be required after loading
def loadDictionaryFromCsvFile(file_path):
    """
    Reads a CSV file and creates a dictionary where:
    - The first column serves as the key.
    - The second column values are stored as a list.

    If the file does not exist, it prints an error message.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None  # Return None if the file doesn't exist

    data_dict = defaultdict(list)  # Dictionary with lists as default values

    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:  # Ensure the row has at least two columns
                key, value = row[0].strip(), row[1].strip()
                data_dict[key].append(value)  # Append value to the list for the key
            else:
                print(f"Warning: Skipping malformed row {row}")

    return dict(data_dict)
    

 

# load csv file as a dictionary. Further preprocessing may be required after loading
def loadDictionaryFromCsvFile_withheader(filePath):
    with open(filePath, 'r') as f:
            # Read the file as a DataFrame
            main_df = pd.read_csv(f)
            groundtruth_dict = main_df.groupby('query_table')['data_lake_table'].apply(list).to_dict()
    return groundtruth_dict       

def loadDictionaryFromCSV_ugenv2(filepath):
                 # Path to the main CSV file
       main_csv_file = filepath
       print("testing common names")
        # Open the main CSV file
       with open(main_csv_file, 'r') as f:
            # Read the file as a DataFrame
            main_df = pd.read_csv(f)
            
        # Filter rows where the third column is 1 or 2 mean uionable: 1 means joinable (or unionable with 1-column) and 2 means unionable.
       filtered_df = main_df[(main_df['manual_unionable'] == 1) | (main_df['manual_unionable'] == 2)]
       filtered_df = filtered_df[['query_table', 'data_lake_table']]
       groundtruth_dict = filtered_df.groupby('query_table')['data_lake_table'].apply(list).to_dict()
       #add '_ugen_v2' to all file names 
   
       groundtruth_dict = {
    f"{key[:-4]}_ugen_v2.csv": [f"{value[:-4]}_ugen_v2.csv" for value in values]
    for key, values in groundtruth_dict.items()
}
   
       return groundtruth_dict
# --------------------------------------------------------------------------------
# New functions specific to this project
# --------------------------------------------------------------------------------
# A function to compute cosine similarity between two numpy arrays
def CosineSimilarity(array1, array2):
    return np.dot(array1,array2)/(norm(array1)*norm(array2))

# A function that takes a table as pandas dataframe and returns a list of its serialized rows. Each row is serialized as a separate sentence.
# Serialization format: COL <col1 name> VAL <col1 value> COL <col2 name> VAL <col2 value> ..... COL <colN name> VAL <colN value>
def SerializeTable(table_df):
    rows = table_df.to_dict(orient='records')
    serialized_rows = []
    for item in rows:
        current_serialization = SerializeRow(item)
        serialized_rows.append(current_serialization)
    return serialized_rows

# input_sentence = "COL column1 name VAL column1 value COL column2 name VAL column2 value COL column3 name VAL column3 value"
def UseSEPToken(sentence):
    # Split the input sentence into pairs of column name and value
    pairs = sentence.split('COL')[1:]
    # Create the transformed sentence
    transformed_sentence = "[CLS] " + " [SEP] ".join(" ".join(pair.strip().replace("VAL", "").split(" ")) for pair in pairs) + " [SEP]"
    transformed_sentence = transformed_sentence.strip()
    return transformed_sentence

def SerializeRow(row):
    current_serialization = str()
    for col_name in row:
        cell_value = str(row[col_name]).replace("\n","").replace("\t", " ")
        col_name = str(col_name).replace("\n", "").replace("\t"," ")
        current_serialization += "COL " + col_name + " VAL " + cell_value + " "
    current_serialization = current_serialization.strip() #remove trailing and leading spaces
    current_serialization = current_serialization.replace("\n", "")
    current_serialization = UseSEPToken(current_serialization) #remove this line to use old serialization
    return current_serialization

# A function that takes a list of serialized rows as input and returns embeddings for the table.
# It computes average embedding of a sample of rows, adds new rows iteratively to the sample and recompute embeddings.
# The table embedding is confirmed when the stopping criteria is reached i.e., the newly added samples are not impacting the embeddings by already selected samples.
def EmbedTable(serialized_rows, model, embedding_type, tokenizer, sample_size = 20, sim_threshold = 0.05):
    total_rows = len(serialized_rows)
    used_rows = 0
    #serialized_rows = set(serialized_rows) #using set of rows so that we can quickly sample without replacement
    sample1_list = random.sample(serialized_rows, min(sample_size, len(serialized_rows)))
    if embedding_type == "sentence_bert":
        sample1_embeddings = model.encode(sample1_list)
    else: #add more for other kinds
        sample1_embeddings = encode_finetuned(sample1_list, model, tokenizer)
    sample1_average_embeddings = np.mean(sample1_embeddings, axis=0)
    serialized_rows = list(set(serialized_rows) - set(sample1_list))
    while(len(serialized_rows) > 0):
        sample2_list = random.sample(serialized_rows, min(sample_size, len(serialized_rows)))
        if embedding_type == "sentence_bert":
            sample2_embeddings = model.encode(sample2_list)
        else:
            sample2_embeddings = encode_finetuned(sample2_list, model, tokenizer)
        sample2_average_embeddings = np.mean(sample2_embeddings, axis = 0)
        serialized_rows = list(set(serialized_rows) - set(sample2_list))
        cosine = CosineSimilarity(sample1_average_embeddings, sample2_average_embeddings)
        sample1_average_embeddings = (sample1_average_embeddings + sample2_average_embeddings) / 2
        #print("Current cosine similarity:", cosine)
        if cosine >= (1 - sim_threshold):
            break
    used_rows = total_rows - len(serialized_rows)               
    # print("Total rows:", total_rows)
    # print("Used rows for serialization:", total_rows - len(serialized_rows))
    return sample1_average_embeddings, total_rows, used_rows

# A function that takes a list of serialized tuples as input and returns embeddings for each tuple as a list with embeddings as value.
def EmbedTuples(tuple_list, model, embedding_type, tokenizer, batch_size = 1000):
    # Initialize an empty dictionary
    final_embedding_list = []

    tuples_batch = []

    if len(tuple_list) > 0:
        # Iterate through sentence list and form batches
        for i in range(0, len(tuple_list)):
            sentence = tuple_list[i]  # For example, Sentence 1, Sentence 2, ...
            tuples_batch.append(sentence)
            # If the batch size is reached or it's the last sentence, embed the batch
            if len(tuples_batch) == batch_size or i == len(tuple_list) - 1:
                if embedding_type == "sentence_bert":
                    tuple_embeddings = model.encode(tuples_batch, convert_to_tensor=True)
                    embeddings_list = tuple_embeddings.cpu().numpy()
                else: #add more for other kinds
                    embeddings_list = encode_finetuned(tuples_batch, model, tokenizer)
                # Add the entries to the dictionary with IDs as the keys and embeddings as the values
                for embedding in embeddings_list:
                    final_embedding_list.append(embedding)
                # Clear the batch for the next set of sentences
                tuples_batch = []
        return final_embedding_list


# A function to load the pretrained model and use it to encode tables.
def encode_finetuned(sentences, model, tokenizer):
    # Tokenize input sentences and convert to tensors
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    # device = "cpu"
    model.to(device)
    encodings = tokenizer(sentences, add_special_tokens = True, truncation = True, padding=True, return_tensors='pt')
    # print("encodings:", encodings)
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    # print("input ids:", input_ids.shape)
    # Generate embeddings for input sentences
    with torch.no_grad():
        embeddings = model(input_ids, attention_mask)
    # print("embeddings tensor:", embeddings.shape)
    # Convert embeddings to numpy array and print
    embeddings = embeddings.cpu().numpy()
    # print("embeddings numpy:", len(embeddings), type(embeddings))
    # print("average embeddings len:", len(np.mean(embeddings, axis = 0)))
    # sys.exit()
    del sentences
    return embeddings

# visualize the results.
def LinePlot(dict_lists, xlabel, ylabel, figname,title):
    # create a list of X-axis values (positions in the list)
    x_values = list(range(1, len(next(iter(dict_lists.values())))+1))
    # print(f'xvalues: {x_values}')
    # create the plot
    for label, values in dict_lists.items():
        plt.plot(x_values, values, label=label)
    
    # set the labels for X and Y axis
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # set the X-axis tick values
    if x_values:
        divisor = max(1,len(x_values)//10)
        # print(f"division:{divisor}")
        if divisor == 0:
            num_ticks = 1
        else:
            num_ticks = len(x_values)//divisor
        step_size = len(x_values)//num_ticks
        # print("step size:", step_size)
        x_ticks = x_values[::step_size]
        # print(f"x ticks: {x_ticks}")
        plt.xticks(x_ticks)
    # plt.ylim(0.1, 1.1)
    # y_ticks = [i/10 for i in range(11)]
    # plt.yticks(y_ticks)
    plt.legend()
    plt.title(title)
    plt.savefig(figname)
    plt.clf()


def read_csv_file(gen_file):
    data = []
    try:
        if "ugen_v2" in gen_file:
           return pd.read_csv(gen_file, sep=';')
        data = pd.read_csv(gen_file, lineterminator='\n', low_memory=False)
        if data.shape[1] < 2:
            data = pd.read_csv(gen_file, sep='|')
    except:
        try:
            data = pd.read_csv(gen_file, sep='|')
        except:
            with open(gen_file) as curr_csv:
                curr_data = curr_csv.read().splitlines()
                curr_data = [len(row.split('|')) for row in curr_data]
                max_col_num = 0
                if len(curr_data) != 0:
                    max_col_num = max(curr_data)
                try:
                    if max_col_num != 0:
                        df = pd.read_csv(gen_file, sep='|', header=None, names=range(max_col_num), low_memory=False)
                        data = df
                        return data
                    else:
                        df = pd.read_csv(gen_file, lineterminator='\n', low_memory=False)
                        data = df
                        return data
                except:
                    df = pd.read_csv(gen_file, lineterminator='\n', low_memory=False)
                    data = df
                    return data
    return data