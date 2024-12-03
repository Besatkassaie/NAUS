import numpy as np
import pandas as pd

import pickle
import random
import heapq
from munkres import Munkres, make_cost_matrix, DISALLOWED, print_matrix
from numpy.linalg import norm
from bounds import verify, upper_bound_bm, lower_bound_bm, get_edges
import os
import csv
from SetSimilaritySearch import all_pairs
import io
from collections import Counter
from scipy.stats import entropy
import math 
from scipy.spatial import distance





class NaiveSearcherNovelty(object):
    def __init__(self,tb_dl_lol,
                 table_path,
                 scale,
                 table_raw_index,
                 tables_raw_data_los, entropy,
                 index_path=None
                 ):
        if index_path != None:
            self.index_path = index_path
        self.table_raw_index=table_raw_index
        self.tables_raw_data=tables_raw_data_los
        self.entropy=entropy
        self.tb_dl_lol=tb_dl_lol
        # load tables to be queried
        tfile = open(table_path,"rb")
        tables = pickle.load(tfile)
        # tables is a list of tuples ; tuple of (str(filename), numpy.ndarray(vectors(each verstor is a numpy.ndarray) for columns) 
        #I assume that the order  of columns are comming from the original csv files 
        # For scalability experiments: load a percentage of tables
        self.tables = random.sample(tables, int(scale*len(tables)))
    
        print("From %d total data-lake tables, scale down to %d tables" % (len(tables), len(self.tables)))
        tfile.close()

    def topk_2(self, enc, query, K, threshold):
        ''' Exact top-k cosine similarity with full bipartite matching
        Args:
            enc (str): choice of encoder (e.g. 'sato', 'cl', 'sherlock') -- mainly to check if the encoder is 'sato'
            query: the query, where query[0] is the query filename, and query[1] is the set of column vectors
            K (int): choice of K
            threshold (float): similarity threshold. For small SANTOS benchmark, we use threshold=0.7. For the larger benchmarks, threshold=0.1
        Return:
            Tables with top-K scores
        '''
        if enc == 'sato':
            # For SATO encoder, the first 1187 items in the vector are from Sherlock. The rest are from topic modeling
            scores = []
            querySherlock = query[1][:, :1187]
            querySato = query[1][0, 1187:]
            for table in self.tables:
                sherlock = table[1][:, :1187]
                sato = table[1][0, 1187:]
                sScore = self._verify(querySherlock, sherlock, threshold)
                sherlockScore = (1/min(len(querySherlock), len(sherlock))) * sScore
                satoScore = self._cosine_sim(querySato, sato)
                score = sherlockScore + satoScore
                scores.append((score, table[0]))
        else:
            scores = [(self._verify_2(query[1], table[1], threshold), table[0]) for table in self.tables]
        scores.sort(reverse=True)
        return scores[:K]
    
    def topk(self, penalty_degree,enc,query_raw ,query, K, penalty, threshold):
            ''' Exact top-k cosine similarity with full bipartite matching with penalization
            Args:
                enc (str): choice of encoder (e.g. 'sato', 'cl', 'sherlock') -- mainly to check if the encoder is 'sato'
                query: the query, where query[0] is the query filename, and query[1] is the set of column vectors
                K (int): choice of K
                threshold (float): similarity threshold. For small SANTOS benchmark, we use threshold=0.7. For the larger benchmarks, threshold=0.1
            Return:
                Tables with top-K scores
            '''
            if enc == 'sato':
                # For SATO encoder, the first 1187 items in the vector are from Sherlock. The rest are from topic modeling
                scores = []
                querySherlock = query[1][:, :1187]
                querySato = query[1][0, 1187:]
                for table in self.tables:
                    sherlock = table[1][:, :1187]
                    sato = table[1][0, 1187:]
                    sScore = self._verify(querySherlock, sherlock, threshold)
                    sherlockScore = (1/min(len(querySherlock), len(sherlock))) * sScore
                    satoScore = self._cosine_sim(querySato, sato)
                    score = sherlockScore + satoScore
                    scores.append((score, table[0]))
            else:
                #print("query file name: "+query[0])
                #print("query file: "+str(len(query[1])))
                #print("query_raw file: "+str(len(query_raw[0])))

                scores = [(self._verify(penalty_degree,query_raw, query[1],self.table_raw_index[table[0]], table[1], threshold, penalty), table[0]) for table in self.tables]

#                 scores = []
#                 for table in self.tables:
#                     # Print table[0] and raw_tables[table[0]]
#                     #print("++++++++++++++++++++++++")
#                     #print("Table name:", table[0])
#                     # Compute the score using _verify
#                     score = self._verify(query_raw, query[1], self.raw_tables[table[0]], table[1], threshold, penalty)

#                     # Append the result to the scores list
#                     scores.append((score, table[0]))
            scores.sort(reverse=True)
            return scores[:K]    

    def item_frequency(self, lst):
        return dict(Counter(lst))
    
    
    def topk_late_penalty(self, penalty_degree,enc,raw_query_asList,q_table_raw_lol_proccessed,query_raw ,query, K, penalty, threshold):
        ''' Exact top-k cosine similarity with full bipartite matching with no penalization and applying the penalization after matching
        Args:
            enc (str): choice of encoder (e.g. 'sato', 'cl', 'sherlock') -- mainly to check if the encoder is 'sato'
            query: the query, where query[0] is the query filename, and query[1] is the set of column vectors
            K (int): choice of K
            threshold (float): similarity threshold. For small SANTOS benchmark, we use threshold=0.7. For the larger benchmarks, threshold=0.1
        Return:
            Tables with top-K scores
        '''
        if enc == 'sato':
            # For SATO encoder, the first 1187 items in the vector are from Sherlock. The rest are from topic modeling
            scores = []
            querySherlock = query[1][:, :1187]
            querySato = query[1][0, 1187:]
            for table in self.tables:
                sherlock = table[1][:, :1187]
                sato = table[1][0, 1187:]
                sScore = self._verify(querySherlock, sherlock, threshold)
                sherlockScore = (1/min(len(querySherlock), len(sherlock))) * sScore
                satoScore = self._cosine_sim(querySato, sato)
                score = sherlockScore + satoScore
                scores.append((score, table[0]))
        else:

            scores = [(self._verify_late_penalty(penalty_degree,q_table_raw_lol_proccessed,query_raw, query[1],query[0],self.table_raw_index[table[0]], table[1],table[0] , threshold, penalty), table[0]) for table in self.tables]


        scores.sort(reverse=True)
        return scores[:K]    
    
    
    def top_all_for_GMC(self, s_i,allowed_columns_table1, threshold):
        ''' TO be completed 
     
        Return:
            dataFram contaning diversity information
        '''
        semi_q=()
        for tup in self.tables:
            if tup[0] == s_i:
                semi_q= tup
        
       # all = [(self.generateDiversityData(allowed_columns_table1,self.tb_dl_lol,self.tables_raw_data, semi_q[1],semi_q[0], table[1],table[0] , threshold), table[0]) for table in self.tables]


      
        
        # Initialize an empty DataFrame with appropriate columns
        all_columns = ['DL_table', 'DL_column', 'Q_si_table', 'Qsi_column', 'dSize', 'distance', 'Lexsim']
        all = pd.DataFrame(columns=all_columns)

        # Get the semi-query table (current table to align)
       # semi_q = self.tables[s_i]

        # Iterate over all tables in self.tables
        for table in self.tables:
            # Call generateDiversityData and get the resulting DataFrame
            diversity_data = self.generateDiversityData(
                allowed_columns_table1,
                self.tb_dl_lol,
                self.tables_raw_data,
                semi_q[1],
                semi_q[0],
                table[1],
                table[0],
                threshold
            )
            # Append the result to the DataFrame
            all = pd.concat([all, diversity_data], ignore_index=True)
      
      
      
        return all 

    def topk_bounds(self, enc, query, K, threshold=0.6):
        ''' Algorithm: Pruning with Bounds
            Bounds Techique: reduce # of verification calls
        Args:
            enc (str): choice of encoder (e.g. 'sato', 'cl', 'sherlock') -- mainly to check if the encoder is 'sato'
            query: the query, where query[0] is the query filename, and query[1] is the set of column vectors
            K (int): choice of K
            threshold (float): similarity threshold. For small SANTOS benchmark, we use threshold=0.7. For the larger benchmarks, threshold=0.1
        Return:
            Tables with top-K scores
        '''
        H = []
        heapq.heapify(H)
        if enc == 'sato':
            querySherlock = query[1][:, :1187]
            querySato = query[1][0, 1187:]
        satoScore = 0.0
        for table in self.tables:
            # get sherlock and sato components if the encoder is 'sato
            if enc == 'sato':
                tScore = table[1][:, :1187]
                qScore = querySherlock
                sato = table[1][0, 1187:]
                satoScore = self._cosine_sim(querySato, sato)
            else:
                tScore = table[1]
                qScore = query[1]

            # add to heap to get len(H) = K
            if len(H) < K: # len(H) = number of elements in H
                score = verify(qScore, tScore, threshold)
                if enc == 'sato': score = self._combine_sherlock_sato(score, qScore, tScore, satoScore)
                heapq.heappush(H, (score, table[0]))
            else:
                topScore = H[0]
                # Helper method from bounds.py for to reduce the cost of the graph
                edges, nodes1, nodes2 = get_edges(qScore, tScore, threshold)
                lb = lower_bound_bm(edges, nodes1, nodes2)
                ub = upper_bound_bm(edges, nodes1, nodes2)
                if enc == 'sato':
                    lb = self._combine_sherlock_sato(lb, qScore, tScore, satoScore)
                    ub = self._combine_sherlock_sato(ub, qScore, tScore, satoScore)

                if lb > topScore[0]:
                    heapq.heappop(H)
                    score = verify(qScore, tScore, threshold)
                    if enc == 'sato': score = self._combine_sherlock_sato(score, qScore, tScore, satoScore)
                    heapq.heappush(H, (score, table[0]))
                elif ub >= topScore[0]:
                    score = verify(qScore, tScore, threshold)
                    if enc == 'sato': score = self._combine_sherlock_sato(score, qScore, tScore, satoScore)
                    if score > topScore[0]:
                        heapq.heappop(H)
                        heapq.heappush(H, (score, table[0]))
        scores = []
        while len(H) > 0:
            scores.append(heapq.heappop(H))
        scores.sort(reverse=True)
        return scores
        

    def _combine_sherlock_sato(self, score, qScore, tScore, satoScore):
        ''' Helper method for topk_bounds() to calculate sherlock and sato scores, if the encoder is SATO
        '''
        sherlockScore = (1/min(len(qScore), len(tScore))) * score
        full_satoScore = sherlockScore + satoScore
        return full_satoScore

    def topk_greedy(self, enc, query, K, threshold=0.6):
        ''' Greedy algorithm for matching
        Args:
            enc (str): choice of encoder (e.g. 'sato', 'cl', 'sherlock') -- mainly to check if the encoder is 'sato'
            query: the query, where query[0] is the query filename, and query[1] is the set of column vectors
            K (int): choice of K
            threshold (float): similarity threshold. For small SANTOS benchmark, we use threshold=0.7. For the larger benchmarks, threshold=0.1
        Return:
            Tables with top-K scores
        '''
        if enc == 'sato':
            scores = []
            querySherlock = query[1][:, :1187]
            querySato = query[1][0, 1187:]
            for table in self.tables:
                sherlock = table[1][:, :1187]
                sato = table[1][0, 1187:]
                sScore = self._verify_greedy(querySherlock, sherlock, threshold)
                sherlockScore = (1/min(len(querySherlock), len(sherlock))) * sScore
                satoScore = self._cosine_sim(querySato, sato)
                score = sherlockScore + satoScore
                scores.append((score, table[0]))
        else: # encoder is sherlock
            scores = [(self._verify_greedy(query[1], table[1], threshold), table[0]) for table in self.tables]
        scores.sort(reverse=True)
        return scores[:K]

    def _cosine_sim(self, vec1, vec2):
        ''' Get the cosine similarity of two input vectors: vec1 and vec2
        '''
        assert vec1.ndim == vec2.ndim
        return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))

    def _verify_2(self, table1, table2, threshold):
        score = 0.0
        nrow = len(table1)
        ncol = len(table2)
        graph = np.zeros(shape=(nrow,ncol),dtype=float)
        for i in range(nrow):
            for j in range(ncol):
                sim = self._cosine_sim(table1[i],table2[j])
                if sim > threshold:
                    graph[i,j] = sim

        max_graph = make_cost_matrix(graph, lambda cost: (graph.max() - cost) if (cost != DISALLOWED) else DISALLOWED)
        m = Munkres()
        indexes = m.compute(max_graph)
        for row,col in indexes:
            score += graph[row,col]
        return score
    
    
    
    def _lexicalsim(self, query_column, table_index, result_length):
        '''retrun an array where the index is tables column index and value is setsimilarity between query column and each table column'''
        results = table_index.query(query_column) # looks like results is empty sometimes. probably meaning that there were nothing similar
        return self.tuples_to_array(results, result_length)
        # [(1, 1.0), (0, 0.2), (2, 0.5), (3, 0.2)]
        # Each tuple is (<index of the found set>, <similarity>).
        # The index is the list index of the sets in the search index.
        
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
        
          
    def tuples_to_array(self, tuples_list, result_length):
        # Find the maximum index to define the array size
        
        # added to check if tuples_list is not empty
        if len(tuples_list)>0:
           # max_index = max(tuples_list, key=lambda x: x[0])[0]
            # Create an array of zeros with the appropriate size
            array = np.zeros(result_length)
            # Assign values to the array based on the tuples
            for index, value in tuples_list:
                array[index] = value
            return array    
        else:
            return None
    
    
    
    
            
    def _sigmoid(self,x):
            return 1 / (1 + math.exp(-x))  
   
        
    
    def _penalize(self, sem_sim, lexical_sim, penalty_degree, distance_, small_domain=0,entropy=0):
       if(entropy==0):
            penval=((1-lexical_sim)**penalty_degree)*sem_sim
       else: 
           if(lexical_sim==0):
               penval=sem_sim # no penalty 
           else:  # if the domain is small 
                if(small_domain==1):
                        penval=((distance_)**penalty_degree)*sem_sim
                else:
                     penval=((1-lexical_sim)**penalty_degree)*sem_sim

       return penval
 
        
    def _verify(self, penalty_degree, table1_raw,table1, table2_raw_index,table2, threshold, penalty):
        '''  this function apply a penalization to the similarity score at the same time as computing semantic similarity for pair of columns 
             table1 is query vector         '''
        
        score = 0.0
        nrow = len(table1)
        ncol = len(table2)
        graph = np.zeros(shape=(nrow,ncol),dtype=float)
        # remove later this is for logingg 
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["semantic sim", "lexical sim", "penalized value"])
        for i in range(nrow):
            #compute the lexical similarity between query column and  and all columns of the raw table
            #sim_results=self._lexicalsim(table1[i],table2_raw_index)
            sim_results=self._lexicalsim(table1_raw[i],table2_raw_index, len(table2))
            for j in range(ncol):
                sim = self._cosine_sim(table1[i],table2[j])
                #print("column of query:")
                #print(len(table1_raw[i]))
                #if sim > threshold:
                    #now compute the penalty between raw table column and raw querly column 
                    #print("cosine similarity: "+str(sim))
                lexical_sim=0
                if penalty>0 and not sim_results is None : # added the condition not sim_results is None, because sometimes it's zero since no similarity exist 
                    lexical_sim=sim_results[j]
                    val_=self._penalize(sim, lexical_sim, penalty_degree)
                    if val_>threshold:
                       writer.writerow([sim,lexical_sim, val_])
                       graph[i,j] = val_
                else:
                    if sim > threshold:
                        writer.writerow([sim,-1, sim])
                        graph[i,j] = sim

        with open('simpenalizevalues.csv', 'a', newline='') as file:  # Open in append mode
                 file.write(buffer.getvalue())
        # Close the buffer
        buffer.close()           
        max_graph = make_cost_matrix(graph, lambda cost: (graph.max() - cost) if (cost != DISALLOWED) else DISALLOWED)
        #print_matrix(max_graph)
        m = Munkres()
        indexes = m.compute(max_graph)
        for row,col in indexes:
            score += graph[row,col]
        return score

        
    def _verify_late_penalty(self, penalty_degree,q_table_raw_lol_proccessed, table1_raw,table1,query_name, table2_raw_index,table2,dl_table_name, threshold, penalty):
        '''  this function apply a penalization to the similarity score  after the most semantically similar columns are found 
             table1 is query vector         '''
        score = 0.0
        nrow = len(table1)
        ncol = len(table2)
        graph = np.zeros(shape=(nrow,ncol),dtype=float)

        for i in range(nrow):
            for j in range(ncol):
                sim = self._cosine_sim(table1[i],table2[j])
                if sim > threshold:
                    graph[i,j] = sim

        max_graph = make_cost_matrix(graph, lambda cost: (graph.max() - cost) if (cost != DISALLOWED) else DISALLOWED)
        m = Munkres()
        indexes = m.compute(max_graph)
        DSize=20
        csv_file = 'full_small_domain_domain_size_Threshold'+str(DSize)+'.csv'
        with open(csv_file, mode='a', newline='') as file:
                     writer = csv.writer(file)
                     writer.writerow(['DL_table', 'DL_column',  'Q_table', 'Q_column','entropy','dSize','distance','Lexsim', 'SemSim', 'final_value'])  # Add headers
        #here we have the raw semantically matched and picked  columns we hope here 
        # that the most semantically similar columns are being paired
        
        for row,col in indexes:
            distance=0
            semsim=graph[row,col]
            #get the comlumn from data lake table
            dl_column=self.tb_dl_lol.get(dl_table_name)[col]
            dl_column_set=self.tables_raw_data.get(dl_table_name)[col]
            # dl_column=self.text_processor.process(dl_column)
            small_domain=0
            row_file=[] 
            row_file.append(dl_table_name)
            row_file.append(col)
            row_file.append(query_name) 
            row_file.append(row)
            row_file.append(penalty)

            if(self.entropy==1):
                #compute distribution of the column
                query_column=q_table_raw_lol_proccessed.get(query_name)[row]
                # query_column=raw_table1_asList[row]
                # query_column=self.text_processor.process(query_column)
                #see what is the number of unique values in the query+ dl columns
                # we have a threshold to determine the smallness of domain called DS(domain size)
                domain_estimate=set.union(set(query_column),set(dl_column) )
                row_file.append(len(domain_estimate))
                if(len(domain_estimate)<DSize):
                    small_domain=1
                    distance=self.Jensen_Shannon_distances(query_column,dl_column,domain_estimate)
                    # log the domian infomrmation
                else: 
                    small_domain=0
                    distance=0
                    
                        

                   
           #'DL_table', 'DL_column',  'Q_table', 'Q_column','entropy','dSize','distance','Lexsim', 'SemSim', 'final_value'
            #now compute the lexical similarity 
            row_file.append(distance)
            lex_sim_=self._lexicalsim_Pair(table1_raw[row],dl_column_set)
            row_file.append(lex_sim_)
            row_file.append(semsim)

            penalized_score=self._penalize(semsim, lex_sim_, penalty_degree, distance,small_domain , self.entropy )
            row_file.append(penalized_score)

            csv_file = 'full_small_domain_domain_size_Threshold'+str(DSize)+'.csv'
            with open(csv_file, mode='a', newline='') as file:
                              writer = csv.writer(file)
                              writer.writerow(row_file)     
            score += penalized_score
        return score     
    
    def get_max(self,graph):
        flattened_graph = graph.flatten()

    # Filter out `disallowed_object` and create a set of unique values
        valid_values = set(value for value in flattened_graph if value != DISALLOWED)

    # Return the maximum value in the set
        return max(valid_values)
    
  
    
    
    
    
    def   Jensen_Shannon_distances(self,query_column,dl_column,domain_estimate):
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
        
    def _verify_greedy(self, table1, table2, threshold):
        nodes1 = set()
        nodes2 = set()
        score = 0.0
        nrow = len(table1)
        ncol = len(table2)
        edges = []
        for i in range(nrow):
            for j in range(ncol):
                sim = self._cosine_sim(table1[i],table2[j])
                if sim > threshold:
                    edges.append((sim,i,j))
                    nodes1.add(i)
                    nodes2.add(j)
        edges.sort(reverse=True)
        for e in edges:
            score += e[0]
            nodes1.discard(e[1])
            nodes2.discard(e[2])
            if len(nodes1) == 0 or len(nodes2) == 0:
                return score
        return score
    
    @staticmethod
    def read_csv_files_to_dict(folder_path):
        """
        Reads all CSV files from the specified folder and creates a dictionary
        mapping file names to lists of columns (including headers) from each file.

        Parameters:
            folder_path (str): The path to the folder containing CSV files.

        Returns:
            dict: A dictionary where keys are file names and values are lists of columns.
        """
        data_dict = {}
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, mode='r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    rows = list(reader)
                    if not rows:
                        continue  # Skip empty files
                    headers = rows[0]
                    # Initialize a list for each column, starting with the header
                    columns = [[header] for header in headers]
                    for row in rows[1:]:
                        for i, value in enumerate(row):
                            # Ensure we don't run into index errors if rows are uneven
                            if i < len(columns):
                                columns[i].append(value)
                            else:
                                # Handle missing columns by appending empty strings
                                columns.append([''] * (len(columns[0]) - 1) + [value])
                    data_dict[filename] = columns
        print(len(data_dict.keys()))
        return data_dict