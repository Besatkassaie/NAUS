from evaluate_novelty import *

import os 
import sys
import pandas as pd
import matplotlib.pyplot as plt
# examples of how to call evaluate fucntions 
 
  
 #             file=compute_metrics(results, "gmc","/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/evaluation_metrics_gmc.csv",gmc_search.k, gmc_search.unionability_scores, gmc_search.diversity_scores)
#             query_duplicate_returned("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/gmc_results.csv","/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/queryDuplicate_results.csv")

#    # evaluate the results 
# # count duplicate query 
#     Cal_P_R_Map("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/gmc_results.csv","data/santos/santosUnionBenchmark.pickle","/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/gmc_results_byQuery.csv" )


#         analyse("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/search_result_penalize_ByQuery.csv","/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/gmc_results_byQuery.csv")

# # count duplicate query 


   # make sure that we do not have extra character in result if you have remove them  
   # Cal_P_R_Map("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/search_result_penalize_diluted.csv","data/santos/santosUnionBenchmark.pickle","/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/penalized_results_byQuery_diluted.csv" )
    # query_duplicate_returned("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/search_result_penalize_diluted_restricted.csv",
    #                                     "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/search_result_penalize_diluted_restricted_duplicate.csv")

gmc_result_file="/u6/bkassaie/NAUS/data/ugen_v2/diveristy_data/search_results/GMC/gmc_results_diluted04_restricted.csv"
penalize_result_file="/u6/bkassaie/NAUS/data/ugen_v2/diveristy_data/search_results/Penalized/search_result_penalize_04diluted_restricted_pdeg1.csv"
starmie_result_file="/u6/bkassaie/NAUS/data/ugen_v2/diveristy_data/search_results/Starmie/starmie_results_04diluted_restricted.csv"
groundtruth="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_unionable_groundtruth_diluted.pickle"

gmc_diversity_data_path="data/ugen_v2/diveristy_data/search_results/GMC/"
penalized_diversity_data_path="data/ugen_v2/diveristy_data/search_results/Penalized/"
starmie_diversity_data_path="data/ugen_v2/diveristy_data/search_results/Starmie/"

import os

# make sure that we do not have extra character in result if you have remove them  
dup_pen_file=penalized_diversity_data_path+"search_result_new_penalize_diluted_restricted_duplicate.csv"
if not os.path.exists(dup_pen_file):

    query_duplicate_returned(penalize_result_file,dup_pen_file)
else:
    print("This file exists: "+dup_pen_file)
dup_gmc_file=   gmc_diversity_data_path+"search_result_gmc_new_diluted_restricted_duplicate.csv" 
    
if not os.path.exists(dup_gmc_file):
    query_duplicate_returned(gmc_result_file,dup_gmc_file)
else:
    print("This file exists: "+dup_gmc_file)    
    
starmie_dup_file=starmie_diversity_data_path+"search_result_starmie_diluted_restricted_duplicate.csv"
if not os.path.exists(starmie_dup_file):
    query_duplicate_returned(starmie_result_file,starmie_dup_file)
else:
    print("This file exists: "+starmie_dup_file) 
    
    
################Compute SNM######################    

starmie_snm_avg_file=starmie_diversity_data_path+"starmie_snm_diluted_restricted_avg_nodup.csv"
starmie_snm_whole_file=starmie_diversity_data_path+"starmie_snm_diluted_restricted_whole_nodup.csv"
if not (os.path.exists(starmie_snm_avg_file) or os.path.exists(starmie_snm_whole_file)):

    compute_syntactic_novelty_measure(groundtruth,starmie_result_file,starmie_snm_avg_file, starmie_snm_whole_file, remove_duplicate=1)    
else:
    print("This file exists: "+starmie_snm_avg_file+" or "+starmie_snm_whole_file) 
    
pnl_snm_avg_file=penalized_diversity_data_path+"new_pnl_snm_diluted_restricted_avg_nodup_pdg1.csv"
pnl_snm_whole_file=penalized_diversity_data_path+"new_pnl_snm_diluted_restricted_whole_nodup_pdg1.csv"
if not (os.path.exists(pnl_snm_whole_file) or os.path.exists(pnl_snm_avg_file)):

    compute_syntactic_novelty_measure(groundtruth,
                                                penalize_result_file,
                                                pnl_snm_avg_file
                                                    , pnl_snm_whole_file
                                                    , remove_duplicate=1)    
else:
    print("This file exists: "+pnl_snm_avg_file+" or "+pnl_snm_whole_file) 


################Compute SSNM######################    
gmc_ssnm_avg_file=gmc_diversity_data_path+"gmc_new_ssnm_diluted_restricted_avg_nodup.csv"
gmc_ssnm_whole_file=gmc_diversity_data_path+"gmc_new_ssnm_diluted_restricted_whole_nodup.csv"

if not (os.path.exists(gmc_ssnm_avg_file) or os.path.exists(gmc_ssnm_whole_file)):

    compute_syntactic_novelty_measure_simplified(groundtruth,gmc_result_file,gmc_ssnm_avg_file
                                                  , gmc_ssnm_whole_file
                                                  , remove_dup=1)  
else:
    print("This file exists: "+gmc_ssnm_avg_file+" or "+gmc_ssnm_whole_file)      

pnl_ssnm_avg_file=penalized_diversity_data_path+"new_pnl_ssnm_diluted_restricted_avg_nodup.csv"
pnl_ssnm_whole_file=penalized_diversity_data_path+"new_pnl_ssnm_diluted_restricted_whole_nodup.csv"
if not (os.path.exists(pnl_ssnm_avg_file) or os.path.exists(pnl_ssnm_whole_file)):
    
    compute_syntactic_novelty_measure_simplified(groundtruth,penalize_result_file,
                                                    pnl_ssnm_avg_file , 
                                                    pnl_ssnm_whole_file, remove_dup=1)    
else:
    print("This file exists: "+pnl_ssnm_avg_file+" or "+pnl_ssnm_whole_file)       
    

starmie_ssnm_avg_file=starmie_diversity_data_path+"starmie_ssnm_diluted_restricted_avg_nodup.csv"
starmie_ssnm_whole_file=starmie_diversity_data_path+"starmie_ssnm_diluted_restricted_whole_nodup.csv"
if not (os.path.exists(starmie_ssnm_avg_file) or os.path.exists(starmie_ssnm_whole_file)):

    compute_syntactic_novelty_measure_simplified(groundtruth,
                                                    starmie_result_file,starmie_ssnm_avg_file
                                                    , starmie_ssnm_whole_file
                                                    , remove_dup=1)    
else:
      print("This file exists: "+starmie_ssnm_avg_file+" or "+starmie_ssnm_whole_file)       
   
print("union size computation for Penalization")
alignemnt_path="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_CL_KMEANS_cosine_alignment_diluted.csv"
compute_union_size_with_null(penalize_result_file,   
                             penalized_diversity_data_path+"null_union_size_new_penalized_04diluted_restricted_notnormal.csv", 
                                alignemnt_path,
                                          "data/ugen_v2//query", 
                                         "data/ugen_v2/datalake",0) 

print("union size computation for Starmie")

compute_union_size_with_null(starmie_result_file,
                            starmie_diversity_data_path+"/null_union_size_starmie_04diluted_restricted_notnormal.csv", 
                                          alignemnt_path,
                                          "data/ugen_v2/query", 
                                         "data/ugen_v2/datalake",0) 

print("union size computation for GMC")

compute_union_size_with_null(gmc_result_file,
                             gmc_diversity_data_path+"/null_union_size_gmc_new_04diluted_restricted_notnormal.csv", 
                                           alignemnt_path,
                                          "data/ugen_v2/query", 
                                         "data/ugen_v2/datalake",0) 


# nscore_result(gmc_result_file,
#               gmc_diversity_data_path+"/nscore_gmc_04diluted_restricted_notnormal_K2K3_parallel.csv", 
#                                            "/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/tus_CL_KMEANS_cosine_alignment_all.csv",
#                                           "/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/query", 
#                                          "/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/datalake",1,1,0) 



#####execution time graph###############

time_pen_file=penalized_diversity_data_path+"time_new_penalize_diluted_restricted.csv"
if not os.path.exists(time_pen_file):
    Avg_executiontime_by_k(penalize_result_file, time_pen_file)
else:
    print("This file exists: "+time_pen_file)
    
    
time_gmc_file=   gmc_diversity_data_path+"time_gmc_new_diluted_restricted.csv" 
    
if not os.path.exists(time_gmc_file):
    Avg_executiontime_by_k(gmc_result_file, time_gmc_file)
else:
    print("This file exists: "+time_gmc_file)    


 