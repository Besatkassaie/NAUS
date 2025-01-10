from evaluate_novelty import *


# examples of how to call evaluate fucntions 
 
 
 #             file=compute_metrics(results, "gmc","/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/evaluation_metrics_gmc.csv",gmc_search.k, gmc_search.unionability_scores, gmc_search.diversity_scores)
#             query_duplicate_returned("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/gmc_results.csv","/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/queryDuplicate_results.csv")
#             Avg_executiontime_by_k("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/gmc_results.csv", "GMC")

#    # evaluate the results 
# # count duplicate query 
#     Cal_P_R_Map("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/gmc_results.csv","data/santos/santosUnionBenchmark.pickle","/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/gmc_results_byQuery.csv" )


#         analyse("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/search_result_penalize_ByQuery.csv","/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/gmc_results_byQuery.csv")



   # make sure that we do not have extra character in result if you have remove them  
   # Cal_P_R_Map("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/search_result_penalize_diluted.csv","data/santos/santosUnionBenchmark.pickle","/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/penalized_results_byQuery_diluted.csv" )
    # query_duplicate_returned("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/search_result_penalize_diluted_restricted.csv",
    #                                     "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/search_result_penalize_diluted_restricted_duplicate.csv")

    # compute_syntactic_novelty_measure_simplified("data/santos/santosUnionBenchmark.pickle",
    #                                               "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/search_result_penalize_diluted.csv",
    #                                               "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/penalized_ssnm_diluted_avg.csv", 
    #                                               "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/penalized_ssnm_diluted_whole.csv")    
    # print("next ...")    
    # compute_syntactic_novelty_measure("data/santos/santosUnionBenchmark.pickle",
    #                                               "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/search_result_penalize_diluted_restricted.csv",
    #                                             "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/penalized_snm_diluted_restricted_avg.csv", 
    #                                              "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/penalized_snm_diluted_restricted_whole.csv")    

    # compute_union_size_simple("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/gmc_results_diluted_restricted.csv",
    #                                       "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/union_size_gmc_normalized_restricted.csv", 
    #                                       "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/DUST_Alignment_Diluted_restricted.csv",
    #                                       "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/query", 
    #                                       "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/datalake",1) 
    
    
compute_union_size_simple("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/starmie/starmie_results_diluted_restricted.csv",
                                          "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/starmie/union_size_starmie_diluted_restricted_notnormal.csv", 
                                          "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/DUST_Alignment_Diluted_restricted.csv",
                                          "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/query", 
                                          "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/datalake",0) 