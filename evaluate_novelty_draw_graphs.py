
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#---------------------------- DRAW PLOT and Tables ---------------------------------------------------
def draw_plots():

    # File paths (update as needed)

    #base_output_path="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/diveristy_data/graphs"
    #base_output_path="/u6/bkassaie/NAUS/data/ugen_v2/diveristy_data/graphs"

    base_output_path="data/santos/small/diveristy_data/graphs"
    #base_output_path="data/table-union-search-benchmark/small/diveristy_data/graphs"

    #benchmark="/u6/bkassaie/NAUS/data/ugen_v2/"
    #benchmark="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/"
    #benchmark= "data/santos/"
    benchmark="data/table-union-search-benchmark/small/"
    
    benchmark_ugenv2="/u6/bkassaie/NAUS/data/ugen_v2/"
    benchmark_ugenv2_small="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/"
    benchmark_santos= "data/santos/small/"
    benchmark_tus="data/table-union-search-benchmark/small/"
    
    #--------------------- search result paths -------------------#
    pen_search_result_csv=benchmark+"diveristy_data/search_results/Penalized/search_result_penalize_diluted_restricted_duplicate.csv"
    star_search_result_csv=benchmark+"diveristy_data/search_results/Starmie/search_result_starmie_diluted_restricted_duplicate.csv"
    gmc_search_result_csv=benchmark+"diveristy_data/search_results/GMC/search_result_gmc_new_diluted_restricted_duplicate.csv"
    search_result_output_text=os.path.join(base_output_path,"search_result.tex")
    search_result_output_png=os.path.join(base_output_path,"search_result.png")
    
    
    #--------------------- Union Size ----------------------------#
    gmc_res_union_size = benchmark+"diveristy_data/search_results/GMC/null_union_size_gmc_new_04diluted_restricted_notnormal.csv"
    pnl_res_union_size = benchmark+"diveristy_data/search_results/Penalized/null_union_size_new_penalized_04diluted_restricted_notnormal.csv"
    starme_res_union_size = benchmark+"diveristy_data/search_results/Starmie/null_union_size_starmie_04diluted_restricted_notnormal.csv"
    
    #--------------------   File paths for SNM results -------------#
    pnl_res_snm = benchmark+"diveristy_data/search_results/Penalized/new_pnl_snm_diluted_restricted_avg_nodup_pdg1.csv"
    starme_res_snm = benchmark+"diveristy_data/search_results/Starmie/starmie_snm_diluted_restricted_avg_nodup.csv"

    #--------------------- File paths for ssnm results ---------------#
    gmc_res_ssnm = benchmark+"diveristy_data/search_results/GMC/gmc_new_ssnm_diluted_restricted_avg_nodup.csv"
    pnl_res_ssnm = benchmark+"diveristy_data/search_results/Penalized/new_pnl_ssnm_diluted_restricted_avg_nodup.csv"
    starme_res_ssnm =benchmark+ "diveristy_data/search_results/Starmie/starmie_ssnm_diluted_restricted_avg_nodup.csv"
    
    #--------------------File paths for exection time results---------------------#
    pnl_res_time = benchmark+"diveristy_data/search_results/Penalized/time_new_penalize_diluted_restricted.csv"
    gmc_res_time = benchmark+"diveristy_data/search_results/GMC/time_gmc_new_diluted_restricted.csv"
    

    
    
   
    
    draw_snm_all_individual("/u6/bkassaie/NAUS/graphs/snm",
                  benchmark_tus+"diveristy_data/search_results/Penalized/pnl_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  benchmark_santos+"diveristy_data/search_results/Penalized/pnl_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  None,
                  benchmark_ugenv2_small+"diveristy_data/search_results/Penalized/pnl_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  benchmark_tus+"diveristy_data/search_results/Starmie/starmie_snm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+ "diveristy_data/search_results/Starmie/starmie_snm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/Starmie/starmie_snm_diluted_restricted_avg_nodup.csv", 
                  benchmark_tus+"diveristy_data/search_results/starmie0/starmie0_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  benchmark_santos+"diveristy_data/search_results/starmie0/starmie0_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/starmie0/starmie0_snm_diluted_restricted_avg_nodup_pdg1.csv", 
                  benchmark_tus+"diveristy_data/search_results/starmie1/starmie1_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  benchmark_santos+"diveristy_data/search_results/starmie1/starmie1_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/starmie1/starmie1_snm_diluted_restricted_avg_nodup_pdg1.csv", 
                  benchmark_tus+"diveristy_data/search_results/semanticNovelty/semNovel_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  benchmark_santos+"diveristy_data/search_results/semanticNovelty/semNovel_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/semanticNovelty/semNovel_snm_diluted_restricted_avg_nodup_pdg1.csv")
    
    draw_snm_all_individual("/u6/bkassaie/NAUS/graphs/snm",
                  benchmark_tus+"diveristy_data/search_results/Penalized/pnl_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  benchmark_santos+"diveristy_data/search_results/Penalized/pnl_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  None,
                  benchmark_ugenv2_small+"diveristy_data/search_results/Penalized/pnl_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  benchmark_tus+"diveristy_data/search_results/Starmie/starmie_snm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+ "diveristy_data/search_results/Starmie/starmie_snm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/Starmie/starmie_snm_diluted_restricted_avg_nodup.csv", 
                  benchmark_tus+"diveristy_data/search_results/starmie0/starmie0_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  benchmark_santos+"diveristy_data/search_results/starmie0/starmie0_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/starmie0/starmie0_snm_diluted_restricted_avg_nodup_pdg1.csv", 
                  None,
                  None,
                  None,
                  None, 
                  benchmark_tus+"diveristy_data/search_results/semanticNovelty/semNovel_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  benchmark_santos+"diveristy_data/search_results/semanticNovelty/semNovel_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/semanticNovelty/semNovel_snm_diluted_restricted_avg_nodup_pdg1.csv")
    
    
    
    
    
    draw_snm_all_individual_filtered("/u6/bkassaie/NAUS/graphs/snm",
                  benchmark_tus+"diveristy_data/search_results/Penalized/pnl_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  benchmark_santos+"diveristy_data/search_results/Penalized/pnl_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  None,
                  benchmark_ugenv2_small+"diveristy_data/search_results/Penalized/pnl_snm_diluted_restricted_avg_nodup_pdg1_filtered.csv",
                  benchmark_tus+"diveristy_data/search_results/Starmie/starmie_snm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+ "diveristy_data/search_results/Starmie/starmie_snm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/Starmie/starmie_snm_diluted_restricted_avg_nodup_filtered.csv", 
                  benchmark_tus+"diveristy_data/search_results/starmie0/starmie0_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  benchmark_santos+"diveristy_data/search_results/starmie0/starmie0_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/starmie0/starmie0_snm_diluted_restricted_avg_nodup_pdg1_filtered.csv", 
                  benchmark_tus+"diveristy_data/search_results/starmie1/starmie1_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  benchmark_santos+"diveristy_data/search_results/starmie1/starmie1_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/starmie1/starmie1_snm_diluted_restricted_avg_nodup_pdg1_filtered.csv", 
                  benchmark_tus+"diveristy_data/search_results/semanticNovelty/semNovel_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  benchmark_santos+"diveristy_data/search_results/semanticNovelty/semNovel_snm_diluted_restricted_avg_nodup_pdg1.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/semanticNovelty/semNovel_snm_diluted_restricted_avg_nodup_pdg1_filtered.csv")
    
    


    draw_ssnm_individual("/u6/bkassaie/NAUS/graphs/ssnm",
                  benchmark_tus+"diveristy_data/search_results/Penalized/pnl_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+"diveristy_data/search_results/Penalized/pnl_ssnm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+"diveristy_data/search_results/Penalized/pnl_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_tus+"diveristy_data/search_results/semanticNovelty/semNovelty_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+"diveristy_data/search_results/semanticNovelty/semNovelty_ssnm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+"diveristy_data/search_results/semanticNovelty/semNovelty_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_tus+"diveristy_data/search_results/Starmie/starmie_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+ "diveristy_data/search_results/Starmie/starmie_ssnm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/Starmie/starmie_ssnm_diluted_restricted_avg_nodup.csv", 
                
                  benchmark_tus+"diveristy_data/search_results/starmie0/starmie0_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+ "diveristy_data/search_results/starmie0/starmie0_ssnm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/starmie0/starmie0_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_tus+"diveristy_data/search_results/starmie1/starmie1_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+ "diveristy_data/search_results/starmie1/starmie1_ssnm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/starmie1/starmie1_ssnm_diluted_restricted_avg_nodup.csv", 
                  benchmark_tus+"diveristy_data/search_results/GMC/gmc_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+"diveristy_data/search_results/GMC/gmc_ssnm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+"diveristy_data/search_results/GMC/gmc_ssnm_diluted_restricted_avg_nodup.csv"
                  ) 
    draw_ssnm_individual("/u6/bkassaie/NAUS/graphs/ssnm",
                  benchmark_tus+"diveristy_data/search_results/Penalized/pnl_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+"diveristy_data/search_results/Penalized/pnl_ssnm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+"diveristy_data/search_results/Penalized/pnl_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_tus+"diveristy_data/search_results/semanticNovelty/semNovelty_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+"diveristy_data/search_results/semanticNovelty/semNovelty_ssnm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+"diveristy_data/search_results/semanticNovelty/semNovelty_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_tus+"diveristy_data/search_results/Starmie/starmie_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+ "diveristy_data/search_results/Starmie/starmie_ssnm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/Starmie/starmie_ssnm_diluted_restricted_avg_nodup.csv", 
                
                  benchmark_tus+"diveristy_data/search_results/starmie0/starmie0_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+ "diveristy_data/search_results/starmie0/starmie0_ssnm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/starmie0/starmie0_ssnm_diluted_restricted_avg_nodup.csv",
                  None,
                  None,
                  None,
                  None, 
                  benchmark_tus+"diveristy_data/search_results/GMC/gmc_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+"diveristy_data/search_results/GMC/gmc_ssnm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+"diveristy_data/search_results/GMC/gmc_ssnm_diluted_restricted_avg_nodup.csv"
                  )    
              
    draw_ssnm_individual_filtered("/u6/bkassaie/NAUS/graphs/ssnm",
                  benchmark_tus+"diveristy_data/search_results/Penalized/pnl_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+"diveristy_data/search_results/Penalized/pnl_ssnm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+"diveristy_data/search_results/Penalized/pnl_ssnm_diluted_restricted_avg_nodup_filtered.csv",
                  benchmark_tus+"diveristy_data/search_results/semanticNovelty/semNovelty_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+"diveristy_data/search_results/semanticNovelty/semNovelty_ssnm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+"diveristy_data/search_results/semanticNovelty/semNovelty_ssnm_diluted_restricted_avg_nodup_filtered.csv",
                  benchmark_tus+"diveristy_data/search_results/Starmie/starmie_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+ "diveristy_data/search_results/Starmie/starmie_ssnm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/Starmie/starmie_ssnm_diluted_restricted_avg_nodup_filtered.csv", 
                  benchmark_tus+"diveristy_data/search_results/starmie0/starmie0_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+ "diveristy_data/search_results/starmie0/starmie0_ssnm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/starmie0/starmie0_ssnm_diluted_restricted_avg_nodup_filtered.csv",
                  benchmark_tus+"diveristy_data/search_results/starmie1/starmie1_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+ "diveristy_data/search_results/starmie1/starmie1_ssnm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+ "diveristy_data/search_results/starmie1/starmie1_ssnm_diluted_restricted_avg_nodup_filtered.csv",
                  benchmark_tus+"diveristy_data/search_results/GMC/gmc_ssnm_diluted_restricted_avg_nodup.csv",
                  benchmark_santos+"diveristy_data/search_results/GMC/gmc_ssnm_diluted_restricted_avg_nodup.csv",
                  None,
                  benchmark_ugenv2_small+"diveristy_data/search_results/GMC/gmc_ssnm_diluted_restricted_avg_nodup_filtered.csv"
                  ) 
                 
    
    #   # base_path,
   
   
   
    #                     gmc_res1, gmc_res2, gmc_res3, gmc_res4,
    #                     pnl_res1, pnl_res2, pnl_res3, pnl_res4
    # dataset_names = ["TUS", "Santos", "UgenV2", "UgenV2 small"] 
    
    # draw_execution_time_all("/u6/bkassaie/NAUS/graphs/executionTime", 
    #                         "data/table-union-search-benchmark/small/diveristy_data/search_results/GMC/time_gmc_new_diluted_restricted.csv"
    #                         ,"data/santos/diveristy_data/search_results/GMC/time_gmc_new_diluted_restricted.csv", 
    #                         "data/ugen_v2/diveristy_data/search_results/GMC/time_gmc_new_diluted_restricted.csv", 
    #                         "data/ugen_v2/ugenv2_small/diveristy_data/search_results/GMC/time_gmc_new_diluted_restricted.csv", 
    #                         "data/table-union-search-benchmark/small/diveristy_data/search_results/Penalized/time_new_penalize_diluted_restricted.csv", 
    #                         "data/santos/diveristy_data/search_results/Penalized/time_new_penalize_diluted_restricted.csv",
    #                         "data/ugen_v2/diveristy_data/search_results/Penalized/time_new_penalize_diluted_restricted.csv",
    #                         "data/ugen_v2/ugenv2_small/diveristy_data/search_results/Penalized/time_new_penalize_diluted_restricted.csv", 
    #                         "data/table-union-search-benchmark/small/diveristy_data/search_results/semanticNovelty/time_semNov_diluted_restricted.csv",
    #                         "data/santos/diveristy_data/search_results/semanticNovelty/time_semNov_diluted_restricted.csv",
    #                         "data/ugen_v2/diveristy_data/search_results/semanticNovelty/time_semNov_diluted_restricted.csv", 
    #                         "data/ugen_v2/ugenv2_small/diveristy_data/search_results/semanticNovelty/time_semNov_diluted_restricted.csv"
    #                         )
    
    # for i, res in enumerate([gmc_res1, gmc_res2, gmc_res3, gmc_res4]):
    #     add_result(res, 'GMC', dataset_names[i])
    # for i, res in enumerate([pnl_res1, pnl_res2, pnl_res3, pnl_res4]):
    #     add_result(res, 'ANTs', dataset_names[i])
    # for i, res in enumerate([semnovel_res1, semnovel_res2, semnovel_res3, semnovel_res4]):
    #     add_result(res, 'SemNovel', dataset_names[i])
    # for i, res in enumerate([starmie0_res1, starmie0_res2, starmie0_res3, starmie0_res4]):
    #     add_result(res, 'Starmie0', dataset_names[i])
    # for i, res in enumerate([starmie1_res1, starmie1_res2, starmie1_res3, starmie1_res4]):
    #     add_result(res, 'Starmie1', dataset_names[i])

    #dataset_names = ["TUS", "Santos", "UgenV2", "UgenV2 small"]

    
    

    
    
    draw_execution_time_all("/u6/bkassaie/NAUS/graphs/executionTime", 
                            "data/table-union-search-benchmark/small/diveristy_data/search_results/GMC/time_gmc_diluted_restricted.csv"
                            ,"data/santos/small/diveristy_data/search_results/GMC/time_gmc_diluted_restricted.csv", 
                            None, 
                           None, 
                            "data/table-union-search-benchmark/small/diveristy_data/search_results/Penalized/time_penalize_diluted_restricted.csv", 
                            "data/santos/small/diveristy_data/search_results/Penalized/time_penalize_diluted_restricted.csv",
                           None,
                          None, 
                            "data/table-union-search-benchmark/small/diveristy_data/search_results/semanticNovelty/time_semNov_diluted_restricted.csv",
                            "data/santos/small/diveristy_data/search_results/semanticNovelty/time_semNov_diluted_restricted.csv",
                           None, 
                          None,
                            "data/table-union-search-benchmark/small/diveristy_data/search_results/starmie0/time_starmie0_diluted_restricted.csv",
                            "data/santos/small/diveristy_data/search_results/starmie0/time_starmie0_diluted_restricted.csv",
                           None, 
                          None, 
                             "data/table-union-search-benchmark/small/diveristy_data/search_results/starmie1/time_starmie1_diluted_restricted.csv",
                            "data/santos/small/diveristy_data/search_results/starmie1/time_starmie1_diluted_restricted.csv",
                           None, 
                          None, 
                            "data/table-union-search-benchmark/small/diveristy_data/search_results/Starmie/time_starmie_04diluted_restricted.csv",
                            "data/santos/small/diveristy_data/search_results/Starmie/time_starmie_04diluted_restricted.csv",
                           None, 
                          None
                            ,"longest")
    
    
    draw_execution_time_all("/u6/bkassaie/NAUS/graphs/executionTime", 
                            "data/table-union-search-benchmark/small/diveristy_data/search_results/GMC/time_gmc_diluted_restricted.csv"
                            ,"data/santos/small/diveristy_data/search_results/GMC/time_gmc_diluted_restricted.csv", 
                            None, 
                           None, 
                            "data/table-union-search-benchmark/small/diveristy_data/search_results/Penalized/time_penalize_diluted_restricted.csv", 
                            "data/santos/small/diveristy_data/search_results/Penalized/time_penalize_diluted_restricted.csv",
                           None,
                          None, 
                            "data/table-union-search-benchmark/small/diveristy_data/search_results/semanticNovelty/time_semNov_diluted_restricted.csv",
                            "data/santos/small/diveristy_data/search_results/semanticNovelty/time_semNov_diluted_restricted.csv",
                           None, 
                          None,
                            "data/table-union-search-benchmark/small/diveristy_data/search_results/starmie0/time_starmie0_diluted_restricted.csv",
                            "data/santos/small/diveristy_data/search_results/starmie0/time_starmie0_diluted_restricted.csv",
                           None, 
                          None, 
                           None,
                          None,
                           None, 
                          None, 
                            "data/table-union-search-benchmark/small/diveristy_data/search_results/Starmie/time_starmie_04diluted_restricted.csv",
                            "data/santos/small/diveristy_data/search_results/Starmie/time_starmie_04diluted_restricted.csv",
                           None, 
                          None
                        ,"longest")
    
    # draw_execution_time_all("/u6/bkassaie/NAUS/graphs/executionTime", 
    #                         None
    #                         ,None, 
    #                         None, 
    #                        None, 
    #                         "data/table-union-search-benchmark/small/diveristy_data/search_results/Penalized/time_penalize_diluted_restricted.csv", 
    #                         "data/santos/diveristy_data/search_results/Penalized/time_penalize_diluted_restricted.csv",
    #                        None,
    #                       None, 
    #                         "data/table-union-search-benchmark/small/diveristy_data/search_results/semanticNovelty/time_semNov_diluted_restricted.csv",
    #                         "data/santos/diveristy_data/search_results/semanticNovelty/time_semNov_diluted_restricted.csv",
    #                        None, 
    #                       None,
    #                         "data/table-union-search-benchmark/small/diveristy_data/search_results/starmie0/time_starmie0_diluted_restricted.csv",
    #                         "data/santos/diveristy_data/search_results/starmie0/time_starmie0_diluted_restricted.csv",
    #                        None, 
    #                       None, 
    #                          "data/table-union-search-benchmark/small/diveristy_data/search_results/starmie1/time_starmie1_diluted_restricted.csv",
    #                         "data/santos/diveristy_data/search_results/starmie1/time_starmie1_diluted_restricted.csv",
    #                        None, 
    #                       None
    #                         , "shortestTimes.png")
    
    
    # draw_execution_time_2("/u6/bkassaie/NAUS/graphs/executionTime", 
    #                         "data/table-union-search-benchmark/small/diveristy_data/search_results/Penalized/time_new_penalize_diluted_restricted.csv", 
    #                         "data/santos/diveristy_data/search_results/Penalized/time_new_penalize_diluted_restricted.csv",
    #                         "data/ugen_v2/diveristy_data/search_results/Penalized/time_new_penalize_diluted_restricted.csv",
    #                         "data/ugen_v2/ugenv2_small/diveristy_data/search_results/Penalized/time_new_penalize_diluted_restricted.csv", 
    #                         "data/table-union-search-benchmark/small/diveristy_data/search_results/semanticNovelty/time_semNov_diluted_restricted.csv",
    #                         "data/santos/diveristy_data/search_results/semanticNovelty/time_semNov_diluted_restricted.csv",
    #                         "data/ugen_v2/diveristy_data/search_results/semanticNovelty/time_semNov_diluted_restricted.csv", 
    #                         "data/ugen_v2/ugenv2_small/diveristy_data/search_results/semanticNovelty/time_semNov_diluted_restricted.csv"
    #                         )
    # draw_execution_time_all_log("/u6/bkassaie/NAUS/graphs/executionTime", 
    #                         "data/table-union-search-benchmark/small/diveristy_data/search_results/GMC/time_gmc_new_diluted_restricted.csv"
    #                         ,"data/santos/diveristy_data/search_results/GMC/time_gmc_new_diluted_restricted.csv", 
    #                         "data/ugen_v2/diveristy_data/search_results/GMC/time_gmc_new_diluted_restricted.csv", 
    #                         "data/ugen_v2/ugenv2_small/diveristy_data/search_results/GMC/time_gmc_new_diluted_restricted.csv", 
    #                         "data/table-union-search-benchmark/small/diveristy_data/search_results/Penalized/time_new_penalize_diluted_restricted.csv", 
    #                         "data/santos/diveristy_data/search_results/Penalized/time_new_penalize_diluted_restricted.csv",
    #                         "data/ugen_v2/diveristy_data/search_results/Penalized/time_new_penalize_diluted_restricted.csv",
    #                         "data/ugen_v2/ugenv2_small/diveristy_data/search_results/Penalized/time_new_penalize_diluted_restricted.csv", 
    #                         "data/table-union-search-benchmark/small/diveristy_data/search_results/semanticNovelty/time_semNov_diluted_restricted.csv",
    #                         "data/santos/diveristy_data/search_results/semanticNovelty/time_semNov_diluted_restricted.csv",None, 
    #                         None  
    #                         )
    
 

def generate_k_table(pen_csv, star_csv, gmc_csv, output_tex, output_png):
    """
    Reads three CSV files (Penalization, Starmie, GMC) with columns:
       k, count
    Renames 'count' to the respective method name, merges on 'k',
    and produces:
      - A LaTeX table (written to output_tex) matching the style of your attached code.
      - A PNG image of the table (saved as output_png) with similar styling.
    """
    # 1. Read CSVs
    df_pen = pd.read_csv(pen_csv)   # Expecting columns: k, count
    df_star = pd.read_csv(star_csv) # Expecting columns: k, count
    df_gmc = pd.read_csv(gmc_csv)   # Expecting columns: k, count

    # 2. Rename 'count' to the method name
    df_pen.rename(columns={'count': 'Penalization'}, inplace=True)
    df_star.rename(columns={'count': 'Starmie'}, inplace=True)
    df_gmc.rename(columns={'count': 'GMC'}, inplace=True)

    # 3. Merge DataFrames on 'k'
    df_merged = pd.merge(df_pen, df_star, on='k', how='outer')
    df_merged = pd.merge(df_merged, df_gmc, on='k', how='outer')
    df_merged.sort_values('k', inplace=True)

    # 4. Generate LaTeX table code using your provided style.
    # Note: Adjust caption and label as needed.
    latex_table = r"""\begin{table}[H]
\centering
\begin{minipage}{0.3\textwidth}
\centering
\resizebox{\textwidth}{!}{%
    \begin{tabular}{|c|c|c|c|}
    \hline
    k & Penalization & Starmie & GMC \\ \hline \hline
"""
    # Append each row from the merged DataFrame.
    for _, row in df_merged.iterrows():
        latex_table += f"{row['k']} & {row['Penalization']} & {row['Starmie']} & {row['GMC']} \\\\ \n"
    latex_table += r"""\hline
    \end{tabular}
}
\caption{Number of queries where a duplicate of the query appears within the top k results on diluted dataset. There is an inconsistency between number of queries should be addressed!}
\label{tab:duplicateQuery_diluted dataset}
\end{minipage}
\end{table}
"""
    # Write the LaTeX code to the specified file.
    with open(output_tex, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table written to {output_tex}")

    # 5. Render the table as a PNG image using Matplotlib.
    # Set a figure size that approximates the minipage (adjust as needed).
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_axis_off()

    # Prepare table data: header row followed by data rows.
    headers = df_merged.columns.tolist()
    rows = df_merged.values.tolist()
    table_data = [headers] + rows

    # Create the table.
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Style each cell to add black borders and bold header.
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1)
        if row_idx == 0:
            cell.set_text_props(weight='bold')

    # Save the PNG image.
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Table PNG image saved to {output_png}")

    
def draw_union_size(base_path,gmc_res,pnl_res,starme_res):
    # gmc_res = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/GMC/null_union_size_gmc_04diluted_restricted_notnormal.csv"
    # pnl_res = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Penalized/null_union_size_penalized_04diluted_restricted_notnormal.csv"
    # starme_res = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Starmie/null_union_size_starmie_04diluted_restricted_notnormal.csv"
    
    # Read CSV files into DataFrames
    df_gmc = pd.read_csv(gmc_res)
    df_pnl = pd.read_csv(pnl_res)
    df_starme = pd.read_csv(starme_res)

    df_gmc=df_gmc.groupby("k").mean("null_union_size")/1000
    df_pnl=df_pnl.groupby("k").mean("null_union_size")/1000
    df_starme=df_starme.groupby("k").mean("null_union_size")/1000
    # Add a column to indicate the method/source for each row
    df_gmc['method'] = 'GMC'
    df_pnl['method'] = 'Pnl'
    df_starme['method'] = 'Starmie'

    
    # Combine the DataFrames into one
    df_combined = pd.concat([df_gmc.reset_index(), df_pnl.reset_index(), df_starme.reset_index()], ignore_index=True)

    # For a plot like your Figure 8, we’ll assume the columns are named 'k' and 'ssnm'.
    # If they differ (e.g., 'novelty'), replace 'ssnm' below with the correct column name.
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # We can define custom markers, colors, etc. for each method
    style_map = {
        'GMC':      {'marker': '^', 'color': 'blue'},
        'Pnl':      {'marker': 'o', 'color': 'red'},
        'Starmie':  {'marker': 's', 'color': 'orange'}
    }

    # Plot each method separately for full control of style
    for method, style_info in style_map.items():
        df_subset = df_combined[df_combined['method'] == method]
        ax.plot(
            df_subset['k'], 
            df_subset['null_union_size'], 
            marker=style_info['marker'],
            color=style_info['color'],
            label=method
        )

    # Labeling and aesthetics
    ax.set_xlabel('K', fontsize=12)
    ax.set_ylabel('Union Size (*.0001)', fontsize=12)
    #ax.set_title('Figure 8: SSNM (Mean)', fontsize=14)

    # If you know your k ranges from 1 to 10, you can set:
    ax.set_xticks(range(1, 11))
    
    # If you want y-axis from 0 to 1:
    #ax.set_ylim([0, 1])

    # Enable grid
    ax.grid(True)

    # Add legend
    ax.legend()
    plt.savefig(os.path.join(base_path,"union_size.pdf"), format='pdf', bbox_inches='tight')
    # Optionally also save as PNG:
    plt.savefig(os.path.join(base_path,"union_size.png"), dpi=300, bbox_inches='tight')

    # Show the plot
    #plt.show()
def draw_execution_time(base_path,gmc_res,pnl_res):
    # Read CSV files into DataFrames
    df_gmc = pd.read_csv(gmc_res)
    df_pnl = pd.read_csv(pnl_res)

    # Add a column to indicate the method/source for each row
    df_gmc['method'] = 'GMC'
    df_pnl['method'] = 'Pnl'

    
    # Combine the DataFrames into one
    df_combined = pd.concat([df_gmc.reset_index(), df_pnl.reset_index()], ignore_index=True)

    # For a plot like your Figure 8, we’ll assume the columns are named 'k' and 'ssnm'.
    # If they differ (e.g., 'novelty'), replace 'ssnm' below with the correct column name.
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # We can define custom markers, colors, etc. for each method
    style_map = {
        'GMC':      {'marker': '^', 'color': 'blue'},
        'Pnl':      {'marker': 'o', 'color': 'red'}
    }

    # Plot each method separately for full control of style
    for method, style_info in style_map.items():
        df_subset = df_combined[df_combined['method'] == method]
        ax.plot(
            df_subset['k'], 
            df_subset['exec_time'], 
            marker=style_info['marker'],
            color=style_info['color'],
            label=method
        )

    # Labeling and aesthetics
    ax.set_xlabel('K', fontsize=12)
    ax.set_ylabel('Execution Time (sec)', fontsize=12)
    #ax.set_title('Figure 8: SSNM (Mean)', fontsize=14)

    # If you know your k ranges from 1 to 10, you can set:
    ax.set_xticks(range(1, 11))
    
    # If you want y-axis from 0 to 1:
    #ax.set_ylim([0, 1])

    # Enable grid
    ax.grid(True)

    # Add legend
    ax.legend()
    plt.savefig(os.path.join(base_path,"executionTime.pdf"), format='pdf', bbox_inches='tight')
    # Optionally also save as PNG:
    plt.savefig(os.path.join(base_path,"executionTime.png"), dpi=300, bbox_inches='tight')

    # Show the plot
    #plt.show()

    
def draw_snm(base_path,pnl_res,starme_res):
    # # File paths for SNM results
    # pnl_res = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Penalized/pnl_snm_diluted_restricted_avg_nodup_pdg1.csv"
    # starme_res = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Starmie/starmie_snm_diluted_restricted_avg_nodup.csv"

    # Read CSV files into DataFrames
    df_pnl = pd.read_csv(pnl_res)
    df_starme = pd.read_csv(starme_res)

    # Add a column to indicate the method/source for each row
    df_pnl['method'] = 'Pnl'
    df_starme['method'] = 'Starmie'

    # Combine the DataFrames into one
    df_combined = pd.concat([df_pnl, df_starme], ignore_index=True)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define custom markers and colors for each method
    style_map = {
        'Pnl': {'marker': 'o', 'color': 'red'},
        'Starmie': {'marker': 's', 'color': 'orange'}
    }

    # Plot each method separately for full control of style
    for method, style_info in style_map.items():
        df_subset = df_combined[df_combined['method'] == method]
        ax.plot(
            df_subset['k'],
            df_subset['avg_snm'],
            marker=style_info['marker'],
            color=style_info['color'],
            label=method
        )

    # Labeling and aesthetics
    ax.set_xlabel('K', fontsize=12)
    ax.set_ylabel('SNM', fontsize=12)
    ax.set_xticks(range(1, 11))  # Adjust if needed
    ax.set_ylim([0, 1])          # Adjust y-axis limits as required
    ax.grid(True)
    ax.legend()

    # Save the plot in PDF and PNG formats for LaTeX inclusion
    plt.savefig(os.path.join(base_path, "snm_mean.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(base_path, "snm_mean.png"), dpi=300, bbox_inches='tight')

    # Display the plot and wait for the user to close it
    # plt.show()
    # input()

def draw_ssnm(base_path,gmc_res,pnl_res,starme_res):
    # gmc_res = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/GMC/gmc_ssnm_diluted_restricted_avg_nodup.csv"
    # pnl_res = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Penalized/pnl_ssnm_diluted_restricted_avg_nodup.csv"
    # starme_res = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Starmie/starmie_ssnm_diluted_restricted_avg_nodup.csv"
    
    # Read CSV files into DataFrames
    df_gmc = pd.read_csv(gmc_res)
    df_pnl = pd.read_csv(pnl_res)
    df_starme = pd.read_csv(starme_res)

    # Add a column to indicate the method/source for each row
    df_gmc['method'] = 'GMC'
    df_pnl['method'] = 'Pnl'
    df_starme['method'] = 'Starmie'

    # Combine the DataFrames into one
    df_combined = pd.concat([df_gmc, df_pnl, df_starme], ignore_index=True)

    # For a plot like your Figure 8, we’ll assume the columns are named 'k' and 'ssnm'.
    # If they differ (e.g., 'novelty'), replace 'ssnm' below with the correct column name.
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # We can define custom markers, colors, etc. for each method
    style_map = {
        'GMC':      {'marker': '^', 'color': 'blue'},
        'Pnl':      {'marker': 'o', 'color': 'red'},
        'Starmie':  {'marker': 's', 'color': 'orange'}
    }

    # Plot each method separately for full control of style
    for method, style_info in style_map.items():
        df_subset = df_combined[df_combined['method'] == method]
        ax.plot(
            df_subset['k'], 
            df_subset['avg_snm'], 
            marker=style_info['marker'],
            color=style_info['color'],
            label=method
        )

    # Labeling and aesthetics
    ax.set_xlabel('K', fontsize=12)
    ax.set_ylabel('SSNM', fontsize=12)
    #ax.set_title('Figure 8: SSNM (Mean)', fontsize=14)

    # If you know your k ranges from 1 to 10, you can set:
    ax.set_xticks(range(1, 11))
    
    # If you want y-axis from 0 to 1:
    ax.set_ylim([0, 1])

    # Enable grid
    ax.grid(True)

    # Add legend
    ax.legend()
    plt.savefig(os.path.join(base_path,"ssnm_mean.pdf"), format='pdf', bbox_inches='tight')
    # Optionally also save as PNG:
    plt.savefig(os.path.join(base_path,"ssnm_mean.png"), dpi=300, bbox_inches='tight')

    # Show the plot
    # plt.show()


def draw_snm_all_2system(base_path,
                 pnl_path1, pnl_path2, pnl_path3, pnl_path4,
                 starme_path1, starme_path2, starme_path3, starme_path4):
    """
    Draws the average SNM curves for 8 CSV files:
      - The first 4 files correspond to the Pnl method (plotted in blue).
      - The last 4 files correspond to the Starmie method (plotted in purple).
    
    Each file represents a different dataset; different markers and line styles
    are used to distinguish these datasets. Matching markers and line styles
    are used for the same dataset between the two methods.
    
    The legend is configured to list the entries in two columns, showing the
    matching method entries side by side.
    
    Parameters:
      - base_path: directory where the output plots (PDF & PNG) will be saved.
      - pnl_path1...pnl_path4: file paths for the Pnl method.
      - starme_path1...starme_path4: file paths for the Starmie method.
    
    Assumes each CSV file contains at least two columns: 'k' and 'avg_snm'.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Define a list of markers (one for each dataset)
    markers = ["o", "s", "D", "^"]
    # Define corresponding line styles for each dataset:
    linestyles = ["-", "--", "-.", ":"]

    # Read and label the four Pnl datasets
    pnl_paths = [pnl_path1, pnl_path2, pnl_path3, pnl_path4]
    pnl_dfs = []
    for i, path in enumerate(pnl_paths):
        df = pd.read_csv(path)
        df['method'] = 'ANTs'
        if i == 0:
            df['dataset'] = "TUS"
        elif i == 1:
            df['dataset'] = "Santos"
        elif i == 2:
            df['dataset'] = "UgenV2"
        elif i == 3:
            df['dataset'] = "UgenV2 small"
        pnl_dfs.append(df)

    # Read and label the four Starmie datasets
    starme_paths = [starme_path1, starme_path2, starme_path3, starme_path4]
    starme_dfs = []
    for i, path in enumerate(starme_paths):
        df = pd.read_csv(path)
        df['method'] = 'Starmie'
        if i == 0:
            df['dataset'] = "TUS"
        elif i == 1:
            df['dataset'] = "Santos"
        elif i == 2:
            df['dataset'] = "UgenV2"
        elif i == 3:
            df['dataset'] = "UgenV2 small"
        starme_dfs.append(df)

    # Create figure and axis for the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Instead of separate loops, plot data for corresponding datasets together.
    # This ensures that each dataset pair (ANTs and Starmie) appear side by side in the legend.
    for i in range(4):
        # Plot ANTs (Pnl) dataset with blue color.
        ax.plot(pnl_dfs[i]['k'], pnl_dfs[i]['avg_snm'],
                marker=markers[i],
                linestyle=linestyles[i],
                color='green',
                label=f"ANTs - {pnl_dfs[i]['dataset'].iloc[0]}")
        # Plot Starmie dataset with purple color.
        ax.plot(starme_dfs[i]['k'], starme_dfs[i]['avg_snm'],
                marker=markers[i],
                linestyle=linestyles[i],
                color='blue',
                label=f"Starmie - {starme_dfs[i]['dataset'].iloc[0]}")

    # Set axis labels and ticks
    ax.set_xlabel('l', fontsize=12)
    ax.set_ylabel('SNM', fontsize=12)
    ax.set_xticks(range(1, 11))  # Adjust if your k-values range is different.
    ax.set_ylim([0, 1])          # Adjust y-axis limits as needed.
    ax.set_yticks(np.arange(0, 1.1, 0.1))  # y-axis ticks with 0.1 increments

    # Enable grid and legend; set legend to have two columns.
    ax.grid(True)
    ax.legend(ncol=2, fontsize=10)

    # Save the plot in both PDF and PNG formats.
    pdf_path = os.path.join(base_path, "snm_all.pdf")
    png_path = os.path.join(base_path, "snm_all.png")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

def draw_snm_all_individual(base_path,
                 pnl_path1=None, pnl_path2=None, pnl_path3=None, pnl_path4=None,
                 starme_path1=None, starme_path2=None, starme_path3=None, starme_path4=None,
                 starme0_path1=None, starme0_path2=None, starme0_path3=None, starme0_path4=None,
                 starme1_path1=None, starme1_path2=None, starme1_path3=None, starme1_path4=None,
                 semnov_path1=None, semnov_path2=None, semnov_path3=None, semnov_path4=None):
    """
    Draws separate SNM curves for each dataset, plotting all methods (ANTs, Starmie, Starmie0, StarmieOne, SemNov) on the same figure.
    Each dataset gets its own plot.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    method_info = {
        'ANTs': ('green', 'o'),
        'Starmie': ('blue', 's'),
        'StarmieGold': ('purple', '^'),
        'StarmieOne': ('cyan', 'v'),
        'SemNov': ('orange', 'D')
    }

    dataset_names = ["TUS", "Santos", "UgenV2", "UgenV2 small"]

    def read_files(paths, method_label):
        dfs = {}
        for i, path in enumerate(paths):
            if path is not None and os.path.isfile(path):
                df = pd.read_csv(path)
                dfs[i] = (method_label, df)
        return dfs

    pnl_paths = [pnl_path1, pnl_path2, pnl_path3, pnl_path4]
    starme_paths = [starme_path1, starme_path2, starme_path3, starme_path4]
    starme0_paths = [starme0_path1, starme0_path2, starme0_path3, starme0_path4]
    starme1_paths = [starme1_path1, starme1_path2, starme1_path3, starme1_path4]
    semnov_paths = [semnov_path1, semnov_path2, semnov_path3, semnov_path4]

    pnl_dfs = read_files(pnl_paths, 'ANTs')
    starme_dfs = read_files(starme_paths, 'Starmie')
    starme0_dfs = read_files(starme0_paths, 'StarmieGold')
    starme1_dfs = read_files(starme1_paths, 'StarmieOne')
    semnov_dfs = read_files(semnov_paths, 'SemNov')

    all_dfs = {}
    for d in [pnl_dfs, semnov_dfs, starme_dfs, starme0_dfs, starme1_dfs]:
        for idx, (method, df) in d.items():
            if idx not in all_dfs:
                all_dfs[idx] = []
            all_dfs[idx].append((method, df))

    os.makedirs(base_path, exist_ok=True)

    for idx, dataset_name in enumerate(dataset_names):
        if idx not in all_dfs:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        for method, df in all_dfs[idx]:
            color, marker = method_info.get(method, ('black', '^'))
            ax.plot(df['k'], df['avg_snm'],
                    marker=marker,
                    linestyle='-',
                    color=color,
                    label=method)

        #ax.set_title(f"{dataset_name}", fontsize=14)
        ax.set_xlabel('l', fontsize=12)
        ax.set_ylabel('SNM', fontsize=12)
        ax.set_xticks(range(1, 11))
        ax.set_ylim([0, 1.05])  # Extend slightly above 1
        ax.set_yticks(np.arange(0, 1.01, 0.05))  # Only up to 1.0 in ticks
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.6)
        ax.legend(fontsize=9, loc='lower left', frameon=False)
        if (starme1_path1!=None): 
            pdf_path = os.path.join(base_path, f"snm_{dataset_name.replace(' ', '_')}extended.pdf")
            png_path = os.path.join(base_path, f"snm_{dataset_name.replace(' ', '_')}extended.png")
        else : 
            pdf_path = os.path.join(base_path, f"snm_{dataset_name.replace(' ', '_')}.pdf")
            png_path = os.path.join(base_path, f"snm_{dataset_name.replace(' ', '_')}.png")      
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Plots saved to {base_path}")

def draw_snm_all_individual_filtered(base_path,
                 pnl_path1=None, pnl_path2=None, pnl_path3=None, pnl_path4=None,
                 starme_path1=None, starme_path2=None, starme_path3=None, starme_path4=None,
                 starme0_path1=None, starme0_path2=None, starme0_path3=None, starme0_path4=None,
                 starme1_path1=None, starme1_path2=None, starme1_path3=None, starme1_path4=None,
                 semnov_path1=None, semnov_path2=None, semnov_path3=None, semnov_path4=None):
    """
    Draws separate SNM curves for each dataset, plotting all methods (ANTs, Starmie, StarmieGold, StarmieOne, SemNov) on the same figure.
    Each dataset gets its own plot.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    method_info = {
        'ANTs': ('green', 'o'),
        'Starmie': ('blue', 's'),
        'StarmieGold': ('purple', '^'),
        'StarmieOne': ('cyan', 'v'),
        'SemNov': ('orange', 'D')
    }

    dataset_names = ["TUS", "Santos", "UgenV2", "UgenV2 small"]

    def read_files(paths, method_label):
        dfs = {}
        for i, path in enumerate(paths):
            if path is not None and os.path.isfile(path):
                df = pd.read_csv(path)
                dfs[i] = (method_label, df)
        return dfs

    pnl_paths = [pnl_path1, pnl_path2, pnl_path3, pnl_path4]
    starme_paths = [starme_path1, starme_path2, starme_path3, starme_path4]
    starme0_paths = [starme0_path1, starme0_path2, starme0_path3, starme0_path4]
    starme1_paths = [starme1_path1, starme1_path2, starme1_path3, starme1_path4]
    semnov_paths = [semnov_path1, semnov_path2, semnov_path3, semnov_path4]

    pnl_dfs = read_files(pnl_paths, 'ANTs')
    starme_dfs = read_files(starme_paths, 'Starmie')
    starme0_dfs = read_files(starme0_paths, 'StarmieGold')
    starme1_dfs = read_files(starme1_paths, 'StarmieOne')
    semnov_dfs = read_files(semnov_paths, 'SemNov')

    all_dfs = {}
    for d in [pnl_dfs, semnov_dfs, starme_dfs, starme0_dfs, starme1_dfs]:
        for idx, (method, df) in d.items():
            if idx not in all_dfs:
                all_dfs[idx] = []
            all_dfs[idx].append((method, df))

    os.makedirs(base_path, exist_ok=True)

    for idx, dataset_name in enumerate(dataset_names):
        if idx not in all_dfs:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        for method, df in all_dfs[idx]:
            color, marker = method_info.get(method, ('black', '^'))
            ax.plot(df['k'], df['avg_snm'],
                    marker=marker,
                    linestyle='-',
                    color=color,
                    label=method)

        #ax.set_title(f"{dataset_name}", fontsize=14)
        ax.set_xlabel('l', fontsize=12)
        ax.set_ylabel('SNM', fontsize=12)
        ax.set_xticks(range(1, 11))
        ax.set_ylim([0, 1.05])  # Extended just above 1 to show y=1 clearly
        ax.set_yticks(np.arange(0, 1.01, 0.05))  # Y-axis ticks only up to 1.0
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.6)
        ax.legend(fontsize=9, loc='lower left', frameon=False)

        pdf_path = os.path.join(base_path, f"snm_{dataset_name.replace(' ', '_')}.pdf")
        png_path = os.path.join(base_path, f"snm_{dataset_name.replace(' ', '_')}_filtered.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Plots saved to {base_path}")

def draw_ssnm_all(base_path,
                  method1_path1=None, method1_path2=None, method1_path3=None, method1_path4=None,
                  method2_path1=None, method2_path2=None, method2_path3=None, method2_path4=None,
                  method3_path1=None, method3_path2=None, method3_path3=None, method3_path4=None,
                  method4_path1=None, method4_path2=None, method4_path3=None, method4_path4=None):
    """
    Draws the average SNM curves for up to four methods (ANTs, Starmie, GMC, NovSem) across four datasets.
    
    Each dataset is distinguished by a unique marker and line style.
    Methods are colored differently.
    
    Skips any missing files.
    
    Parameters:
      - base_path: Directory where the output plots (PDF & PNG) will be saved.
      - *_path1...*_path4: File paths for each method (can be None if missing).
    
    Assumes each CSV file contains at least two columns: 'k' and 'avg_snm'.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Define markers and linestyles (based on dataset)
    markers = ["o", "s", "D", "^"]
    linestyles = ["-", "--", "-.", ":"]
    
    # Dataset names
    dataset_names = ["TUS", "Santos", "UgenV2", "UgenV2 small"]

    # Method mapping: method name -> (color)
    method_info = {
        'ANTs': 'blue',
        'Starmie': 'purple',
        'GMC': 'green',
        'NovSem': 'orange'
    }
    
    # Collect available DataFrames
    all_dfs = []

    # Helper function to load and tag DataFrame if file exists
    def add_df(file, method_name, dataset_name):
        if file is not None and os.path.isfile(file):
            df = pd.read_csv(file)
            df['method'] = method_name
            df['dataset'] = dataset_name
            all_dfs.append(df)

    # Load Method 1 (ANTs)
    for i, file in enumerate([method1_path1, method1_path2, method1_path3, method1_path4]):
        add_df(file, 'ANTs', dataset_names[i])

    # Load Method 2 (Starmie)
    for i, file in enumerate([method2_path1, method2_path2, method2_path3, method2_path4]):
        add_df(file, 'Starmie', dataset_names[i])

    # Load Method 3 (GMC)
    for i, file in enumerate([method3_path1, method3_path2, method3_path3, method3_path4]):
        add_df(file, 'GMC', dataset_names[i])

    # Load Method 4 (NovSem)
    for i, file in enumerate([method4_path1, method4_path2, method4_path3, method4_path4]):
        add_df(file, 'NovSem', dataset_names[i])

    if not all_dfs:
        print("No valid data files found to plot.")
        return

    # Combine into one DataFrame
    df_all = pd.concat(all_dfs, ignore_index=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each curve grouped by dataset and method
    for (dataset, method), df_group in df_all.groupby(['dataset', 'method']):
        # Choose marker and linestyle based on dataset
        dataset_idx = dataset_names.index(dataset)
        ax.plot(df_group['k'], df_group['avg_snm'],
                marker=markers[dataset_idx],
                linestyle=linestyles[dataset_idx],
                color=method_info.get(method, 'black'),  # fallback color black
                label=f"{method} - {dataset}")

    # Set labels and grid
    ax.set_xlabel('l', fontsize=12)
    ax.set_ylabel('SSNM', fontsize=12)
    ax.set_xticks(range(1, 11))
    ax.set_ylim([0, 1])
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True)

    # Legend settings: 4 columns if many methods
    ax.legend(ncol=4, fontsize=9)

    # Save plots
    os.makedirs(base_path, exist_ok=True)
    pdf_path = os.path.join(base_path, "ssnm_all_4sys.pdf")
    png_path = os.path.join(base_path, "ssnm_all_4sys.png")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()


def draw_ssnm_all_3system(base_path,
             method1_path1, method1_path2, method1_path3, method1_path4,
             method2_path1, method2_path2, method2_path3, method2_path4,
             method3_path1, method3_path2, method3_path3, method3_path4):
    """
    Draws the average SNM curves for 12 CSV files, corresponding to three methods:
      - The first 4 files correspond to Method 1 (plotted in blue; labeled "ANTs").
      - The next 4 files correspond to Method 2 (plotted in purple; labeled "Starmie").
      - The last 4 files correspond to Method 3 (plotted in green; labeled "GMC").
    
    Each CSV file represents a different dataset; the datasets are distinguished by unique markers and line styles.
    Matching markers and line styles are shared across the three methods for each dataset.
    
    The legend displays the entries in three columns so that the three method entries for each dataset are side by side.
    
    Parameters:
      - base_path: Directory where the output plots (PDF & PNG) will be saved.
      - method1_path1...method1_path4: File paths for the first method (ANTs).
      - method2_path1...method2_path4: File paths for the second method (Starmie).
      - method3_path1...method3_path4: File paths for the third method (GMC).
    
    Assumes each CSV file contains at least two columns: 'k' and 'avg_snm'.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Define markers and line styles (one for each dataset)
    markers = ["o", "s", "D", "^"]
    linestyles = ["-", "--", "-.", ":"]

    # Define dataset names for each of the 4 files
    dataset_names = ["TUS", "Santos", "UgenV2", "UgenV2 small"]

    # Read and label the four CSV files for Method 1 (ANTs)
    method1_paths = [method1_path1, method1_path2, method1_path3, method1_path4]
    method1_dfs = []
    for i, path in enumerate(method1_paths):
        df = pd.read_csv(path)
        df['method'] = 'ANTs'
        df['dataset'] = dataset_names[i]
        method1_dfs.append(df)

    # Read and label the four CSV files for Method 2 (Starmie)
    method2_paths = [method2_path1, method2_path2, method2_path3, method2_path4]
    method2_dfs = []
    for i, path in enumerate(method2_paths):
        df = pd.read_csv(path)
        df['method'] = 'Starmie'
        df['dataset'] = dataset_names[i]
        method2_dfs.append(df)

    # Read and label the four CSV files for Method 3 (GMC)
    method3_paths = [method3_path1, method3_path2, method3_path3, method3_path4]
    method3_dfs = []
    for i, path in enumerate(method3_paths):
        df = pd.read_csv(path)
        df['method'] = 'GMC'
        df['dataset'] = dataset_names[i]
        method3_dfs.append(df)

    # Create figure and axis for the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # For each dataset index, plot the curves for all three methods using the same marker and linestyle.
    # This will help group the legend entries side by side.
    for i in range(4):
        # Plot Method 1 (ANTs) in blue
        ax.plot(method1_dfs[i]['k'], method1_dfs[i]['avg_snm'],
                marker=markers[i],
                linestyle=linestyles[i],
                color='blue',
                label=f"ANTs - {method1_dfs[i]['dataset'].iloc[0]}")
        # Plot Method 2 (Starmie) in purple
        ax.plot(method2_dfs[i]['k'], method2_dfs[i]['avg_snm'],
                marker=markers[i],
                linestyle=linestyles[i],
                color='purple',
                label=f"Starmie - {method2_dfs[i]['dataset'].iloc[0]}")
        # Plot Method 3 (GMC) in green
        ax.plot(method3_dfs[i]['k'], method3_dfs[i]['avg_snm'],
                marker=markers[i],
                linestyle=linestyles[i],
                color='green',
                label=f"GMC - {method3_dfs[i]['dataset'].iloc[0]}")

    # Set axis labels and ticks
    ax.set_xlabel('l', fontsize=12)
    ax.set_ylabel('SSNM', fontsize=12)
    ax.set_xticks(range(1, 11))         # Adjust if your k-values range is different.
    ax.set_ylim([0, 1])                 # Adjust y-axis limits as needed.
    ax.set_yticks(np.arange(0, 1.1, 0.1))  # y-axis ticks with 0.1 increments

    # Enable grid
    ax.grid(True)

    # Configure the legend to have three columns so corresponding method entries appear side by side.
    ax.legend(ncol=3, fontsize=10)

    # Save the plot in both PDF and PNG formats
    pdf_path = os.path.join(base_path, "ssnm_all.pdf")
    png_path = os.path.join(base_path, "ssnm_all.png")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

def draw_ssnm_individual_3Systems(base_path,
                        method1_path1, method1_path2, method1_path3, method1_path4,
                        method2_path1, method2_path2, method2_path3, method2_path4,
                        method3_path1, method3_path2, method3_path3, method3_path4):
    """
    Draws individual plots for each dataset from 12 CSV files (three methods and four datasets):
      - The first 4 files correspond to Method 1 (plotted in blue; labeled "ANTs").
      - The next 4 files correspond to Method 2 (plotted in purple; labeled "Starmie").
      - The last 4 files correspond to Method 3 (plotted in green; labeled "GMC").
    
    For each dataset (TUS, Santos, UgenV2, UgenV2 small), a separate figure is created that 
    plots the average SNM curves from the three methods. Each figure includes a legend displaying
    the three methods.
    
    The plots are saved individually (e.g., snm_TUS.pdf / snm_TUS.png) in the given base path.
    
    Assumes each CSV file contains at least two columns: 'k' and 'avg_snm'.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    # Define markers and line styles for each dataset.
    markers = ["o", "s", "D", "^"]
    linestyles = ["-", "--", "-.", ":"]
    # Define the dataset names.
    dataset_names = ["TUS", "Santos", "UgenV2", "UgenV2 small"]

    # Helper function to read files and label them by dataset.
    def read_files(paths, method_label):
        dfs = []
        for i, path in enumerate(paths):
            df = pd.read_csv(path)
            df['method'] = method_label
            df['dataset'] = dataset_names[i]
            dfs.append(df)
        return dfs

    # Read CSV files for each method.
    method1_paths = [method1_path1, method1_path2, method1_path3, method1_path4]
    method2_paths = [method2_path1, method2_path2, method2_path3, method2_path4]
    method3_paths = [method3_path1, method3_path2, method3_path3, method3_path4]

    method1_dfs = read_files(method1_paths, 'ANTs')
    method2_dfs = read_files(method2_paths, 'Starmie')
    method3_dfs = read_files(method3_paths, 'GMC')

    # Loop over each dataset index to create a separate plot.
    for i in range(4):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot for Method 1 (ANTs) in blue.
        ax.plot(method1_dfs[i]['k'], method1_dfs[i]['avg_snm'],
                marker=markers[i],
                linestyle=linestyles[i],
                color='blue')
        # Plot for Method 2 (Starmie) in purple.
        ax.plot(method2_dfs[i]['k'], method2_dfs[i]['avg_snm'],
                marker=markers[i],
                linestyle=linestyles[i],
                color='purple')
        # Plot for Method 3 (GMC) in green.
        ax.plot(method3_dfs[i]['k'], method3_dfs[i]['avg_snm'],
                marker=markers[i],
                linestyle=linestyles[i],
                color='green')
        
        # Set axis labels and title.
        ax.set_xlabel('l', fontsize=12)
        ax.set_ylabel('SSNM', fontsize=12)
       #ax.set_title(f"Benchmark: {dataset_names[i]}", fontsize=14)
        ax.set_xticks(range(1, 11))      # Modify as needed based on your data.
        ax.set_ylim([0, 1])              # Modify as needed.
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(True)
        
        # Create custom legend handles for the methods.
        method_handles = [
            Line2D([], [], color='blue', linestyle='-', label="ANTs"),
            Line2D([], [], color='purple', linestyle='-', label="Starmie"),
            Line2D([], [], color='green', linestyle='-', label="GMC")
        ]
        # Place the legend in the bottom left corner.
        ax.legend(handles=method_handles, title="Method", loc='lower left', fontsize=10)
        
        # Construct file names using the dataset name.
        pdf_file = os.path.join(base_path, f"snm_{dataset_names[i]}.pdf")
        png_file = os.path.join(base_path, f"snm_{dataset_names[i]}.png")
        
        # Save the current figure.
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.close()


def draw_ssnm_individual(base_path,
                         method1_path1=None, method1_path2=None, method1_path3=None, method1_path4=None,
                         method2_path1=None, method2_path2=None, method2_path3=None, method2_path4=None,
                         method3_path1=None, method3_path2=None, method3_path3=None, method3_path4=None,
                         method4_path1=None, method4_path2=None, method4_path3=None, method4_path4=None,
                         method5_path1=None, method5_path2=None, method5_path3=None, method5_path4=None,
                         method6_path1=None, method6_path2=None, method6_path3=None, method6_path4=None):
    """
    Draws individual plots for each dataset from up to six methods:
      - ANTs (blue, o)
      - Starmie (purple, s)
      - GMC (green, D)
      - SemNov (orange, ^)
      - StarmieGold (cyan, v)
      - StarmieOne (brown, P)

    Skips missing files gracefully.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    #linestyles = ["-", "--", "-.", ":"]
    dataset_names = ["TUS", "Santos", "UgenV2", "UgenV2 small"]


    
    
    method_info = {
        'ANTs': {'color':'green', 'marker': 'o'},
        'Starmie': {'color':'blue', 'marker': 's'},
        'StarmieGold': {'color': 'purple', 'marker': '^'},
        'StarmieOne': {'color': 'cyan', 'marker': 'v'},
        'SemNov': {'color':'orange', 'marker': 'D'}, 
        'GMC': {'color':'gray', 'marker': 'P'}
    }

    def read_files(paths, method_label):
        dfs = []
        for i, path in enumerate(paths):
            if path is not None and os.path.isfile(path):
                df = pd.read_csv(path)
                df['method'] = method_label
                df['dataset'] = dataset_names[i]
                dfs.append((i, df))
        return dfs

    method1_paths = [method1_path1, method1_path2, method1_path3, method1_path4]
    method2_paths = [method2_path1, method2_path2, method2_path3, method2_path4]
    method3_paths = [method3_path1, method3_path2, method3_path3, method3_path4]
    method4_paths = [method4_path1, method4_path2, method4_path3, method4_path4]
    method5_paths = [method5_path1, method5_path2, method5_path3, method5_path4]
    method6_paths = [method6_path1, method6_path2, method6_path3, method6_path4]

    method1_dfs = read_files(method1_paths, 'ANTs')
    method3_dfs = read_files(method3_paths, 'Starmie')
    method6_dfs = read_files(method6_paths, 'GMC')
    method2_dfs = read_files(method2_paths, 'SemNov')
    method4_dfs = read_files(method4_paths, 'StarmieGold')
    method5_dfs = read_files(method5_paths, 'StarmieOne')

    all_methods = method1_dfs + method2_dfs + method3_dfs + method4_dfs + method5_dfs + method6_dfs

    for i, dataset_name in enumerate(dataset_names):
        fig, ax = plt.subplots(figsize=(8, 6))
        legend_handles = []
        used_methods = set()

        for dataset_idx, df in all_methods:
            if dataset_idx == i:
                method = df['method'].iloc[0]
                style = method_info.get(method, {})
                color = style.get('color', 'black')
                marker = style.get('marker', 'o')
                #linestyle = linestyles[i % len(linestyles)]

                ax.plot(df['k'], df['avg_snm'],
                        marker=marker,
                        color=color,
                        label=method)

                if method not in used_methods:
                    legend_handles.append(Line2D([], [], color=color,
                                                 marker=marker, label=method))
                    used_methods.add(method)

        ax.set_xlabel('l', fontsize=12)
        ax.set_ylabel('SSNM', fontsize=12)
        #ax.set_title(f"{dataset_name}", fontsize=14)
        ax.set_xticks(range(1, 11))
        ax.set_ylim([0, 1.03])  # ✅ Slightly extend above 1.0
        ax.set_yticks(np.arange(0, 1.01, 0.05))  # ✅ Limit y-tick labels to 1.0
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgray')

        if legend_handles:
            ax.legend(handles=legend_handles, loc='lower left', fontsize=10, title=None)

        os.makedirs(base_path, exist_ok=True)
        if (method5_path1!=None):
            png_file = os.path.join(base_path, f"ssnm_{dataset_name}Extended.png")
        else: 
            png_file = os.path.join(base_path, f"ssnm_{dataset_name}.png")    
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.close()
        
  
def draw_ssnm_individual_filtered(base_path,
                         method1_path1=None, method1_path2=None, method1_path3=None, method1_path4=None,
                         method2_path1=None, method2_path2=None, method2_path3=None, method2_path4=None,
                         method3_path1=None, method3_path2=None, method3_path3=None, method3_path4=None,
                         method4_path1=None, method4_path2=None, method4_path3=None, method4_path4=None,
                         method5_path1=None, method5_path2=None, method5_path3=None, method5_path4=None,
                         method6_path1=None, method6_path2=None, method6_path3=None, method6_path4=None):
    """
    Draws SSNM plots per dataset and method, skipping missing files gracefully.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    dataset_names = ["TUS", "Santos", "UgenV2", "UgenV2 small"]
     
    method_info = {
        'ANTs': {'color':'green', 'marker': 'o'},
        'Starmie': {'color':'blue', 'marker': 's'},
        'StarmieGold': {'color': 'purple', 'marker': '^'},
        'StarmieOne': {'color': 'cyan', 'marker': 'v'},
        'SemNov': {'color':'orange', 'marker': 'D'}, 
        'GMC': {'color':'gray', 'marker': 'P'}
    }

    def read_files(paths, method_label):
        dfs = []
        for i, path in enumerate(paths):
            if path is not None and os.path.isfile(path):
                df = pd.read_csv(path)
                df['method'] = method_label
                df['dataset'] = dataset_names[i]
                dfs.append((i, df))
        return dfs

    method1_paths = [method1_path1, method1_path2, method1_path3, method1_path4]
    method2_paths = [method2_path1, method2_path2, method2_path3, method2_path4]
    method3_paths = [method3_path1, method3_path2, method3_path3, method3_path4]
    method4_paths = [method4_path1, method4_path2, method4_path3, method4_path4]
    method5_paths = [method5_path1, method5_path2, method5_path3, method5_path4]
    method6_paths = [method6_path1, method6_path2, method6_path3, method6_path4]

    method1_dfs = read_files(method1_paths, 'ANTs')
    method3_dfs = read_files(method3_paths, 'Starmie')
    method2_dfs = read_files(method2_paths, 'SemNov')
    method4_dfs = read_files(method4_paths, 'StarmieGold')
    method5_dfs = read_files(method5_paths, 'StarmieOne')
    method6_dfs = read_files(method6_paths, 'GMC')

    all_methods = method1_dfs + method2_dfs + method3_dfs + method4_dfs + method5_dfs + method6_dfs

    for i, dataset_name in enumerate(dataset_names):
        fig, ax = plt.subplots(figsize=(8, 6))
        legend_handles = []
        used_methods = set()

        for dataset_idx, df in all_methods:
            if dataset_idx == i:
                method = df['method'].iloc[0]
                style = method_info.get(method, {})
                color = style.get('color', 'black')
                marker = style.get('marker', 'o')

                ax.plot(df['k'], df['avg_snm'],
                        marker=marker,
                        color=color,
                        label=method)

                if method not in used_methods:
                    legend_handles.append(Line2D([], [], color=color, 
                                                 marker=marker, label=method))
                    used_methods.add(method)

        ax.set_xlabel('l', fontsize=12)
        ax.set_ylabel('SSNM', fontsize=12)
        #ax.set_title(f"{dataset_name}", fontsize=14)
        ax.set_xticks(range(1, 11))
        ax.set_ylim([0, 1.03])  # ✅ Extend slightly beyond 1
        ax.set_yticks(np.arange(0, 1.01, 0.05))  # ✅ Max label at 1.0 only
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgray')

        if legend_handles:
            ax.legend(handles=legend_handles, loc='lower left', fontsize=10, title=None)

        os.makedirs(base_path, exist_ok=True)
        pdf_file = os.path.join(base_path, f"ssnm_{dataset_name}.pdf")
        png_file = os.path.join(base_path, f"ssnm_{dataset_name}_filtered.png")
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.close()
                
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def draw_execution_time_all(base_path,
                             gmc_res1=None, gmc_res2=None, gmc_res3=None, gmc_res4=None,
                             pnl_res1=None, pnl_res2=None, pnl_res3=None, pnl_res4=None,
                             semnovel_res1=None, semnovel_res2=None, semnovel_res3=None, semnovel_res4=None,
                             StarmieGold_res1=None, StarmieGold_res2=None, StarmieGold_res3=None, StarmieGold_res4=None,
                             StarmieOne_res1=None, StarmieOne_res2=None, StarmieOne_res3=None, StarmieOne_res4=None,
                             Starmie_res1=None, Starmie_res2=None, Starmie_res3=None, Starmie_res4=None,
                             filename=None):

    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    method_colors = {
        'ANTs': 'green',
        'Starmie': 'blue',
        'StarmieGold': 'purple',
        'StarmieOne': 'cyan',
        'SemNov': 'orange',
        'GMC': 'gray'
    }

    dataset_names = ["TUS", "Santos", "UgenV2", "UgenV2 small"]
    marker_styles = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>']

    # Generate unique markers per (method, dataset)
    method_dataset_markers = {}
    marker_index = 0
    for method in method_colors.keys():
        for dataset in dataset_names:
            method_dataset_markers[(method, dataset)] = marker_styles[marker_index % len(marker_styles)]
            marker_index += 1

    df_all_list = []

    def add_result(file, method_name, dataset_name):
        if file is not None and os.path.isfile(file):
            df = pd.read_csv(file)
            df['method'] = method_name
            df['dataset'] = dataset_name
            df_all_list.append(df)

    for i, res in enumerate([gmc_res1, gmc_res2, gmc_res3, gmc_res4]):
        add_result(res, 'GMC', dataset_names[i])
    for i, res in enumerate([pnl_res1, pnl_res2, pnl_res3, pnl_res4]):
        add_result(res, 'ANTs', dataset_names[i])
    for i, res in enumerate([semnovel_res1, semnovel_res2, semnovel_res3, semnovel_res4]):
        add_result(res, 'SemNov', dataset_names[i])
    for i, res in enumerate([StarmieGold_res1, StarmieGold_res2, StarmieGold_res3, StarmieGold_res4]):
        add_result(res, 'StarmieGold', dataset_names[i])
    for i, res in enumerate([StarmieOne_res1, StarmieOne_res2, StarmieOne_res3, StarmieOne_res4]):
        add_result(res, 'StarmieOne', dataset_names[i])
    for i, res in enumerate([Starmie_res1, Starmie_res2, Starmie_res3, Starmie_res4]):
        add_result(res, 'Starmie', dataset_names[i])

    if not df_all_list:
        print("No valid data files found to plot.")
        return

    df_all = pd.concat(df_all_list, ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    handles_labels = []
    for (dataset, method), df_group in df_all.groupby(['dataset', 'method']):
        color = method_colors.get(method, 'black')
        marker = method_dataset_markers.get((method, dataset), 'o')
        line, = ax.plot(df_group['k'], df_group['exec_time'],
                        marker=marker, color=color, label=f"{dataset} - {method}")
        handles_labels.append((f"{dataset} - {method}", line))

    # Sort legend by dataset, then method
    handles_labels.sort(key=lambda x: (dataset_names.index(x[0].split(' - ')[0]), x[0].split(' - ')[1]))
    labels, handles = zip(*[(label, handle) for label, handle in handles_labels])

    ax.set_xlabel('l', fontsize=12)
    ax.set_ylabel('Execution Time (sec)', fontsize=12)
    ax.set_xticks(np.arange(1, 11, 1))

    y_min, y_max = df_all["exec_time"].min(), df_all["exec_time"].max()
    ax.set_yticks(np.linspace(y_min, y_max, 20))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)

    ax.legend(handles, labels, loc='upper left', fontsize=8, frameon=False)

    os.makedirs(base_path, exist_ok=True)
    png_path = os.path.join(base_path, f"{filename}Extended.png" if StarmieOne_res1 is not None else f"{filename}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    
def draw_execution_time_2(base_path,
                             pnl_res1=None, pnl_res2=None, pnl_res3=None, pnl_res4=None,
                             semnovel_res1=None, semnovel_res2=None, semnovel_res3=None, semnovel_res4=None):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Dataset names and their colors
    dataset_names = ["TUS", "Santos", "UgenV2", "UgenV2 small"]
    color_map = {
        "TUS": "blue",
        "Santos": "red",
        "UgenV2": "green",
        "UgenV2 small": "orange"
    }

    # Markers and line styles for different systems
    marker_map = {
        "ANTs": "o",
        "SemNov": "s"
    }
    line_style_map = {
        "ANTs": "-",
        "SemNov": "-."
    }

    df_all_list = []

    def add_result(file, method_name, dataset_name):
        if file is not None and os.path.isfile(file):
            df = pd.read_csv(file)
            df['method'] = method_name
            df['dataset'] = dataset_name
            df_all_list.append(df)

    for i, res in enumerate([pnl_res1, pnl_res2, pnl_res3, pnl_res4]):
        add_result(res, 'ANTs', dataset_names[i])
    for i, res in enumerate([semnovel_res1, semnovel_res2, semnovel_res3, semnovel_res4]):
        add_result(res, 'SemNov', dataset_names[i])

    if not df_all_list:
        print("No valid data files found to plot.")
        return

    df_all = pd.concat(df_all_list, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 7))  # Adjust figure size as needed

    handles_labels = []
    for (dataset, method), df_group in df_all.groupby(['dataset', 'method']):
        line, = ax.plot(df_group['k'], df_group['exec_time'],
                        marker=marker_map.get(method, 'x'),
                        linestyle=line_style_map.get(method, '-'),
                        color=color_map.get(dataset, 'black'),
                        label=f"{dataset} - {method}")
        handles_labels.append((f"{dataset} - {method}", line))

    # Group by dataset then sort methods within each group
    handles_labels.sort(key=lambda x: (dataset_names.index(x[0].split(' - ')[0]), x[0].split(' - ')[1]))
    labels, handles = zip(*[(label, handle) for label, handle in handles_labels])

    ax.set_xlabel('l', fontsize=12)
    ax.set_ylabel('Execution Time (sec)', fontsize=12)

    # More granular and light gray grid
    ax.set_xticks(np.arange(1, 11, 1))
    y_min, y_max = df_all["exec_time"].min(), df_all["exec_time"].max()
    ax.set_yticks(np.linspace(y_min, y_max, 20))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)

    # Add legend inside the plot box (top-left), vertical, small font
    ax.legend(handles, labels,
              loc='upper left',
              fontsize=8,
              frameon=False)

    os.makedirs(base_path, exist_ok=True)
    pdf_path = os.path.join(base_path, "executionTime_singleGraph2System.pdf")
    png_path = os.path.join(base_path, "executionTime_singleGraph2System.png")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()    
def draw_execution_time_all_log(base_path,
                             gmc_res1=None, gmc_res2=None, gmc_res3=None, gmc_res4=None,
                             pnl_res1=None, pnl_res2=None, pnl_res3=None, pnl_res4=None,
                             semnovel_res1=None, semnovel_res2=None, semnovel_res3=None, semnovel_res4=None):
    """
    Draws execution time curves for up to three systems (GMC, ANTs, SemNovel) over four datasets on a single graph,
    with a logarithmic scale on the y-axis.
    """
    # Dataset names and their colors
    dataset_names = ["TUS", "Santos", "UgenV2", "UgenV2 small"]
    color_map = {
        "TUS": "blue",
        "Santos": "red",
        "UgenV2": "green",
        "UgenV2 small": "orange"
    }

    # Markers and line styles for different systems
    marker_map = {
        "GMC": "^",
        "ANTs": "o",
        "SemNovel": "s"
    }
    line_style_map = {
        "GMC": "--",
        "ANTs": "-",
        "SemNovel": "-."
    }

    df_all_list = []

    # Helper function to load and label data
    def add_result(file, method_name, dataset_name):
        if file is not None and os.path.isfile(file):
            df = pd.read_csv(file)
            df['method'] = method_name
            df['dataset'] = dataset_name
            df_all_list.append(df)

    # Load all results
    for i, res in enumerate([gmc_res1, gmc_res2, gmc_res3, gmc_res4]):
        add_result(res, 'GMC', dataset_names[i])
    for i, res in enumerate([pnl_res1, pnl_res2, pnl_res3, pnl_res4]):
        add_result(res, 'ANTs', dataset_names[i])
    for i, res in enumerate([semnovel_res1, semnovel_res2, semnovel_res3, semnovel_res4]):
        add_result(res, 'SemNovel', dataset_names[i])

    if not df_all_list:
        print("No valid data files found to plot.")
        return

    df_all = pd.concat(df_all_list, ignore_index=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each group
    for (dataset, method), df_group in df_all.groupby(['dataset', 'method']):
        ax.plot(df_group['k'], df_group['exec_time'],
                marker=marker_map.get(method, 'x'),
                linestyle=line_style_map.get(method, '-'),
                color=color_map.get(dataset, 'black'),
                label=f"{dataset} - {method}")

    # Set labels and grid
    ax.set_xlabel('l', fontsize=12)
    ax.set_ylabel('Execution Time (sec, log scale)', fontsize=12)
    ax.set_xticks(range(1, 11))
    ax.set_yscale('log')  # <--- HERE: Set y-axis to log scale
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)

    # Save plot
    os.makedirs(base_path, exist_ok=True)
    pdf_path = os.path.join(base_path, "executionTime_singleGraph3System_log.pdf")
    png_path = os.path.join(base_path, "executionTime_singleGraph3System_log.png")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def draw_execution_time_all_2system(base_path,
                        gmc_res1, gmc_res2, gmc_res3, gmc_res4,
                        pnl_res1, pnl_res2, pnl_res3, pnl_res4):
    """
    Draws execution time curves for four datasets from two systems (GMC and Pnl) on a single graph.
    
    For each dataset (TUS, Santos, UgenV2, UgenV2 small), a unique color is assigned.
    The two systems are distinguished by their markers and line styles.
    
    Parameters:
      - base_path: Directory where the output plots (PDF & PNG) will be saved.
      - gmc_res1...gmc_res4: File paths for the GMC system results (one per dataset).
      - pnl_res1...pnl_res4: File paths for the Pnl system results (one per dataset).
    
    Assumes each CSV file contains at least two columns: 'k' and 'exec_time'.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Define dataset names and assign a unique color to each dataset.
    dataset_names = ["TUS", "Santos", "UgenV2", "UgenV2 small"]
    color_map = {
        "TUS": "blue",
        "Santos": "red",
        "UgenV2": "green",
        "UgenV2 small": "orange"
    }
    
    # Define markers and line styles to differentiate the two systems.
    marker_map = {
        "GMC": "^",
        "ANTs": "o"
    }
    line_style_map = {
        "GMC": "--",   # Dashed for GMC
        "ANTs": "-"     # Solid for Pnl
    }
    
    # Read CSV files for the GMC system, add dataset label.
    df_gmc_list = []
    for i, res in enumerate([gmc_res1, gmc_res2, gmc_res3, gmc_res4]):
        df = pd.read_csv(res)
        df['method'] = 'GMC'
        df['dataset'] = dataset_names[i]
        df_gmc_list.append(df)
        
    # Read CSV files for the Pnl system, add dataset label.
    df_pnl_list = []
    for i, res in enumerate([pnl_res1, pnl_res2, pnl_res3, pnl_res4]):
        df = pd.read_csv(res)
        df['method'] = 'ANTs'
        df['dataset'] = dataset_names[i]
        df_pnl_list.append(df)
        
    # Combine all data into a single DataFrame.
    df_all = pd.concat(df_gmc_list + df_pnl_list, ignore_index=True)
    
    # Create the figure and axis for the single plot.
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group data by dataset and method to plot each series separately.
    for (dataset, method), df_group in df_all.groupby(['dataset', 'method']):
        ax.plot(df_group['k'], df_group['exec_time'],
                marker=marker_map[method],
                linestyle=line_style_map[method],
                color=color_map[dataset],
                label=f"{dataset} - {method}")
                
    # Set axis labels and other plot settings.
    ax.set_xlabel('l', fontsize=12)
    ax.set_ylabel('Execution Time (sec)', fontsize=12)
    ax.set_xticks(range(1, 11))
    ax.grid(True)
    
    # Create a legend and display it.
    ax.legend(fontsize=10)
    
    # Save the plot to PDF and PNG.
    pdf_path = os.path.join(base_path, "executionTime_singleGraph.pdf")
    png_path = os.path.join(base_path, "executionTime_singleGraph.png")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()  

# Call the function to draw the plot
if __name__ == "__main__":
    draw_plots()