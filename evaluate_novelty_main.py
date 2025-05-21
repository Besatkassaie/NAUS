from evaluate_novelty import *

import os 




 # make sure that we do not have extra character in result if you have remove them  

# #santos 
# datafolder="/u6/bkassaie/NAUS/data/santos/small/"
# gtruthfilename="santos_small_union_groundtruth_diluted.pickle"

#  tus small
datafolder="/u6/bkassaie/NAUS/data/table-union-search-benchmark/small/"
gtruthfilename="tus_small_noverlap_groundtruth_all.csv"

#ugen v2
# datafolder="/u6/bkassaie/NAUS/data/ugen_v2/"
# gtruthfilename="ugenv2_unionable_groundtruth_diluted.pickle"

#ugen v2 small
# datafolder="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/"
# gtruthfilename="ugenv2_small_unionable_groundtruth_diluted.pickle"


gmc_result_file=datafolder+"diveristy_data/search_results/GMC/gmc_results_diluted04_restricted.csv"
penalize_result_file=datafolder+"diveristy_data/search_results/Penalized/search_result_penalize_04diluted_restricted_pdeg1.csv"
starmie_result_file=datafolder+"diveristy_data/search_results/Starmie/starmie_results_04diluted_restricted.csv"
semanticNovelty_result_file=datafolder+"diveristy_data/search_results/semanticNovelty/search_result_semNovelty_04diluted_restricted_pdeg1.csv"
starmie0_result_file=datafolder+"diveristy_data/search_results/starmie0/search_result_starmie0_04diluted_restricted_pdeg1.csv"
starmie1_result_file=datafolder+"diveristy_data/search_results/starmie1/search_result_starmie1_04diluted_restricted_pdeg1.csv"


gmc_diversity_data_path=datafolder+"diveristy_data/search_results/GMC/"
penalized_diversity_data_path=datafolder+"diveristy_data/search_results/Penalized/"
starmie_diversity_data_path=datafolder+"diveristy_data/search_results/Starmie/"
semanticNovelty_diversity_data_path=datafolder+"diveristy_data/search_results/semanticNovelty/"
starmie0_diversity_data_path=datafolder+"diveristy_data/search_results/starmie0/"
starmie1_diversity_data_path=datafolder+"diveristy_data/search_results/starmie1/"


groundtruth=datafolder+gtruthfilename

# make sure that we do not have extra character in Starmie results if you have remove them  

#######################################Blatant Duplicate###########################

dup_pen_file=penalized_diversity_data_path+"search_result_penalize_diluted_restricted_duplicate.csv"
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
    
semanticNovelty_dup_file=semanticNovelty_diversity_data_path+"search_result_semNovel_diluted_restricted_duplicate.csv"
if not os.path.exists(semanticNovelty_dup_file):
    query_duplicate_returned(semanticNovelty_result_file,semanticNovelty_dup_file)
else:
    print("This file exists: "+semanticNovelty_dup_file)     
    
starmie0_dup_file=starmie0_diversity_data_path+"search_result_starmie0_diluted_restricted_duplicate.csv"
if not os.path.exists(starmie0_dup_file):
    query_duplicate_returned(starmie0_result_file,starmie0_dup_file)
else:
    print("This file exists: "+starmie0_dup_file)       
    
starmie1_dup_file=starmie1_diversity_data_path+"search_result_starmie1_diluted_restricted_duplicate.csv"
if not os.path.exists(starmie1_dup_file):
    query_duplicate_returned(starmie1_result_file,starmie1_dup_file)
else:
    print("This file exists: "+starmie1_dup_file)      
       
 
##########################Ugen-v2 small compute duplicate exclude some queries

# the queries having more than 12 unionable tables  to make sure they have 10 non duplicate unionable 
exclude_set={'Art-History_YZMEPGTH_ugen_v2.csv', 'Veterinary-Medicine_V4B1K1KD_ugen_v2.csv', 
             'Criminology_JHT51TIW_ugen_v2.csv', 'Cooking_ZUIB5SQ0_ugen_v2.csv', 'Architecture_EPZHPCF0_ugen_v2.csv', 
             'Horticulture_FAALFS04_ugen_v2.csv', 'Math_XSK28T7A_ugen_v2.csv', 'Economics_XSEF6Y39_ugen_v2.csv',
             'Philosophy_SK28T7A9_ugen_v2.csv', 'International-Relations_YFSPEFRJ_ugen_v2.csv', 'Religion_8F1CBFNO_ugen_v2.csv',
             'Fashion_80QMAA7H_ugen_v2.csv', 'Medicine_CVFJJ50Y_ugen_v2.csv', 'Culture_4QOD4K7Y_ugen_v2.csv',
             'Sociology_GIQHG9JR_ugen_v2.csv'}
 
dup_pen_file=penalized_diversity_data_path+"search_result_penalize_diluted_restricted_duplicate_excluded.csv"
if not os.path.exists(dup_pen_file):

    query_duplicate_returned_exclude(penalize_result_file,dup_pen_file, exclude_set)
else:
    print("This file exists: "+dup_pen_file)
dup_gmc_file=   gmc_diversity_data_path+"search_result_gmc_new_diluted_restricted_duplicate_excluded.csv" 
    
if not os.path.exists(dup_gmc_file):
    query_duplicate_returned_exclude(gmc_result_file,dup_gmc_file, exclude_set)
else:
    print("This file exists: "+dup_gmc_file)    
    
starmie_dup_file=starmie_diversity_data_path+"search_result_starmie_diluted_restricted_duplicate_excluded.csv"
if not os.path.exists(starmie_dup_file):
    query_duplicate_returned_exclude(starmie_result_file,starmie_dup_file,exclude_set)
else:
    print("This file exists: "+starmie_dup_file) 
    
semanticNovelty_dup_file=semanticNovelty_diversity_data_path+"search_result_semNovel_diluted_restricted_duplicate_excluded.csv"
if not os.path.exists(semanticNovelty_dup_file):
    query_duplicate_returned_exclude(semanticNovelty_result_file,semanticNovelty_dup_file, exclude_set)
else:
    print("This file exists: "+semanticNovelty_dup_file)     
    
starmie0_dup_file=starmie0_diversity_data_path+"search_result_starmie0_diluted_restricted_duplicate_excluded.csv"
if not os.path.exists(starmie0_dup_file):
    query_duplicate_returned_exclude(starmie0_result_file,starmie0_dup_file, exclude_set)
else:
    print("This file exists: "+starmie0_dup_file)       
    
starmie1_dup_file=starmie1_diversity_data_path+"search_result_starmie1_diluted_restricted_duplicate_excluded.csv"
if not os.path.exists(starmie1_dup_file):
    query_duplicate_returned_exclude(starmie1_result_file,starmie1_dup_file, exclude_set)
else:
    print("This file exists: "+starmie1_dup_file)          
       
       
       
################Compute SNM######################    

starmie_snm_avg_file=starmie_diversity_data_path+"starmie_snm_diluted_restricted_avg_nodup.csv"
starmie_snm_whole_file=starmie_diversity_data_path+"starmie_snm_diluted_restricted_whole_nodup.csv"
if not (os.path.exists(starmie_snm_avg_file) or os.path.exists(starmie_snm_whole_file)):

    compute_syntactic_novelty_measure(groundtruth,starmie_result_file,starmie_snm_avg_file, starmie_snm_whole_file, remove_duplicate=1)    
else:
    print("This file exists: "+starmie_snm_avg_file+" or "+starmie_snm_whole_file) 
    
pnl_snm_avg_file=penalized_diversity_data_path+"pnl_snm_diluted_restricted_avg_nodup_pdg1.csv"
pnl_snm_whole_file=penalized_diversity_data_path+"pnl_snm_diluted_restricted_whole_nodup_pdg1.csv"
if not (os.path.exists(pnl_snm_whole_file) or os.path.exists(pnl_snm_avg_file)):

    compute_syntactic_novelty_measure(groundtruth,
                                                penalize_result_file,
                                                pnl_snm_avg_file
                                                    , pnl_snm_whole_file
                                                    , remove_duplicate=1)    
else:
    print("This file exists: "+pnl_snm_avg_file+" or "+pnl_snm_whole_file) 

semNovelty_snm_avg_file=semanticNovelty_diversity_data_path+"semNovel_snm_diluted_restricted_avg_nodup_pdg1.csv"
semNovelty_snm_whole_file=semanticNovelty_diversity_data_path+"semNovel_snm_diluted_restricted_whole_nodup_pdg1.csv"
if not (os.path.exists(semNovelty_snm_whole_file) or os.path.exists(semNovelty_snm_avg_file)):

    compute_syntactic_novelty_measure(groundtruth,
                                                semanticNovelty_result_file,
                                                semNovelty_snm_avg_file
                                                    , semNovelty_snm_whole_file
                                                    , remove_duplicate=1)    
else:
    print("This file exists: "+semNovelty_snm_whole_file+" or "+semNovelty_snm_avg_file) 
    
    
starmie0_snm_avg_file=starmie0_diversity_data_path+"starmie0_snm_diluted_restricted_avg_nodup_pdg1.csv"
starmie0_snm_whole_file=starmie0_diversity_data_path+"starmie0_snm_diluted_restricted_whole_nodup_pdg1.csv"
if not (os.path.exists(starmie0_snm_whole_file) or os.path.exists(starmie0_snm_avg_file)):

    compute_syntactic_novelty_measure(groundtruth,
                                                starmie0_result_file,
                                                starmie0_snm_avg_file
                                                    , starmie0_snm_whole_file
                                                    , remove_duplicate=1)    
else:
    print("This file exists: "+starmie0_snm_whole_file+" or "+starmie0_snm_avg_file)   
    
    
starmie1_snm_avg_file=starmie1_diversity_data_path+"starmie1_snm_diluted_restricted_avg_nodup_pdg1.csv"
starmie1_snm_whole_file=starmie1_diversity_data_path+"starmie1_snm_diluted_restricted_whole_nodup_pdg1.csv"
if not (os.path.exists(starmie1_snm_whole_file) or os.path.exists(starmie1_snm_avg_file)):

    compute_syntactic_novelty_measure(groundtruth,
                                                starmie1_result_file,
                                                starmie1_snm_avg_file
                                                    , starmie1_snm_whole_file
                                                    , remove_duplicate=1)    
else:
    print("This file exists: "+starmie1_snm_whole_file+" or "+starmie1_snm_avg_file)        
    
################Compute SSNM######################    
gmc_ssnm_avg_file=gmc_diversity_data_path+"gmc_ssnm_diluted_restricted_avg_nodup.csv"
gmc_ssnm_whole_file=gmc_diversity_data_path+"gmc_ssnm_diluted_restricted_whole_nodup.csv"

if not (os.path.exists(gmc_ssnm_avg_file) or os.path.exists(gmc_ssnm_whole_file)):

    compute_syntactic_novelty_measure_simplified(groundtruth,gmc_result_file,gmc_ssnm_avg_file
                                                  , gmc_ssnm_whole_file
                                                  , remove_dup=1)  
else:
    print("This file exists: "+gmc_ssnm_avg_file+" or "+gmc_ssnm_whole_file)      

pnl_ssnm_avg_file=penalized_diversity_data_path+"pnl_ssnm_diluted_restricted_avg_nodup.csv"
pnl_ssnm_whole_file=penalized_diversity_data_path+"pnl_ssnm_diluted_restricted_whole_nodup.csv"
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
   

semNovelty_ssnm_avg_file=semanticNovelty_diversity_data_path+"semNovelty_ssnm_diluted_restricted_avg_nodup.csv"
semNovelty_ssnm_whole_file=semanticNovelty_diversity_data_path+"semNovelty_ssnm_diluted_restricted_whole_nodup.csv"
if not (os.path.exists(semNovelty_ssnm_avg_file) or os.path.exists(semNovelty_ssnm_whole_file)):

    compute_syntactic_novelty_measure_simplified(groundtruth,
                                                    semanticNovelty_result_file,semNovelty_ssnm_avg_file
                                                    , semNovelty_ssnm_whole_file
                                                    , remove_dup=1)    
else:
      print("This file exists: "+semNovelty_ssnm_avg_file+" or "+starmie_ssnm_whole_file)     



starmie0_ssnm_avg_file=starmie0_diversity_data_path+"starmie0_ssnm_diluted_restricted_avg_nodup.csv"
starmie0_ssnm_whole_file=starmie0_diversity_data_path+"starmie0_ssnm_diluted_restricted_whole_nodup.csv"
if not (os.path.exists(starmie0_ssnm_avg_file) or os.path.exists(starmie0_ssnm_whole_file)):

    compute_syntactic_novelty_measure_simplified(groundtruth,
                                                    starmie0_result_file,starmie0_ssnm_avg_file
                                                    , starmie0_ssnm_whole_file
                                                    , remove_dup=1)    
else:
      print("This file exists: "+starmie0_ssnm_avg_file+" or "+starmie0_ssnm_whole_file)     



starmie1_ssnm_avg_file=starmie1_diversity_data_path+"starmie1_ssnm_diluted_restricted_avg_nodup.csv"
starmie1_ssnm_whole_file=starmie1_diversity_data_path+"starmie1_ssnm_diluted_restricted_whole_nodup.csv"
if not (os.path.exists(starmie1_ssnm_avg_file) or os.path.exists(starmie1_ssnm_whole_file)):

    compute_syntactic_novelty_measure_simplified(groundtruth,
                                                    starmie1_result_file,starmie1_ssnm_avg_file
                                                    , starmie1_ssnm_whole_file
                                                    , remove_dup=1)    
else:
      print("This file exists: "+starmie1_ssnm_avg_file+" or "+starmie1_ssnm_whole_file) 
      
      
# print("union size computation for Penalization")
# alignemnt_path="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/ugenv2_small_manual_alignment_all.csv"
# compute_union_size_with_null(penalize_result_file,   
#                              penalized_diversity_data_path+"null_union_size_new_penalized_04diluted_restricted_notnormal.csv", 
#                                 alignemnt_path,
#                                           "data/ugen_v2/ugenv2_small/query", 
#                                          "data/ugen_v2/ugenv2_small/datalake",0) 

# print("union size computation for Starmie")

# compute_union_size_with_null(starmie_result_file,
#                             starmie_diversity_data_path+"/null_union_size_starmie_04diluted_restricted_notnormal.csv", 
#                                           alignemnt_path,
#                                           "data/ugen_v2/ugenv2_small/query", 
#                                          "data/ugen_v2/ugenv2_small/datalake",0) 

# print("union size computation for GMC")

# compute_union_size_with_null(gmc_result_file,
#                              gmc_diversity_data_path+"/null_union_size_gmc_new_04diluted_restricted_notnormal.csv", 
#                                            alignemnt_path,
#                                           "data/ugen_v2/ugenv2_small/query", 
#                                          "data/ugen_v2/ugenv2_small/datalake",0) 




#####execution time graph###############

time_pen_file=penalized_diversity_data_path+"time_penalize_diluted_restricted.csv"
if not os.path.exists(time_pen_file):
    Avg_executiontime_by_k(penalize_result_file, time_pen_file)
else:
    print("This file exists: "+time_pen_file)
    
    
time_gmc_file=   gmc_diversity_data_path+"time_gmc_diluted_restricted.csv" 
    
if not os.path.exists(time_gmc_file):
    Avg_executiontime_by_k(gmc_result_file, time_gmc_file)
else:
    print("This file exists: "+time_gmc_file)    
    
time_semNovelty_file= semanticNovelty_diversity_data_path+"time_semNov_diluted_restricted.csv" 
    
if not os.path.exists(time_semNovelty_file):
    Avg_executiontime_by_k(semanticNovelty_result_file, time_semNovelty_file)
else:
    print("This file exists: "+time_semNovelty_file)    
    
        
time_starmie0_file= starmie0_diversity_data_path+"time_starmie0_diluted_restricted.csv" 
    
if not os.path.exists(time_starmie0_file):
    Avg_executiontime_by_k(starmie0_result_file, time_starmie0_file)
else:
    print("This file exists: "+time_starmie0_file)   
    
        
time_starmie1_file= starmie1_diversity_data_path+"time_starmie1_diluted_restricted.csv" 
    
if not os.path.exists(time_starmie1_file):
    Avg_executiontime_by_k(starmie1_result_file, time_starmie1_file)
else:
    print("This file exists: "+time_starmie1_file)   


time_starmie_file= starmie_diversity_data_path+"time_starmie_04diluted_restricted.csv" 
    
if not os.path.exists(time_starmie_file):
    Avg_executiontime_by_k(starmie_result_file, time_starmie_file)
else:
    print("This file exists: "+time_starmie_file)           
         
    
    

 