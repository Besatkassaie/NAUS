import utilities as utl   # assuming 'utl' is a custom module you have
# Load the dictionary
import os
from pathlib import Path


import pandas as pd

import pandas as pd


def filter_and_group_by_k_no_B(input_csv_path: str,
                                queries_to_exclude: set,
                                filtered_output_path: str):
    """
    Filters out rows with queries in `queries_to_exclude`, drops column 'B',
    and computes averages grouped by 'k' for columns ['snm', 'L', 'G'],
    with the average of 'snm' renamed to 'avg_snm'.
    
    Args:
        input_csv_path (str): Path to the input CSV.
        queries_to_exclude (set): Set of queries to remove.
        filtered_output_path (str): Path to save filtered data (without column 'B').
        group_output_path (str): Path to save grouped averages by 'k'.
    """
    # Read input CSV
    df = pd.read_csv(input_csv_path)

    # Drop column B if present
    if 'B' in df.columns:
        df = df.drop(columns=['B'])

    # Filter out excluded queries
    filtered_df = df[~df['query'].isin(queries_to_exclude)]

    # Group by 'k' and compute mean
    grouped_avg = filtered_df.groupby('k')[['snm', 'L', 'G']].mean().reset_index()

    # Rename snm column to avg_snm
    grouped_avg = grouped_avg.rename(columns={'snm': 'avg_snm'})

    # Save grouped average data
    grouped_avg.to_csv(filtered_output_path, index=False)
def filter_and_group_by_k(input_csv_path: str,
                          queries_to_exclude: set,
                          filtered_output_path: str):
    """
    Filters out rows with queries in `queries_to_exclude`, then computes averages grouped by 'k'.
    
    Args:
        input_csv_path (str): Path to the input CSV.
        queries_to_exclude (set): Set of queries to remove.
        filtered_output_path (str): Path to save filtered data.
        group_output_path (str): Path to save grouped averages by 'k'.
    """
    # Read input CSV
    df = pd.read_csv(input_csv_path)

    # Filter out excluded queries
    filtered_df = df[~df['query'].isin(queries_to_exclude)]


    # Group by 'k' and compute mean of numeric columns
    grouped_avg = filtered_df.groupby('k')[['snm', 'B', 'L', 'G']].mean().reset_index()
    grouped_avg = grouped_avg.rename(columns={'snm': 'avg_snm'})

    # Save grouped average data
    grouped_avg.to_csv(filtered_output_path, index=False)


groundtruth_file = "data/ugen_v2/ugenv2_small/ugenv2_small_unionable_groundtruth_diluted.pickle"
groundtruth = utl.loadDictionaryFromPickleFile(groundtruth_file)

# Collect all unique strings from the values
lessth=set()
for key in groundtruth:
    print(len(groundtruth[key]))
    print(key)
    if len(groundtruth[key]) <10: 
         lessth.add(key)

print("queries with less than 10 unionable")
print(lessth)
print(len(lessth))

# limit the SNM and SSNM reuslts to those with more than 10 unionable an write them down  over ugenv2_small dataset

pen_snm_in="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/diveristy_data/search_results/Penalized/pnl_snm_diluted_restricted_whole_nodup_pdg1.csv"
pen_ssnm_in="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/diveristy_data/search_results/Penalized/pnl_ssnm_diluted_restricted_whole_nodup.csv"

semnove_snm_in="data/ugen_v2/ugenv2_small/diveristy_data/search_results/semanticNovelty/semNovel_snm_diluted_restricted_whole_nodup_pdg1.csv"
semnove_ssnm_in="data/ugen_v2/ugenv2_small/diveristy_data/search_results/semanticNovelty/semNovelty_ssnm_diluted_restricted_whole_nodup.csv"

starmie_snm_in="data/ugen_v2/ugenv2_small/diveristy_data/search_results/Starmie/starmie_snm_diluted_restricted_whole_nodup.csv"
starmie_ssnm_in="data/ugen_v2/ugenv2_small/diveristy_data/search_results/Starmie/starmie_ssnm_diluted_restricted_whole_nodup.csv"

starmie0_snm_in="data/ugen_v2/ugenv2_small/diveristy_data/search_results/starmie0/starmie0_snm_diluted_restricted_whole_nodup_pdg1.csv"
starmie0_ssnm_in="data/ugen_v2/ugenv2_small/diveristy_data/search_results/starmie0/starmie0_ssnm_diluted_restricted_whole_nodup.csv"

starmie1_snm_in="data/ugen_v2/ugenv2_small/diveristy_data/search_results/starmie1/starmie1_snm_diluted_restricted_whole_nodup_pdg1.csv"
starmie1_ssnm_in="data/ugen_v2/ugenv2_small/diveristy_data/search_results/starmie1/starmie1_ssnm_diluted_restricted_whole_nodup.csv"
gmc_ssnm_in="data/ugen_v2/ugenv2_small/diveristy_data/search_results/GMC/gmc_new_ssnm_diluted_restricted_whole_nodup.csv"


pen_snm_out="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/diveristy_data/search_results/Penalized/pnl_snm_diluted_restricted_avg_nodup_pdg1_filtered.csv"
pen_ssnm_out="/u6/bkassaie/NAUS/data/ugen_v2/ugenv2_small/diveristy_data/search_results/Penalized/pnl_ssnm_diluted_restricted_avg_nodup_filtered.csv"

semnove_snm_out="data/ugen_v2/ugenv2_small/diveristy_data/search_results/semanticNovelty/semNovel_snm_diluted_restricted_avg_nodup_pdg1_filtered.csv"
semnove_ssnm_out="data/ugen_v2/ugenv2_small/diveristy_data/search_results/semanticNovelty/semNovelty_ssnm_diluted_restricted_avg_nodup_filtered.csv"

starmie_snm_out="data/ugen_v2/ugenv2_small/diveristy_data/search_results/Starmie/starmie_snm_diluted_restricted_avg_nodup_filtered.csv"
starmie_ssnm_out="data/ugen_v2/ugenv2_small/diveristy_data/search_results/Starmie/starmie_ssnm_diluted_restricted_avg_nodup_filtered.csv"

starmie0_snm_out="data/ugen_v2/ugenv2_small/diveristy_data/search_results/starmie0/starmie0_snm_diluted_restricted_avg_nodup_pdg1_filtered.csv"
starmie0_ssnm_out="data/ugen_v2/ugenv2_small/diveristy_data/search_results/starmie0/starmie0_ssnm_diluted_restricted_avg_nodup_filtered.csv"

starmie1_snm_out="data/ugen_v2/ugenv2_small/diveristy_data/search_results/starmie1/starmie1_snm_diluted_restricted_avg_nodup_pdg1_filtered.csv"
starmie1_ssnm_out="data/ugen_v2/ugenv2_small/diveristy_data/search_results/starmie1/starmie1_ssnm_diluted_restricted_avg_nodup_filtered.csv"

gmc_ssnm_out="data/ugen_v2/ugenv2_small/diveristy_data/search_results/GMC/gmc_new_ssnm_diluted_restricted_avg_nodup_filtered.csv"

# filter_and_group_by_k(pen_snm_in, lessth,pen_snm_out )
# filter_and_group_by_k_no_B(pen_ssnm_in, lessth,pen_ssnm_out )

# filter_and_group_by_k(semnove_snm_in, lessth,semnove_snm_out )
# filter_and_group_by_k_no_B(semnove_ssnm_in, lessth,semnove_ssnm_out )


filter_and_group_by_k(starmie_snm_in, lessth,starmie_snm_out )
filter_and_group_by_k_no_B(starmie_ssnm_in, lessth,starmie_ssnm_out )

# filter_and_group_by_k(starmie0_snm_in, lessth,starmie0_snm_out )
# filter_and_group_by_k_no_B(starmie0_ssnm_in, lessth,starmie0_ssnm_out )

# filter_and_group_by_k(starmie1_snm_in, lessth, starmie1_snm_out )
# filter_and_group_by_k_no_B(starmie1_ssnm_in, lessth,starmie1_ssnm_out )


# filter_and_group_by_k_no_B(gmc_ssnm_in, lessth,gmc_ssnm_out ) 
# # Compute intersections
# keys_set = set(groundtruth.keys())
# common_elements = keys_set.intersection(all_values_set)

# # Report
# print(f"Number of keys: {len(keys_set)}")
# print(f"Number of unique strings in values: {len(all_values_set)}")
# print(f"Number of common elements between keys and values: {len(common_elements)}")



# keys_set = set(groundtruth.keys())
# values_set = set()
# for vlist in groundtruth.values():
#     values_set.update(vlist)

# # Full set of valid filenames
# valid_filenames = keys_set.union(values_set)

# print("valid file name size"+str(len(valid_filenames)))

# # Folder to clean
# folder_x = "/u6/bkassaie/NAUS/data/santos/datalake"

# # Iterate and remove invalid files
# for file in os.listdir(folder_x):
#     if file not in valid_filenames:
#         file_path = os.path.join(folder_x, file)
#         if os.path.isfile(file_path):
#             print(f"Removing: {file_path}")
#             os.remove(file_path)