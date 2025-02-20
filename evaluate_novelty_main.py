from evaluate_novelty import *

import os 
import sys
import pandas as pd
import matplotlib.pyplot as plt
# examples of how to call evaluate fucntions 
 
 
 #             file=compute_metrics(results, "gmc","/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/evaluation_metrics_gmc.csv",gmc_search.k, gmc_search.unionability_scores, gmc_search.diversity_scores)
#             query_duplicate_returned("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/gmc_results.csv","/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/queryDuplicate_results.csv")
#             Avg_executiontime_by_k("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/gmc_results.csv", "GMC")

#    # evaluate the results 
# # count duplicate query 
#     Cal_P_R_Map("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/gmc_results.csv","data/santos/santosUnionBenchmark.pickle","/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/gmc_results_byQuery.csv" )


#         analyse("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/search_result_penalize_ByQuery.csv","/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/gmc/gmc_results_byQuery.csv")

# # count duplicate query 


   # make sure that we do not have extra character in result if you have remove them  
   # Cal_P_R_Map("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/search_result_penalize_diluted.csv","data/santos/santosUnionBenchmark.pickle","/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/penalized_results_byQuery_diluted.csv" )
    # query_duplicate_returned("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/search_result_penalize_diluted_restricted.csv",
    #                                     "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/diversity_data/search_result/penalized/search_result_penalize_diluted_restricted_duplicate.csv")

gmc_result_file="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/GMC/gmc_results_diluted04_restricted.csv"
penalize_result_file="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Penalized/search_result_penalize_04diluted_restricted_pdeg1.csv"
starmie_result_file="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Starmie/starmie_results_04diluted_restricted.csv"
groundtruth="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/santos_union_groundtruth_diluted.pickle"

gmc_diversity_data_path="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/GMC/"
penalized_diversity_data_path="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Penalized/"
starmie_diversity_data_path="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Starmie/"

import os

# make sure that we do not have extra character in result if you have remove them  
dup_pen_file=penalized_diversity_data_path+"search_result_penalize_diluted_restricted_duplicate.csv"
if not os.path.exists(dup_pen_file):

    query_duplicate_returned(penalize_result_file,dup_pen_file)
else:
    print("This file exists: "+dup_pen_file)
dup_gmc_file=   gmc_diversity_data_path+"search_result_gmc_diluted_restricted_duplicate.csv" 
    
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
   
#print("union size computation for Penalization")
# compute_union_size_with_null("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Penalized/search_result_penalize_04diluted_restricted_pdeg1.csv",
#                                           "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Penalized/null_union_size_penalized_04diluted_restricted_notnormal.csv", 
#                                            "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/Santos_dlt_CL_KMEANS_cosine_alignment.csv",
#                                           "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/query", 
#                                          "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/datalake",0) 

# print("union size computation for Starmie")

# compute_union_size_with_null("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Starmie/starmie_results_04diluted_restricted.csv",
#                                           "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Starmie/null_union_size_starmie_04diluted_restricted_notnormal.csv", 
#                                            "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/Santos_dlt_CL_KMEANS_cosine_alignment.csv",
#                                           "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/query", 
#                                          "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/datalake",0) 

# print("union size computation for GMC")

# compute_union_size_with_null("/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/GMC/gmc_results_diluted04_restricted.csv",
#                                           "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/GMC/null_union_size_gmc_04diluted_restricted_notnormal.csv", 
#                                            "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/Santos_dlt_CL_KMEANS_cosine_alignment.csv",
#                                           "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/query", 
#                                          "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/datalake",0) 


#---------------------------- DRAW PLOT and Tables ---------------------------------------------------
def draw_plots():

    # File paths (update as needed)

    base_output_path="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/graphs"
    
    #--------------------- search result paths -------------------#
    pen_search_result_csv="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Penalized/search_result_penalize_diluted_restricted_duplicate.csv"
    star_search_result_csv="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Starmie/search_result_starmie_diluted_restricted_duplicate.csv"
    gmc_search_result_csv="/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/GMC/search_result_gmc_diluted_restricted_duplicate.csv"
    search_result_output_text=os.path.join(base_output_path,"search_result.tex")
    search_result_output_png=os.path.join(base_output_path,"search_result.png")
    
    
    #--------------------- Union Size ----------------------------#
    gmc_res_union_size = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/GMC/null_union_size_gmc_04diluted_restricted_notnormal.csv"
    pnl_res_union_size = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Penalized/null_union_size_penalized_04diluted_restricted_notnormal.csv"
    starme_res_union_size = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Starmie/null_union_size_starmie_04diluted_restricted_notnormal.csv"
    
    #--------------------   File paths for SNM results -------------#
    pnl_res_snm = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Penalized/pnl_snm_diluted_restricted_avg_nodup_pdg1.csv"
    starme_res_snm = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Starmie/starmie_snm_diluted_restricted_avg_nodup.csv"

    #--------------------- File paths for ssnm results ---------------#
    gmc_res_ssnm = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/GMC/gmc_ssnm_diluted_restricted_avg_nodup.csv"
    pnl_res_ssnm = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Penalized/pnl_ssnm_diluted_restricted_avg_nodup.csv"
    starme_res_ssnm = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/diveristy_data/search_results/Starmie/starmie_ssnm_diluted_restricted_avg_nodup.csv"
    
    
    generate_k_table(pen_search_result_csv,star_search_result_csv,gmc_search_result_csv,search_result_output_text,search_result_output_png)
    draw_ssnm(base_output_path,gmc_res_ssnm,pnl_res_ssnm,starme_res_ssnm)
    draw_snm(base_output_path,pnl_res_snm,starme_res_snm)
    draw_union_size (base_output_path,gmc_res_union_size,pnl_res_union_size,starme_res_union_size)

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


# Call the function to draw the plot
if __name__ == "__main__":
    draw_plots()