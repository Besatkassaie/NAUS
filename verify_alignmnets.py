import pandas as pd

# Replace 'input.csv' with the path to your CSV file.
input_csv = '/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/santos/Santos_CL_KMEANS_euclidean_alignment.csv'

# Read the CSV file into a DataFrame.
df = pd.read_csv(input_csv)

# --- Condition 1 ---
# Group rows that share the same query_table_name, query_column, query_column#, and dl_table_name.
grouped1 = df.groupby(['query_table_name', 'query_column', 'query_column#', 'dl_table_name'])
# Filter groups where there is more than one unique combination of (dl_column#, dl_column).
condition1 = grouped1.filter(lambda group: group[['dl_column#', 'dl_column']].drop_duplicates().shape[0] > 1)

# --- Condition 2 ---
# Group rows that share the same query_table_name, dl_table_name, dl_column#, and dl_column.
grouped2 = df.groupby(['query_table_name', 'dl_table_name', 'dl_column#', 'dl_column'])
# Filter groups where there is more than one unique combination of (query_column, query_column#).
condition2 = grouped2.filter(lambda group: group[['query_column', 'query_column#']].drop_duplicates().shape[0] > 1)

# Print the results.
print("Rows meeting Condition 1 (same query_table_name, query_column, query_column#, dl_table_name but different dl_column#/dl_column):")
print(condition1)

print("\nRows meeting Condition 2 (same query_table_name, dl_table_name, dl_column#, dl_column but different query_column/query_column#):")
print(condition2)

# # Optionally, write the results to separate CSV files.
# condition1.to_csv('condition1.csv', index=False)
# condition2.to_csv('condition2.csv', index=False)