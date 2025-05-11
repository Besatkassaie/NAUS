import utilities as utl   # assuming 'utl' is a custom module you have
# Load the dictionary
import os
from pathlib import Path

groundtruth_file = "/u6/bkassaie/NAUS/data/santos/santos_union_groundtruth.pickle"
groundtruth = utl.loadDictionaryFromPickleFile(groundtruth_file)

# Collect all unique strings from the values
all_values_set = set()
for value_list in groundtruth.values():
    all_values_set.update(value_list)

# Compute intersections
keys_set = set(groundtruth.keys())
common_elements = keys_set.intersection(all_values_set)

# Report
print(f"Number of keys: {len(keys_set)}")
print(f"Number of unique strings in values: {len(all_values_set)}")
print(f"Number of common elements between keys and values: {len(common_elements)}")



keys_set = set(groundtruth.keys())
values_set = set()
for vlist in groundtruth.values():
    values_set.update(vlist)

# Full set of valid filenames
valid_filenames = keys_set.union(values_set)

print("valid file name size"+str(len(valid_filenames)))

# Folder to clean
folder_x = "/u6/bkassaie/NAUS/data/santos/datalake"

# Iterate and remove invalid files
for file in os.listdir(folder_x):
    if file not in valid_filenames:
        file_path = os.path.join(folder_x, file)
        if os.path.isfile(file_path):
            print(f"Removing: {file_path}")
            os.remove(file_path)