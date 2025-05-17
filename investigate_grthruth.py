from collections import defaultdict
from itertools import combinations
import utilities as utl 
def find_common_values(input_dict):
    # To store results: key pairs and count of common values
    result = []

    # Generate all combinations of key pairs
    for key1, key2 in combinations(input_dict.keys(), 2):
        values1 = set(input_dict[key1])
        values2 = set(input_dict[key2])
        common = values1 & values2  # Intersection
        if common:
            result.append({
                'keys': (key1, key2),
                'common_count': len(common),
                'common_values': list(common)
            })

    return result

unionable_tables= utl.loadDictionaryFromPickleFile("/u6/bkassaie/NAUS/data/santos/santos_union_groundtruth.pickle_diluted.pickle") 
output = find_common_values(unionable_tables)
for entry in output:
    print(f"Keys: {entry['keys']}, Common Count: {entry['common_count']}, Common Values: {entry['common_values']}")