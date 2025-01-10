import test_naive_search_export
import logging

# from scipy.stats import entropy
# p = [0.6, 0.2, 0.2] # probability distribution
# ent = entropy(p, base=2)
# print("Entropy:", ent)

# # Sample list of numbers
# my_numbers = [1, '2', '3', 2, 1, 4, 2, 1]

# # Count occurrences using Counter
# from collections import Counter
# counts = Counter(my_numbers)
# nemvers=counts.get('2')
# list_length=len(my_numbers)
# prob_counts_ = {item: count / list_length for item, count in counts.items()}
# divided_numbers = [count /list_length for count in counts.values()]

# print(counts)




logging.basicConfig(level=logging.DEBUG)





args2 = ['--encoder', 'cl', '--benchmark', 'santos', 
        '--augment_op', 'drop_col','--sample_meth', 'tfidf_entity',
        '--matching', 'exact',   '--table_order', 'column',
        '--run_id', '0' ,   '--K','20', '--threshold', '0.7', '--restrict', '1']

test_naive_search_export.main(args2)