import pickle

# Path to the pickle file
pickle_file = "/Users/besatkassaie/Work/Research/DataLakes/TableUnionSearch/NAUS/data/processed/santos/q_tbls_processed_set.pkl"
data={}
# Load the pickle file into a list
with open(pickle_file, 'rb') as file:
    data = pickle.load(file)

# Check if the data is a list (optional)
if isinstance(data, list):
    print("Data successfully loaded into a list!")
else:
    print("The loaded data is not a list. Please check the pickle file.")

# Print the loaded list (optional)
print(data)