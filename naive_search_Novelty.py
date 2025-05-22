
import os
import csv

class NaiveSearcherNovelty(object):


    @staticmethod
    def read_csv_files_to_dict(folder_path):
        ''' Read all CSV files from specified folder and create dictionary mapping file names to lists of columns '''
        delim = ','
        data_dict = {}
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                filepath = os.path.join(folder_path, filename)
                if 'ugen_v2' in filename:
                    delim = ';'
                    
                with open(filepath, mode='r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile, delimiter=delim)
                    rows = list(reader)
                    if not rows:
                        continue  # Skip empty files
                    headers = rows[0]
                    # Initialize a list for each column, starting with the header
                    columns = [[header] for header in headers]
                    for row in rows[1:]:
                        for i, value in enumerate(row):
                            # Ensure we don't run into index errors if rows are uneven
                            if i < len(columns):
                                columns[i].append(value)
                            else:
                                # Handle missing columns by appending empty strings
                                columns.append([''] * (len(columns[0]) - 1) + [value])
                    data_dict[filename] = columns
        print(len(data_dict.keys()))
        return data_dict