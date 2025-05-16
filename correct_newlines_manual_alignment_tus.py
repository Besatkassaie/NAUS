INPUT_FILE = "data/table-union-search-benchmark/small/manual_alignment_tus_benchmark_all.csv"
OUTPUT_FILE = "data/table-union-search-benchmark/small/manual_alignment_tus_benchmark_all_corrected.csv"

import re

def remove_stray_cr(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8', newline='') as infile:
        text = infile.read()
    # Replace all \r not followed by \n
    text = re.sub(r'\r(?!\n)', '', text)
    with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        outfile.write(text)
    print(f"Corrected file written to: {output_path}")

if __name__ == "__main__":
    remove_stray_cr(INPUT_FILE, OUTPUT_FILE) 