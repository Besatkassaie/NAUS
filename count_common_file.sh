#!/bin/bash

# Define the source and target directories.
QUERY_DIR="/path/to/query"
DATALAKE_DIR="/u6/bkassaie/NAUS/data/ugen_v2/query_original"

# Initialize a counter for missing files.
missing_count=0

# Loop through all files in the query directory.
for file in "$QUERY_DIR"/*; do
    # Check if it's a regular file.
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        # If the file is missing in the datalake folder, increment the counter.
        if [ ! -e "$DATALAKE_DIR/$filename" ]; then
            echo "Missing file: $filename"
            ((missing_count++))
        fi
    fi
done

# Output the total count of missing files.
echo "Total missing files: $missing_count"