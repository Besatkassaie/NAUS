#!/bin/bash

# Submit this script using:
# sbatch batch-submit.sh

# Set resource requirements
#SBATCH --mem=50GB              # Memory allocation
#SBATCH --cpus-per-task=20       # Number of CPU cores
#SBATCH --gres=gpu:1            # Request 1 GPU

# Set output file destinations
#SBATCH -o JOB%j.out            # Standard output file
#SBATCH -e JOB%j-err.out        # Standard error file

# Load up your Conda environment
source activate TableUnionNew   # Replace <env> with your actual Conda environment name

# Run the Python script
python /u6/bkassaie/NAUS/preprocess_align_parallelize.py
