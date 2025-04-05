#!/bin/bash

# Submit this script using:
# sbatch batch-submit.sh

# Set resource requirements
#SBATCH --mem=60GB              # Memory allocation
#SBATCH --cpus-per-task=2       # Number of CPU cores
##SBATCH --gres=gpu:1            # Request 1 GPU

#SBATCH --nodelist=watgpu408
#SBATCH --partition=RJMILLER

# Set output file destinations
#SBATCH -o JOB%j.out            # Standard output file
#SBATCH -e JOB%j-err.out        # Standard error file

## email notifications: Get email when your job starts, stops, fails, completes...
## Set email address
#SBATCH --mail-user=wk5ng@uwaterloo.ca

# Set types of notifications (from the options: BEGIN, END, FAIL, REQUEUE, ALL):
#SBATCH --mail-type=ALL

# Load up your Conda environment
source activate TableUnionNew   # Replace <env> with your actual Conda environment name

# Run the Python script
python /u6/bkassaie/NAUS/evaluate_novelty.py
