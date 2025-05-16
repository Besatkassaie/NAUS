#!/bin/bash

# Submit this script using:
# sbatch batch-submit.sh

# Set resource requirements
#SBATCH --mem=20GB              # Memory allocation
#SBATCH --cpus-per-task=5     # Number of CPU cores
##SBATCH --gres=gpu:1            # Request 1 GPU


#SBATCH --partition=RJMILLER

# Set output file destinations
#SBATCH -o JOB%j.out            # Standard output file
#SBATCH -e JOB%j-err.out        # Standard error file

## email notifications: Get email when your job starts, stops, fails, completes...
## Set email address
#SBATCH --mail-user=bkassaie@uwaterloo.ca
# Set output file destinations
#SBATCH -o JOB%j.out            # Standard output file
#SBATCH -e JOB%j-err.out        # Standard error file

# Set types of notifications (from the options: BEGIN, END, FAIL, REQUEUE, ALL):
#SBATCH --mail-type=ALL
# Load up your Conda environment
source activate TableUnionNew   # Replace <env> with your actual Conda environment name

# Run the Python script
python /u6/bkassaie/NAUS/Starmie0_search.py
