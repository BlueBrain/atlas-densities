#!/bin/bash

#SBATCH --account=proj72 # PUT YOUR PROJ HERE
#SBATCH --job-name=prconvert_t_type_nrrds_to_me_type_nrrds    # Job name
#SBATCH --array=0-9                  # Job array range (0 to 9 for 10 batches)
#SBATCH --output=./logs/batch_%A_%a.out     # Output file
#SBATCH --error=./logs/batch_%A_%a.err      # Error file
#SBATCH --time=24:00:00              # Time limit
#SBATCH --mem=0                    # Memory per job
#SBATCH --constraint=uc2
#SBATCH --partition=prod

# Load Python module
module load unstable
source ../myvenv/bin/activate

# Define paths
PATH_TO_T_TYPES_NRRDS="/gpfs/bbp.cscs.ch/data/project/proj84/csaba/aibs_10x_mouse_wholebrain/results/density_calculations/scaled_nrrd_CCFv3a"

# Dynamically calculate total T-types and batch size
TOTAL_T_TYPES=$(ls $PATH_TO_T_TYPES_NRRDS/*.nrrd | wc -l)  # Total T-types
BATCH_SIZE=8                                              # Maximum batch size
NUM_BATCHES=$(( (TOTAL_T_TYPES + BATCH_SIZE - 1) / BATCH_SIZE ))  # Total number of batches

# Calculate start and end indices for this job
START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
END=$((START + BATCH_SIZE))
if [ $END -gt $TOTAL_T_TYPES ]; then
  END=$TOTAL_T_TYPES
fi

# Check if there are T-types to process
if [ $START -ge $TOTAL_T_TYPES ]; then
  echo "No T-types to process for task ID $SLURM_ARRAY_TASK_ID."
  exit 0
fi

# Run the Python script
python process_t_types.py --start $START --end $END