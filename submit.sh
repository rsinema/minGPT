#!/bin/bash

#SBATCH --job-name=minGPT_train    # A descriptive name for your job
#SBATCH --output=output/slurm_logs/slurm-%j.out   # File to capture both standard output and error
#SBATCH --error=output/slurm_logs/slurm-%j.err    # Redirect standard error to the same file as output
#SBATCH --qos=dw87              # The quality of service queue
#SBATCH --time=00:30:00         # 30 minutes of wall-clock time
#SBATCH --gpus=1                # Request 1 GPU
#SBATCH --mem=32G               # Request 32GB of memory
#SBATCH --cpus-per-task=16       # Request 16 CPU cores

module load python/3.11
cd /home/rsinema/minGPT
source .venv/bin/activate
python train.py --model gpt-nano --batch_size 1 --exp_name test_base

# run module load python/3.11
# make sure you cd into your compute folder
# activate your environment: source ./myenv/bin/activate
# run your python training/inference script