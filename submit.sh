#!/bin/bash

#SBATCH --job-name=minGPT_train    # A descriptive name for your job
#SBATCH --output=output/slurm_logs/slurm-%j.out   # File to capture both standard output and error
#SBATCH --error=output/slurm_logs/slurm-%j.err    # Redirect standard error to the same file as output
#SBATCH --qos=dw87              # The quality of service queue
#SBATCH --time=07:00:00         # 7 hours of wall-clock time
#SBATCH --gpus=1                # Request 1 GPU
#SBATCH --mem=32G               # Request 32GB of memory
#SBATCH --cpus-per-task=16       # Request 16 CPU cores

module load python/3.11
cd /home/rsinema/minGPT
source .venv/bin/activate
python train.py --model gpt2 --batch_size 16 --exp_name base
python train.py --model gpt2 --batch_size 16 --swiglu --exp_name swiglu
python train.py --model gpt2 --batch_size 16 --rope --exp_name rope
python train.py --model gpt2 --batch_size 16 --lw_scheduler --exp_name lw_scheduler
python train.py --model gpt2 --batch_size 16 --cos_scheduler --exp_name cos_scheduler
python train.py --model gpt2 --batch_size 16 --rms_norm --exp_name rms_norm
python train.py --model gpt2 --batch_size 16 -a --exp_name all_features

# run module load python/3.11
# make sure you cd into your compute folder
# activate your environment: source ./myenv/bin/activate
# run your python training/inference script

# cancel the job with: scancel <job_id> or scancel -u <username>