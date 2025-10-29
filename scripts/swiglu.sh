#!/bin/bash

#SBATCH --job-name=minGPT_swiglu   # A descriptive name for your job
#SBATCH --output=output/slurm_logs/slurm-%j.out   # File to capture standard output
#SBATCH --error=output/slurm_logs/slurm-%j.err    # Redirect standard error to a separate file
#SBATCH --qos=dw87              # The quality of service queue
#SBATCH --time=03:00:00         # 3 hours of wall-clock time
#SBATCH --gpus=1                # Request 1 GPU
#SBATCH --mem=128G               # Request 128GB of memory
#SBATCH --cpus-per-task=32       # Request 32 CPU cores

module load python/3.11
cd /home/rsinema/minGPT
source .venv/bin/activate
# python train.py --model gpt2 --batch_size 16 --exp_name base
# python train.py --model gpt2 --batch_size 16 --swiglu --exp_name swiglu
# python train.py --model gpt2 --batch_size 16 --rope --exp_name rope
# python train.py --model gpt2 --batch_size 16 --lw_scheduler --exp_name lw_scheduler
# python train.py --model gpt2 --batch_size 16 --cos_scheduler --exp_name cos_scheduler
# python train.py --model gpt2 --batch_size 16 --rms_norm --exp_name rms_norm
python train.py --model gpt2 --batch_size 16 -a --exp_name all_features

# run module load python/3.11
# make sure you cd into your compute folder
# activate your environment: source ./myenv/bin/activate
# run your python training/inference script

# cancel the job with: scancel <job_id> or scancel -u <username>