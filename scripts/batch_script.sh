#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=pretrained_pong
#SBATCH --output=/usrhomes/d1377/server_results/%x%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=7-0:00:00
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=gnunha.k@gmx.de

# Activate everything you need
module load cuda/10.1
pyenv activate venv
# Run your python code

python -m run_probe --method infonce-stdim --env-name PongNoFrameskip-v4 --collect-mode pretrained_ppo

#python -c 'print("start")'
# rllib train -f ray_config.yaml
# python -c 'print("end")'
# python test_slurm_stuff.py
