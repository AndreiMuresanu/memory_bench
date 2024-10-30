#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=30G
#SBATCH --qos=m2
#SBATCH --gres=gpu:t4:1
#SBATCH --exclude=gpu050,gpu038,gpu072,gpu063,,gpu070,gpu061,gpu069,gpu074,gpu075,gpu111,gpu112,gpu113,gpu114,gpu115
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mem
#SBATCH --output=mem_%j.out
#SBATCH --error=mem_%j.error

# prepare your environment here
source /h/andrei/.bashrc
conda activate unity_3
echo $PATH
chmod -R 755 /h/andrei/memory_bench
cd /h/andrei/memory_bench/training_scripts

# put your command here

xvfb-run -s "-screen 0 100x100x24" -a python sb_training.py "/h/andrei/memory_bench/training_scripts/sbatch_temp_configs/2023-10-30_00-21-40-282552_config.json"