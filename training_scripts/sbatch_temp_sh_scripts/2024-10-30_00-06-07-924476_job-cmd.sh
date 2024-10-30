#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=30G
#SBATCH --qos=m2
#SBATCH --gres=gpu:t4:1
#SBATCH --exclude=gpu038,gpu063,gpu070,gpu061,gpu074,gpu111,gpu113,gpu091,gpu068,gpu062,gpu072,gpu069,gpu075,gpu017,gpu112,gpu114
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

xvfb-run -s "-screen 0 100x100x24" -a python eval_random_agent.py "/h/andrei/memory_bench/training_scripts/sbatch_temp_configs/2024-10-30_00-06-07-924476_config.json"