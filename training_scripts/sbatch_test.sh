#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=20G
#SBATCH --partition=a40
#SBATCH --qos=m3
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu008,gpu015,gpu055,gpu003,gpu056,gpu051,gpu052,gpu029,gpu009
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
mkdir ./this_is_cool