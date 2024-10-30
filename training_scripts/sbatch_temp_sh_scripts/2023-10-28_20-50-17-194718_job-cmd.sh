#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=30G
#SBATCH --partition=a40
#SBATCH --qos=m2
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu027,gpu045,gpu055,gpu001,gpu010,gpu034,gpu056,gpu003,gpu004,gpu030,gpu031,gpu046,gpu036,gpu043,gpu054,gpu035,gpu042,gpu048,gpu041,gpu013,gpu008,gpu047,gpu005,gpu052,gpu028,gpu050,gpu044,gpu011,gpu012
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

xvfb-run -s "-screen 0 100x100x24" -a python sb_training.py "/h/andrei/memory_bench/training_scripts/sbatch_temp_configs/2023-10-28_20-50-17-194718_config.json"