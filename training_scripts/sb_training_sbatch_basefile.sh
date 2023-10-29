#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=30G
#SBATCH --partition=a40
#SBATCH --qos=m2
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu027,gpu045,gpu055,gpu001,gpu010,gpu034,gpu056,gpu003,gpu004,gpu030,gpu031,gpu046,gpu036,gpu043,gpu054,gpu035,gpu042,gpu048,gpu041,gpu013,gpu008,gpu005,gpu028,gpu044,gpu011,gpu012,gpu007,gpu002,gpu040,gpu039
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
