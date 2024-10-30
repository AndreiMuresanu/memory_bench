#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --mem=100G
#SBATCH --partition=a40
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu051,gpu056
#SBATCH --cpus-per-task=4
#SBATCH --job-name=mem
#SBATCH --output=mem_%j.out
#SBATCH --error=mem_%j.error

# protclip_10580617.pth
export PATH=/pkgs/anaconda3/bin:$PATH
#module load anaconda/3.9

# prepare your environment here
conda init bash
source /h/andrei/.bashrc
conda activate unity_3
echo $PATH
chmod -R 755 /h/andrei/memory_bench
cd /h/andrei/memory_bench/training_scripts
# put your command here
xvfb-run -s "-screen 0 100x100x24" -a python optuna_hparam_search_multi.py "Hallway" "PPO" 0 50 50