#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --job-name=run
#SBATCH --output=res.txt
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=8:00:00

mkdir /scratch/colmodels
python3 run.py places10.yaml
cp /scratch/colmodels/colnet-the-best.pt .
rm -r /scratch/colmodels
#python3 run.py places10.yaml --model <best_model_path>
