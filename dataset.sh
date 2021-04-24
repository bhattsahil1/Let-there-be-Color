#!/bin/bash
#SBATCH -A research
#SBATCH -n 4
#SBATCH --job-name=dataset
#SBATCH --output=down.txt
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00

cd /scratch/
mkdir bwcolor
cd bwcolor
pwd
wget -nc http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
tar --keep-old-files -xf places365standard_easyformat.tar
python3 datasplit.py /scratch/bwcolor/places365_standard/train/ ~/data/places10/ places10.txt train 4096 --bname val --bsize 128
python3 datasplit.py /scratch/bwcolor/places365_standard/val/ ~/data/places10/ places10.txt test 96
