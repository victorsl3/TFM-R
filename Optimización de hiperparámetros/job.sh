#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=hip_detalle
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=160:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victorsalvadorlopezz@gmail.com
##------------------------ End job description ------------------------
srun python hip_detalle.py
