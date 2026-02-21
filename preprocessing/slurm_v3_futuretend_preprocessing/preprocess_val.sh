#!/bin/bash
  
#SBATCH -n 1
#SBATCH -o out-val-%j
#SBATCH -e eo-val-%j
#SBATCH -p huce_ice,huce_cascade
#SBATCH --contiguous
#SBATCH --mail-type=END
#SBATCH --mail-user=qsong@g.harvard.edu
#SBATCH -J create_data_val # job name
#SBATCH -t 60
#SBATCH --mem-per-cpu=256000
#SBATCH --no-requeue

cd /n/home03/qiyusong/ClimCorrector
source activate /n/holylfs06/LABS/kuang_lab/Lab/qiyusong/mamba_envs/climcorr

python preprocessing/preprocess_climcorr_val_data_v3_futuretend.py
