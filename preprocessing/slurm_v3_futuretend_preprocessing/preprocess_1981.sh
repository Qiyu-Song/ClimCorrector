#!/bin/bash
  
#SBATCH -n 1
#SBATCH -o out-1981-%j
#SBATCH -e eo-1981-%j
#SBATCH -p huce_ice,huce_cascade
#SBATCH --contiguous
#SBATCH --mail-type=END
#SBATCH --mail-user=qsong@g.harvard.edu
#SBATCH -J create_data_1981 # job name
#SBATCH -t 60
#SBATCH --mem-per-cpu=128000
#SBATCH --no-requeue

cd /n/home03/qiyusong/ClimCorrector
source activate /n/holylfs06/LABS/kuang_lab/Lab/qiyusong/mamba_envs/climcorr

python preprocessing/preprocess_climcorr_train_data_v3_futuretend.py 1981
