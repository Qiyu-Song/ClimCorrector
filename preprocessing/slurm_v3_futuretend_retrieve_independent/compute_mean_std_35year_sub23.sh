#!/bin/bash
  
#SBATCH -n 1 # TODO number of cpu
#SBATCH -o out-%j
#SBATCH -e eo-%j
#SBATCH -p huce_ice,huce_cascade
#SBATCH --contiguous
##SBATCH -x holy2a18106
##SBATCH -x holy2b[05101-05108,05201-05208,05301-05308,07101-07108,09202-09208]
#SBATCH --mail-type=END
#SBATCH --mail-user=qsong@g.harvard.edu
#SBATCH -J mean_std # job name
#SBATCH -t 600                         # job time in minutes
#SBATCH --mem-per-cpu=180000 #MB
#SBATCH --no-requeue


cd /n/home03/qiyusong/ClimCorrector
source activate /n/holylfs06/LABS/kuang_lab/Lab/qiyusong/mamba_envs/climcorr

python preprocessing/compute_norm_stats_simple_v3_futuretend.py \
  --input_h5 /n/home03/qiyusong/scratch/climcorr_preprocessing/v3_futuretend_35year_sub23/train_input.h5 \
  --target_dc_h5 /n/home03/qiyusong/scratch/climcorr_preprocessing/v3_futuretend_35year_sub23/train_target_dc.h5 \
  --target_sum_h5 /n/home03/qiyusong/scratch/climcorr_preprocessing/v3_futuretend_35year_sub23/train_target_sum.h5 \
  --out_dir /n/home03/qiyusong/ClimCorrector/preprocessing/normalization
