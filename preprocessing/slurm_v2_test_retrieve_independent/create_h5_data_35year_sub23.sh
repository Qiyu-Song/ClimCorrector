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
#SBATCH -J create_data2 # job name
#SBATCH -t 600                         # job time in minutes
#SBATCH --mem-per-cpu=180000 #MB
#SBATCH --no-requeue


cd /n/home03/qiyusong/ClimCorrector
source activate /n/holylfs06/LABS/kuang_lab/Lab/qiyusong/mamba_envs/climcorr

python preprocessing/create_h5_data_v2_retrieve_independent.py \
    'fixreplay_2iter*.cam.h1.*.nc' \
    --data_path '/n/home04/sweidman/holylfs06/CESM215_out/Run/archive/fixreplay_2iter1/atm/hist/' \
    --save_path '/n/home03/qiyusong/scratch/climcorr_preprocessing/v2_iter2_35year_sub23/' \
    --start_idx 0 \
    --stride_sample 23 
