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
#SBATCH -J create_data # job name
#SBATCH -t 600                         # job time in minutes
#SBATCH --mem-per-cpu=180000 #MB
#SBATCH --no-requeue


cd /n/home03/qiyusong/ClimCorrector
source activate /n/holylfs06/LABS/kuang_lab/Lab/qiyusong/mamba_envs/climcorr

python preprocessing/create_h5_data_v3_futuretend_retrieve_independent.py \
    'fixreplay_2iter1.cam.h1.1985-*.nc' \
    --data_path '/n/home04/sweidman/holylfs06/CESM215_out/Run/archive/fixreplay_2iter1/atm/hist/' \
    --save_path '/n/home03/qiyusong/scratch/climcorr_preprocessing/v3_futuretend/1985/' \
    --start_idx 0 \
    --stride_sample 1 

python preprocessing/create_h5_data_v3_futuretend_retrieve_independent.py \
    'fixreplay_2iter1.cam.h1.1986-*.nc' \
    --data_path '/n/home04/sweidman/holylfs06/CESM215_out/Run/archive/fixreplay_2iter1/atm/hist/' \
    --save_path '/n/home03/qiyusong/scratch/climcorr_preprocessing/v3_futuretend/1986/' \
    --start_idx 0 \
    --stride_sample 1 


python preprocessing/create_h5_data_v3_futuretend_retrieve_independent.py \
    'fixreplay_2iter1.cam.h1.1987-*.nc' \
    --data_path '/n/home04/sweidman/holylfs06/CESM215_out/Run/archive/fixreplay_2iter1/atm/hist/' \
    --save_path '/n/home03/qiyusong/scratch/climcorr_preprocessing/v3_futuretend/1987/' \
    --start_idx 0 \
    --stride_sample 1 


python preprocessing/create_h5_data_v3_futuretend_retrieve_independent.py \
    'fixreplay_2iter1.cam.h1.1988-*.nc' \
    --data_path '/n/home04/sweidman/holylfs06/CESM215_out/Run/archive/fixreplay_2iter1/atm/hist/' \
    --save_path '/n/home03/qiyusong/scratch/climcorr_preprocessing/v3_futuretend/1988/' \
    --start_idx 0 \
    --stride_sample 1 


python preprocessing/create_h5_data_v3_futuretend_retrieve_independent.py \
    'fixreplay_2iter1.cam.h1.1989-*.nc' \
    --data_path '/n/home04/sweidman/holylfs06/CESM215_out/Run/archive/fixreplay_2iter1/atm/hist/' \
    --save_path '/n/home03/qiyusong/scratch/climcorr_preprocessing/v3_futuretend/1989/' \
    --start_idx 0 \
    --stride_sample 1 

