# preprocess_climcorr_val_data.py
import numpy as np
import h5py
import gc
import os

def preprocess_climcorr_val_data():
    """
    Preprocesses ClimCorr data for validation.
    """
    parent_path = f'/n/home03/qiyusong/scratch/climcorr_preprocessing/spcam_v2_iter2_2012_sub3/'
    # mean and std should be the same as for training
    input_mean = np.load('/n/home03/qiyusong/ClimCorrector/preprocessing/normalization/inputs/input_mean_spcam_v2_iter2_5year_sub23.npy')
    input_std = np.load('/n/home03/qiyusong/ClimCorrector/preprocessing/normalization/inputs/input_std_spcam_v2_iter2_5year_sub23.npy')
    target_mean_dc = np.load('/n/home03/qiyusong/ClimCorrector/preprocessing/normalization/outputs/target_dc_mean_spcam_v2_iter2_5year_sub23.npy')
    target_std_dc = np.load('/n/home03/qiyusong/ClimCorrector/preprocessing/normalization/outputs/target_dc_std_spcam_v2_iter2_5year_sub23.npy')
    target_mean_sum = np.load('/n/home03/qiyusong/ClimCorrector/preprocessing/normalization/outputs/target_sum_mean_spcam_v2_iter2_5year_sub23.npy')
    target_std_sum = np.load('/n/home03/qiyusong/ClimCorrector/preprocessing/normalization/outputs/target_sum_std_spcam_v2_iter2_5year_sub23.npy')

    # make std for cloud variables to be max(std, 1e-5)
    input_std[26*4:26*6] = np.maximum(input_std[26*4:26*6], 1e-5)

    input_path = f'{parent_path}/val_input.h5'
    target_path_dc = f'{parent_path}/val_target_dc.h5'
    target_path_sum = f'{parent_path}/val_target_sum.h5'
    input_file = h5py.File(input_path, 'r')
    target_file_dc = h5py.File(target_path_dc, 'r')
    target_file_sum = h5py.File(target_path_sum, 'r')

    x = input_file['data']
    y_dc = target_file_dc['data']
    y_sum = target_file_sum['data']
    
    lat= x[:,-4]
    lon = x[:,-3]
    tod = x[:,-2]
    toy = x[:,-1]
    x = (x - input_mean) / input_std
    y_dc = (y_dc - target_mean_dc) / target_std_dc
    y_sum = (y_sum - target_mean_sum) / target_std_sum
    lat_norm = lat/90.
    lon_cos = np.cos(lon/360.*2*np.pi)
    lon_sin = np.sin(lon/360.*2*np.pi)
    tod_cos = np.cos(tod/24.*2*np.pi)
    tod_sin = np.sin(tod/24.*2*np.pi)
    toy_cos = np.cos(toy/365.*2*np.pi)
    toy_sin = np.sin(toy/365.*2*np.pi)

    x = np.concatenate((x[:,:-4], np.array([lat_norm, lon_cos, lon_sin, tod_cos, tod_sin, toy_cos, toy_sin]).T), axis=1)
    y_dc = np.clip(y_dc, -100, 100)
    y_sum = np.clip(y_sum, -100, 100)

    xshape = x.shape
    x = x.reshape(xshape[0]//(96*144),96,144,xshape[1])
    yshape = y_dc.shape
    y_dc = y_dc.reshape(yshape[0]//(96*144),96,144,yshape[1])
    y_sum = y_sum.reshape(yshape[0]//(96*144),96,144,yshape[1])
    

    assert xshape[0] == yshape[0]

    target_path = f'/n/home03/qiyusong/scratch/climcorr_preprocessing/spcam_v2_iter2_2012_sub3_processed/'
    os.makedirs(target_path, exist_ok=True)
    h5_path = target_path + 'val_input.h5'
    with h5py.File(h5_path, 'w') as hdf:
        hdf.create_dataset('data', data=x, dtype=np.float32)

    h5_path = target_path + 'val_target_dc.h5'
    with h5py.File(h5_path, 'w') as hdf:
        hdf.create_dataset('data', data=y_dc, dtype=np.float32)

    h5_path = target_path + 'val_target_sum.h5'
    with h5py.File(h5_path, 'w') as hdf:
        hdf.create_dataset('data', data=y_sum, dtype=np.float32)
        
    # Clear variables
    del x, y_dc, y_sum, input_mean, input_std, target_mean_dc, target_std_dc, target_mean_sum, target_std_sum
    gc.collect()  # Force garbage collection to free memory
    
    return None

if __name__ == "__main__":
    preprocess_climcorr_val_data()
