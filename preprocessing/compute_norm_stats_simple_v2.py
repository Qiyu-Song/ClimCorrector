import os
import argparse
import numpy as np
import h5py

def compute_mean_std(h5_path: str, dataset: str = "data"):
    with h5py.File(h5_path, "r") as f:
        x = f[dataset][:]  # loads the whole dataset into RAM
    x = x.astype(np.float64, copy=False)

    mean = np.nanmean(x, axis=0)
    std  = np.nanstd(x, axis=0, ddof=0)  # population std (typical for ML normalization)

    return mean, std

def save_npy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_h5", required=True, help="35-year sub23 aggregated input h5")
    ap.add_argument("--target_dc_h5", required=True, help="35-year sub23 aggregated target_dc h5")
    ap.add_argument("--target_sum_h5", required=True, help="35-year sub23 aggregated target_sum h5")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset", default="data", help="dataset key inside h5 (default: data)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    inputs_dir = os.path.join(args.out_dir, "inputs")
    outputs_dir = os.path.join(args.out_dir, "outputs")
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    in_mean, in_std = compute_mean_std(args.input_h5, args.dataset)
    dc_mean, dc_std = compute_mean_std(args.target_dc_h5, args.dataset)
    sm_mean, sm_std = compute_mean_std(args.target_sum_h5, args.dataset)

    save_npy(os.path.join(inputs_dir,  "input_mean_v2_iter2_35year_sub23.npy"), in_mean)
    save_npy(os.path.join(inputs_dir,  "input_std_v2_iter2_35year_sub23.npy"),  in_std)

    save_npy(os.path.join(outputs_dir, "target_dc_mean_v2_iter2_35year_sub23.npy"), dc_mean)
    save_npy(os.path.join(outputs_dir, "target_dc_std_v2_iter2_35year_sub23.npy"),  dc_std)
    save_npy(os.path.join(outputs_dir, "target_sum_mean_v2_iter2_35year_sub23.npy"), sm_mean)
    save_npy(os.path.join(outputs_dir, "target_sum_std_v2_iter2_35year_sub23.npy"),  sm_std)
    
    print("Saved stats to:", args.out_dir)
    print("input mean/std shapes:", in_mean.shape, in_std.shape)
    print("dc    mean/std shapes:", dc_mean.shape, dc_std.shape)
    print("sum   mean/std shapes:", sm_mean.shape, sm_std.shape)

if __name__ == "__main__":
    main()

