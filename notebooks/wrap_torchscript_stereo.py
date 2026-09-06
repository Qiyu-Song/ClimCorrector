"""
Export a trained corrector to TorchScript for online Fortran coupling.
Mirrors notebooks/swinv3-wrapper.ipynb (the NewModel wrapper: raw-input preprocessing ->
model -> de-normalized correction, with the top model level pruned to 0). CPU-only; no data
needed. Produces a single .pt via torch.jit.script.

MODEL_NAME selects which run to export. Default = the StereoRoPE run just trained.
NOTE: on the aggregate (unweighted) val metric, rope_mixed (0.1649) edged rope_stereo (0.1653);
switch MODEL_NAME below if you'd rather deploy that one (identical wrapper).
"""
import sys, os, glob, re
MODELDIR = "/n/home03/qiyusong/ClimCorrector/models/swintransformer_v2_constrained_conserve_rope"
sys.path.insert(0, MODELDIR); os.chdir(MODELDIR)
import numpy as np
import torch
import torch.nn as nn
import modulus
from swintransformer_modulus_polepadding_conserve_rope import SwinTransformerV2CrModulus_polepadding_conserve  # noqa: F401 (registers class)

NORM = "/n/home03/qiyusong/ClimCorrector/preprocessing/normalization"
SAVE = "/n/home03/qiyusong/saved_models"
OUT  = "/n/home03/qiyusong/saved_models_wrapper"
os.makedirs(OUT, exist_ok=True)

# --- SPCAM 5yr-sub23 normalization stats (must match training) ---
# Norm-stat tag MUST match the model's training data. Default = SPCAM; set
# WRAP_NORM_TAG=camnudge_<arm>_9yr_sub23 for the CAM camnudge models.
NTAG = os.environ.get("WRAP_NORM_TAG", "spcam_v2_iter2_5year_sub23")
input_mean     = np.load(f"{NORM}/inputs/input_mean_{NTAG}.npy")
input_std      = np.load(f"{NORM}/inputs/input_std_{NTAG}.npy")
target_mean_dc = np.load(f"{NORM}/outputs/target_dc_mean_{NTAG}.npy")
target_std_dc  = np.load(f"{NORM}/outputs/target_dc_std_{NTAG}.npy")


class NewModel(nn.Module):
    """Raw (197-ch) input -> normalize + cyclic features (200 ch) -> model -> de-normalize;
    top model level of each of S,Q,U,V pruned to 0. Copied from swinv3-wrapper.ipynb."""
    def __init__(self, original_model, input_mean, input_std, target_mean, target_std):
        super().__init__()
        self.original_model = original_model
        self.input_mean = torch.tensor(input_mean, dtype=torch.float32).view(-1, 1, 1)
        input_std[26 * 4:26 * 6] = np.maximum(input_std[26 * 4:26 * 6], 1e-5)  # floor cloud std
        self.input_std = torch.tensor(input_std, dtype=torch.float32).view(-1, 1, 1)
        self.target_mean = torch.tensor(target_mean, dtype=torch.float32).view(-1, 1, 1)
        target_std = np.maximum(target_std, 1e-5)  # floor target std (matches stage-2; handles uvq's zero-T)
        self.target_std = torch.tensor(target_std, dtype=torch.float32).view(-1, 1, 1)

    def preprocessing(self, x):
        lat = x[:, 197 - 4]; lon = x[:, 197 - 3]; tod = x[:, 197 - 2]; toy = x[:, 197 - 1]
        x = (x - self.input_mean) / self.input_std
        lat_norm = lat / 90.0
        lon_cos = torch.cos(lon / 360.0 * 2 * torch.pi); lon_sin = torch.sin(lon / 360.0 * 2 * torch.pi)
        tod_cos = torch.cos(tod / 24.0 * 2 * torch.pi);  tod_sin = torch.sin(tod / 24.0 * 2 * torch.pi)
        toy_cos = torch.cos(toy / 365.0 * 2 * torch.pi); toy_sin = torch.sin(toy / 365.0 * 2 * torch.pi)
        enc = torch.stack((lat_norm, lon_cos, lon_sin, tod_cos, tod_sin, toy_cos, toy_sin), dim=1)
        return torch.cat((x[:, :197 - 4], enc), dim=1)

    def postprocessing(self, x):
        x = x * self.target_std + self.target_mean
        x[:, 0, :, :] = 0.0; x[:, 26, :, :] = 0.0; x[:, 52, :, :] = 0.0; x[:, 78, :, :] = 0.0
        return x

    def forward(self, x):
        return self.postprocessing(self.original_model(self.preprocessing(x)))


def best_ckpt(model_name):
    ck = glob.glob(f"{SAVE}/{model_name}/ckpt/*.mdlus")
    if not ck:
        raise FileNotFoundError(f"no checkpoints under {SAVE}/{model_name}/ckpt/")
    return min(ck, key=lambda p: float(re.search(r"metric_([0-9.]+)\.mdlus", p).group(1)))


MODEL_NAME = os.environ.get("WRAP_MODEL_NAME",
    "swinv2_polepadding_dim1024_depth8_mlp4_rope_stereo_areawt_lr3e-3_cosine_spcam_v2_dc")
# alts: "..._rope_stereo_..." (no area-wt) | "..._rope_..." (rope_mixed, aggregate-best on unweighted)
# Set WRAP_CKPT=<abs path to a .mdlus> to export a specific epoch instead of the best-metric one.

if __name__ == "__main__":
    device = torch.device("cpu")
    ckpt = os.environ.get("WRAP_CKPT") or best_ckpt(MODEL_NAME)
    print("exporting:", ckpt)
    model_inf = modulus.Module.from_checkpoint(ckpt).to(device).eval()
    new_model = NewModel(model_inf, input_mean, input_std, target_mean_dc, target_std_dc).eval()

    scripted = torch.jit.script(new_model).eval()
    out_path = os.path.join(OUT, f"{MODEL_NAME}_{os.path.basename(ckpt).replace('.mdlus','')}.pt")
    scripted.save(out_path)

    # sanity: raw 197-ch input -> finite 104-ch output, scripted == eager
    x = torch.randn(1, 197, 96, 144)
    with torch.no_grad():
        e = new_model(x); s = scripted(x)
    print("output shape:", tuple(s.shape), "| finite:", bool(torch.isfinite(s).all()),
          "| scripted==eager:", bool(torch.allclose(e, s, atol=1e-4)))
    print("saved TorchScript to:", out_path)
