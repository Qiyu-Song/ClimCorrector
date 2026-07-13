# SPCAM (2000–2013) preprocessing workflow

This documents how the SPCAM training/validation datasets were generated for the
v2 bias-corrector experiments. It is an **archival record**: the raw 6-hourly
source data has since been purged (see Caveats), so this pipeline is fully
re-runnable only from the aggregated `.h5` stage onward.

## Data source

Sarah Weidman's SP-CESM ("SPCAM") replay runs, CAM `h1` (6-hourly) history files,
stitched from three run segments to cover **2000–2013**:

| Years     | Case                     | Path (`.../CESM215_out/Run/`)                         |
|-----------|--------------------------|-------------------------------------------------------|
| 2000–2004 | `spcam_replay_iter2a`    | `archive/spcam_replay_iter2a/atm/hist/`               |
| 2005–2009 | `spcam_replay_iter2b`    | `archive/spcam_replay_iter2b/atm/hist/`               |
| 2010–2013 | `spcam_replay_fix_iter2` | `archive/spcam_replay_fix_iter2/atm/hist/` (orig. `spcam_replay_fix_iter2/run/`) |

Each raw `.nc` holds 6 timesteps: steps 0 & 3 are the input state, steps 1 & 4
are the output/target increments.

**State-independent (SI) corrector** (used to build the `ic`/`sum` targets):
- directory: `/n/home04/sweidman/holylfs06/IC_CESM2/`
- prefix: `spcam_replay_fix`  (files `spcam_replay_fix.MM-DD-SSSSS.nc`)
- from Sarah's `conv_replay_with_corrector` branch.

Grid: 96 × 144, 26 vertical levels. Variable set: **v2** (197 raw input features;
104 target features = 4 vars × 26 levels).

## Pipeline

### Stage 1 — aggregate `.nc` → `.h5`
`create_h5_data_v2_retrieve_independent.py` (+ `_val.py`), driven by:
- train: `slurm_v2_test_retrieve_independent/create_h5_data_spcam_{1,2,3}.sh` (stride 1)
- val:   `create_h5_data_spcam_2012_sub3.val.sh` (2012, stride 3)
- stats source: `create_h5_data_spcam_5year_sub23.sh` (2005–2009, stride 23)

Per year, produces `train_input.h5` (N×197) and `train_target_{dc,ic,sum}.h5` (N×104),
where `dc` = state-dependent (SD), `ic` = state-independent (SI), `sum = dc + ic`.

### Stage 2 — normalization statistics
`compute_norm_stats_simple_v2.py`, driven by `compute_mean_std_spcam_5year_sub23.sh`.
Computes per-feature mean/std for input, `target_dc`, and `target_sum` from the
**2005–2009 stride-23** aggregate, saved to
`normalization/{inputs,outputs}/*_spcam_v2_iter2_5year_sub23.npy`.

### Stage 3 — normalize + feature-engineer + reshape
`preprocess_climcorr_train_data_spcam_v2.py {year}` (via `preprocess_spcam_{year}.sh`)
and `preprocess_climcorr_val_data_spcam_v2.py` (via `preprocess_spcam_val.sh`):
- z-score inputs & targets with the Stage-2 stats;
- floor cloud-variable (CLDLIQ/CLDICE) std to ≥ 1e-5;
- replace the 4 raw attributes (lat, lon, TOD, TOY) with 7 cyclic features
  → input becomes **200 channels**;
- clip normalized targets to [−100, 100];
- reshape `(N, feat)` → `(time, 96, 144, feat)`.

## Data splits

| Split | Years | Processed folder |
|-------|-------|------------------|
| Train | 2000–2011 + 2013 (13 yr; **2012 held out**) | `spcam_v2_iter2_processed/` |
| Val   | 2012, stride 3 | `spcam_v2_iter2_2012_sub3_processed/` |

The model trains on **`target_dc` (SD) only** (`target_filename='target_dc'`).

**Note on 2012 / normalization:** normalization statistics are computed from the
separate 2005–2009 stride-23 subsample, so they do not depend on the train/val
split. Including 2012 in the normalization step therefore does not contaminate the
statistics or leak validation-specific information into training — this is
considered acceptable.

## Reproduction recipe (from the aggregated `.h5` on scratch)

1. `compute_mean_std_spcam_5year_sub23.sh`  → normalization stats.
2. `preprocess_spcam_{2000..2011,2013}.sh` + `preprocess_spcam_val.sh`  → processed train/val.
3. Train with the SPCAM launch script in `~/ClimCorr_trial/` (`target_filename='target_dc'`).

Scratch locations (as of 2026-07): `/n/home03/qiyusong/scratch/climcorr_preprocessing/`
`spcam_v2_iter2/`, `spcam_v2_iter2_processed/`, `spcam_v2_iter2_2012_sub3{,_processed}/`,
`spcam_v2_iter2_5year_sub23/`, `spcam_v2_iter2_heldout/` (2012 held out).

## Caveats

- **Raw `h1` purged.** Sarah's archive `hist/` dirs now hold only `h0` (monthly)
  files, so Stage 1 cannot be re-run. The aggregated `.h5` on scratch are the frozen
  source; the committed `.npy` stats are the surviving statistical fingerprint.
- **No provenance is stored in the `.h5`** for which corrector produced the `ic`/`sum`.
  `corrector_filename` is now set to `spcam_replay_fix`; older aggregated files may
  predate this and are irrelevant to SD-only training.
- **The 2012 holdout is not enforced by the scripts.** `preprocess_spcam_2012.sh`
  exists but should not be run for the training set (skip 2012).
- `preprocess_climcorr_spcam_train_data_v2.py` (note the different word order) is a
  stale, unused near-duplicate — do not use it.
