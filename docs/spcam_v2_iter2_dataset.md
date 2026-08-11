# SPCAM v2_iter2 dataset — provenance and regeneration notes

**Status as of 2026-08-11: the dataset no longer exists, and neither does the model
output it was built from.** This file records everything needed to rebuild it, and the
one non-obvious trap that would silently corrupt a rebuild.

Written after the dataset was lost, so that the experiment behind
`factorized_vertical_d256_depth8_spcam_v2_dc` (best val 0.1357, epoch 29) stays
reproducible in principle.

## What was lost, and when

| what | where | state |
|---|---|---|
| preprocessed h5 (train, val, 5-yr aggregate) | `$SCRATCH/climcorr_preprocessing/spcam_v2_iter2*` | **empty** — netscratch purge, 2026-08-10 10:25 |
| source h1 (sub-daily) history | 3 archives below | **deleted** — 0 h1 files in all three |
| normalization statistics | `preprocessing/normalization/{inputs,outputs}/*spcam_v2_iter2_5year_sub23.npy` | **SURVIVES** (tracked in git) |
| trained checkpoints + full optimizer state | `holylfs06/.../saved_models/factorized_vertical_d256_depth8_spcam_v2_dc/` | **SURVIVES** |
| CESM case dirs for the 3 source runs | `/n/home04/sweidman/CESM215/CESM/cime/scripts/cases/` | **SURVIVE** (with `user_nl_cam`) |

Only monthly `h0` output remains in the archives; the corrector needs sub-daily `h1`.
Restart files: `spcam_replay_iter2a` 1, `spcam_replay_iter2b` 1, `spcam_replay_fix_iter2` 20.

## The dataset

- **Train**: years 2000–2011 **+ 2013** (13 years), stride 1
- **Validation**: year **2012 only**, stride 3
- **Normalization**: computed from 2005–2009 only, stride 23
- Input 197 channels (`variable_subsets='v2'`), target 104 (`target_dc`, 4 vars x 26 levels)
- Train/val are temporally disjoint; the norm stats touch only training years, so there is
  no validation leakage. Verified 2026-08-11.

## Stage 1 — netCDF -> raw h5

`preprocessing/create_h5_data_v2_retrieve_independent.py`, driven by
`preprocessing/slurm_v2_test_retrieve_independent/`:

| years | source run | source path | stride | script | output |
|---|---|---|---|---|---|
| 2000–2004 | `spcam_replay_iter2a` | `archive/spcam_replay_iter2a/atm/hist/` | 1 | `create_h5_data_spcam_1.sh` | `spcam_v2_iter2/{year}/` |
| 2005–2009 | `spcam_replay_iter2b` | `archive/spcam_replay_iter2b/atm/hist/` | 1 | `create_h5_data_spcam_2.sh` | `spcam_v2_iter2/{year}/` |
| 2010–2013 | `spcam_replay_fix_iter2` | `Run/spcam_replay_fix_iter2/**run/**` | 1 | `create_h5_data_spcam_3.sh` | `spcam_v2_iter2/{year}/` |
| 2012 (val) | `spcam_replay_fix_iter2` | `Run/spcam_replay_fix_iter2/**run/**` | 3 | `create_h5_data_spcam_2012_sub3.val.sh` | `spcam_v2_iter2_2012_sub3/` |
| 2005–2009 (norm) | `spcam_replay_iter2b` | `archive/spcam_replay_iter2b/atm/hist/` | 23 | `create_h5_data_spcam_5year_sub23.sh` | `spcam_v2_iter2_5year_sub23/` |

All source roots are under `/n/home04/sweidman/holylfs06/CESM215_out/Run/`.

**The training set is stitched from three different simulations.** 2010–2013 (and the
validation year) come from a *run* directory rather than an archive — easy to miss when
checking whether the data still exists.

## Stage 2 — normalization statistics

`preprocessing/compute_norm_stats_simple_v2.py`, driven by
`slurm_v2_test_retrieve_independent/compute_mean_std_spcam_5year_sub23.sh`, reading
`spcam_v2_iter2_5year_sub23/train_{input,target_dc,target_sum}.h5`, tag
`spcam_v2_iter2_5year_sub23`. Produces the 6 `.npy` files that are **still in the repo** —
so a rebuild does not have to reproduce these bit-for-bit, and arguably should reuse the
existing ones so new numbers stay comparable to val 0.1357.

## Stage 3 — raw h5 -> normalized "processed"

`preprocess_climcorr_train_data_spcam_v2.py <year>` -> `spcam_v2_iter2_processed/{year}/`
`preprocess_climcorr_val_data_spcam_v2.py`          -> `spcam_v2_iter2_2012_sub3_processed/`
driven by `slurm_v2_preprocessing/preprocess_spcam_{2000..2013,val}.sh`.

Applies the stage-2 statistics, with a variance floor on the condensate channels:

```python
input_std[26*4:26*6] = np.maximum(input_std[26*4:26*6], 1e-5)   # QLIQ, QICE blocks
```

The processed files are **already normalized**; the training loader does not renormalise.

## !! The trap when regenerating !!

**Stage 1 and stage 3 both generate year 2012, but 2012 is the validation year and must
not be in the training set.**

- `create_h5_data_spcam_3.sh` writes `spcam_v2_iter2/2012/`
- `slurm_v2_preprocessing/preprocess_spcam_2012.sh` exists and would write
  `spcam_v2_iter2_processed/2012/`

The training directory that was actually used contained **2000–2011 and 2013 only** — no
2012. The exclusion was manual and is not expressed anywhere in the scripts. Running all
the slurm scripts blindly during a rebuild would put the validation year into training and
silently invalidate every number. **Delete `spcam_v2_iter2_processed/2012/` after
preprocessing, or skip `preprocess_spcam_2012.sh`.**

## To regenerate

1. The h1 output no longer exists anywhere. Either ask whether it can be restored from
   backup/tape, or re-run the three SPCAM cases from their surviving case dirs and restarts
   (expensive — 14 simulated years of SPCAM).
2. Ensure `user_nl_cam` still requests the same `h1` fields; the corrector's 197-channel
   input depends on it.
3. Run stages 1 and 3 as tabled above, reusing the existing stage-2 `.npy` statistics.
4. **Exclude 2012 from training** (see above).
5. Write the output to **holylfs06, not netscratch** — netscratch is purged with no backup,
   which is how this dataset was lost.

## The experiment that used it

`~/ClimCorr_trial/factorized_vertical_d256_depth8_spcam_v2_dc.sh` — SOAP, lr 3e-3, huber,
`cosine_warmup` with `T_0=41` / `eta_min=1e-5` (one cosine leg, no restarts within 40
epochs), `batch_size=2` on 4 GPUs (effective batch 8, 2372 iterations/epoch).
Best val **0.1357 at epoch 29**; stopped by the 3-day wall clock, still improving, ~70%
through the cosine. Model 12.65 M params vs the 101.09 M swinv2-rope baseline (best 0.1649).

Caveat for any future comparison: the baseline used `batch_size=4` (effective 16, 1186
iterations/epoch), so the two runs differ in gradient steps per epoch as well as in
architecture. A matched-batch control was never run.
