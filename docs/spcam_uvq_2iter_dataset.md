# SPCAM UVQ 2iter — dataset, A/B experiment, and how to tell it apart from the old one

Written 2026-08-17. This is the **current** SPCAM dataset (Sarah, Aug 2026). It is NOT the
dataset the 0.1357 factorized-vertical result came from — that one is gone; see
[spcam_v2_iter2_dataset.md](spcam_v2_iter2_dataset.md).

## !! Read this first: four similarly-named datasets !!

| name | model | years | T replayed? | SI corrector | status |
|---|---|---|---|---|---|
| **`spcam_uvq_2iter*`** | **SPCAM** | **1980–1999** | **NO** (SDIFF≡0) | **`spcam_nudge_uvq_40`** | **current** |
| `spcam_v2_iter2*` | SPCAM | 2000–2013 | yes | `spcam_replay_fix` | **DESTROYED** (netscratch purge + source h1 deleted) |
| `v2_iter2*` (no prefix!) | **CAM** | 1980–2011 | yes | `camreplay_nudgeforce1` | intact, 3.0 TB on holylfs06 |
| `camnudge_{uvq,uvqt}*` | CAM | 1980s | uvq: no / uvqt: yes | camnudge | intact |

The prefix is the ONLY thing distinguishing SPCAM from CAM in the `v2_iter2` family. A
directory called `v2_iter2_2012_sub3_processed` is **CAM**, not SPCAM.

Numbers from different rows are **not comparable**: different model, period, forcing and
normalization. In particular **val 0.1357 (old SPCAM) and 0.0836 (new SPCAM) mean nothing
side by side.**

## Source (Sarah, delivered ~2026-08-08)

`/n/home04/sweidman/holylfs06/CESM215_out/Run/archive/`

| case | years | role |
|---|---|---|
| `spcam_nudge_uvq_2iter1` | 1980–1989 | production, all levels |
| `spcam_nudge_uvq_2iter2` | 1990–1999 | production, all levels |
| `spcam_nudge_uvq_nostrat_2iter1` | 1980–1989 | **A/B control** — same decade, replay only below ~50 hPa |

7301 h1 files each; 730/year × 6 samples = ~4380 samples/year; 63 variables; identical
variable sets. `nostrat` re-simulates 2iter1's decade — it is a control, NOT extra data.

**Why T was dropped:** Sarah found precipitation strongly damped when forcing T ("reducing T
bias by introducing temperature tendencies might be replacing convection / instability"),
consistent with our own finding that CAM SD degrades the MJO. Hence `SDIFF ≡ 0` throughout.

**`nostrat` additionally has UDIFF/VDIFF/QDIFF ≡ 0 at levels 0–3.**

**SI corrector = `spcam_nudge_uvq_40.%m-%d-%s.nc`** (1460 files, climatological annual cycle,
no year). This differs from the old `spcam_replay_fix` — it matters for online runs, see below.

## Preprocessed products — all on holylfs06, never netscratch

Root `/n/holylfs06/LABS/kuang_lab/Lab/qiyusong/climcorr_preprocessing/`

| stage | all-levels arm | nostrat arm |
|---|---|---|
| raw h5 (stride 1) | `spcam_uvq_2iter/{year}/` | `spcam_uvq_nostrat_2iter/{year}/` |
| val raw (stride 3) | `spcam_uvq_2iter_1989_sub3/` | `spcam_uvq_nostrat_2iter_1989_sub3/` |
| norm aggregate (stride 23) | `spcam_uvq_2iter_5yr_sub23/` | `spcam_uvq_nostrat_2iter_5yr_sub23/` |
| processed train | `spcam_uvq_2iter_processed/{year}/` | `spcam_uvq_nostrat_2iter_processed/{year}/` |
| processed val | `spcam_uvq_2iter_1989_sub3_processed/` | `spcam_uvq_nostrat_2iter_1989_sub3_processed/` |

Norm stats: `preprocessing/normalization/{inputs,outputs}/*_spcam_uvq{,_nostrat}_2iter_5yr_sub23.npy`
(computed from 1980–1984, stride 23 — training years only, no val leakage).

**Split: train 1980–1988, val 1989.** Both arms identical.

Scripts (new, this dataset only):
- stage 1: `preprocessing/slurm_v2_test_retrieve_independent/build_spcam_uvq*{train,val1989,5yr_sub23}.sh`
- stage 2: `.../norm_spcam_uvq*.sh`
- stage 3: `preprocessing/preprocess_climcorr_{train,val}_data_spcam_uvq.py <arm> [year]`, arm ∈ {alllev, nostrat}

## Traps specific to this dataset

1. **Zero-variance target channels.** SDIFF≡0 gives **26 target channels with exactly zero
   std** (alllev) and **38** (nostrat, +levels 0–3 of U/V/Q). Stage 3 divides by target std,
   so without a floor this is 0/0 = NaN over a quarter to a third of the target.
   `TARGET_STD_FLOOR = 1e-5` is applied — templated from
   `preprocess_climcorr_*_data_camnudge.py`, the other UVQ-only dataset. Do NOT template from
   the old `*_spcam_v2.py` scripts, which have no floor.
2. **Raw val loss is NOT comparable between arms.** nostrat has 38 identically-zero target
   channels vs 26, and the zeroed levels carry alllev's largest increments (UDIFF RMS 6.1 at
   L0 vs ~1.0 by L3). It gets an easier target and wins by default. Compare only on
   **levels 4–25, signal-bearing channels, physical units** (`ab_score.py`).
3. **The stage-1 builder always writes `train_*.h5`** (`data_split='train'` hardcoded), including
   for the val set. The stage-3 val script reads `train_*.h5` accordingly. The old workflow
   renamed these by hand, which was recorded nowhere.
4. **Exclude the val year.** 1989 must not appear in `*_processed/`. Verified absent in both arms.

## The A/B experiment

Question (Sarah's): is replaying the stratosphere detrimental in the SD step? She is holding
generation of the remaining training period pending the answer.

Runs: `~/ClimCorr_trial/ab_{alllev,nostrat}.sh` — identical except dataset/expname/norm paths
(verified by diff). factorized-vertical d256 depth8, **bs2**, 12 epochs, `cosine_warmup`
`T_0=13` (full decay), lr 3e-3, soap, huber, `amp_bf16=True`, **`compile_blocks=False`**.
expname `spcam_uvq_{alllev,nostrat}_ab`, wandb project `ClimCorr_spcam`.

Batch size 2 chosen because the model is memory-bandwidth bound: s/sample is flat from bs2 to
bs4, so an epoch costs the same wall-clock at any batch and smaller batch buys more optimizer
steps for free.

### Result

| | epoch 8 val | epoch 12 val | pooled R² (lev 4–25, signal-bearing) @ep8 |
|---|---|---|---|
| all-levels | 0.0926 | *(blocked)* | 0.5894 |
| nostrat | 0.0854 | **0.0836** | 0.5958 |

Per variable @ep8: Q 0.6852/0.6885, U 0.5622/0.5690, V 0.6138/0.6199 (alllev/nostrat).
Per level, the arms agree to **±0.003 at every level from 7 to 25**. The only large
difference is at level 4 — the nostrat truncation boundary, where target variance ≈ 0 and R²
is unstable (it read −0.64 vs −2.02); that level is masked out.

**Conclusion: no measurable offline difference.** Offline cannot answer Sarah's question, and
her actual concern (model crashes / stability) is not something offline R² can see. The
decision must be made online. Precedent: RoPE vs RPB had near-identical offline R² and
behaved differently online.

### Status

nostrat complete (12/12). alllev stuck at epoch 8 — see storage note. The conclusion rests on
the **matched epoch-8 comparison**, which is fair; finishing alllev only tidies it, and
nostrat moved just 0.0018 from epoch 8 to 12.

## For the online test

**The SI corrector differs from the old runs.** New data uses `spcam_nudge_uvq_40`; the
existing SPCAM SD MJO case (`sp_sd_rope_mjo_0206`) and its 60 restarts at 2012 use
`spcam_replay_fix`. Those restarts are equilibrated to the WRONG SI for these correctors.

Restarts under the correct SI — only three exist, one date each:
`spcam_nudge_uvq_2iter1` 1990-01-01, `spcam_nudge_uvq_2iter2` 2000-01-01,
`spcam_nudge_uvq_nostrat_2iter1` 1990-01-01.

So an ensemble is not currently possible under the right SI. Ask Sarah for a restart set under
`spcam_nudge_uvq_40` (she produced `spcam_gen_restart_sandy` the same way).

Template case to clone: `sp_sd_rope_mjo_0206` (`Force_Model=.true.`, `nnCorrector_Model=.true.`,
one `Force_torch_model` path per arm). TorchScript export: `notebooks/wrap_torchscript_stereo.py`
— **must be pointed at the new norm stats**, since the wrapper bakes normalization in.

## 2026-08-14 storage outage (holylfs06)

Five OSTs offline from ~19:45 Aug 14. Affected files **hang forever** on read — unkillable,
jobs stay RUNNING at zero CPU and burn their walltime. Metadata is fine. Safe check:
`lfs find <dir> --ost holylfs6-OST0021 --ost holylfs6-OST0026 --ost holylfs6-OST002a --ost holylfs6-OST002e --ost holylfs6-OST002f`
Avoid `ls -l`, `du`, `stat`, `cat`, `cp`, `rsync` in affected dirs. Writing new files is safe.

Our affected files (7):
```
spcam_uvq_2iter_processed/1982/train_target_dc.h5          <- blocks alllev training
spcam_uvq_nostrat_2iter_processed/1982/train_target_dc.h5
spcam_uvq_nostrat_2iter_processed/1985/train_input.h5
spcam_uvq_nostrat_2iter_processed/1988/train_input.h5
saved_models/spcam_uvq_alllev_ab/ckpt/ckpt_full/checkpoint.0.7.pt
saved_models/spcam_uvq_nostrat_ab/ckpt/ckpt_epoch_2_metric_0.0966.mdlus
saved_models/spcam_uvq_nostrat_ab/ckpt/ckpt_full/SwinTransformerV2CrModulus.0.11.mdlus
```
Clean and usable: **both val sets (0 affected)**, alllev `checkpoint.0.8.pt`, nostrat's final
epoch-12 model. When the OSTs return, regenerate the four preprocessed files (~23 min/year;
new writes avoid the broken targets) so both arms are retrainable independently of the outage.
