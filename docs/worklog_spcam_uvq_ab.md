# WORKLOG — SPCAM UVQ A/B: all-levels vs nostrat

Started 2026-08-11. Purpose: answer Sarah's question — is replaying the stratosphere
detrimental in the SD step? She is blocked on this before generating the rest of the
training period.

## Source data (Sarah, delivered ~2026-08-08)

`/n/home04/sweidman/holylfs06/CESM215_out/Run/archive/`

| case | years | role |
|---|---|---|
| `spcam_nudge_uvq_2iter1` | 1980–1989 | production, all levels |
| `spcam_nudge_uvq_2iter2` | 1990–1999 | production, all levels |
| `spcam_nudge_uvq_nostrat_2iter1` | 1980–1989 | A/B arm, replay only below ~50 hPa (level >= 4) |

Verified: 730 h1 files/year, 6 samples/file (~4380/yr), 63 vars, identical var sets.
`nostrat` duplicates 2iter1's decade — it is a control, NOT extra training data.
Trailing single files (1990 in 2iter1, 2000 in 2iter2) are 1-timestep boundary artifacts.

**SDIFF == 0 in both** (T forcing dropped, as Sarah described). `nostrat` additionally has
UDIFF/VDIFF/QDIFF == 0 at levels 0-3.

## A/B design

Both arms: years 1980-1988 train, **1989 val**, identical model config.
Norm stats per arm from its own 1980-1984, stride 23.

**Scoring: levels 4-25 only, in physical (denormalised) units.** Do NOT compare raw val
loss between arms — nostrat has 38 of 104 target channels identically zero vs 26 for
all-levels, and the zeroed levels are where all-levels has its largest increments
(UDIFF RMS 6.1 at L0 vs ~1.0 by L3). It would win by default and the result would be
meaningless.

Caveat to state to Sarah: offline can only say which dataset is more learnable at the
levels that matter. Her actual question (detrimental? stability?) is an ONLINE question,
and offline R2 has already proven a weak proxy for online behaviour in this project
(RoPE vs RPB). Offline first, then a short coupled run for each.

## Landmines

1. **NaN targets.** `compute_norm_stats_simple_v2.py` applies no floor, and
   `preprocess_climcorr_*_data_spcam_v2.py` divides by target std unguarded. With
   SDIFF==0 that is 0/0 = NaN over 26 (all-lev) / 38 (nostrat) channels.
   -> Use `preprocess_climcorr_*_data_camnudge.py` as the template: it already has
   `TARGET_STD_FLOOR = 1e-5` applied to target_dc and target_sum std. camnudge is the
   correct analogue because it is the other UVQ-only (no-T) dataset.
2. **Validation year must be excluded manually.** This is what bit the old dataset: stage-1
   and stage-3 scripts happily generate the val year, and its exclusion from the training
   directory was never expressed in code. Here: after preprocessing, confirm
   `.../<arm>_processed/` contains 1980-1988 and NOT 1989.
3. **Write to holylfs06, never netscratch.** netscratch purged the entire old SPCAM
   dataset on 2026-08-10 with no backup. See docs/spcam_v2_iter2_dataset.md.

## Naming (deliberate; old dataset was `spcam_v2_iter2*` — do not collide)

Root: `/n/holylfs06/LABS/kuang_lab/Lab/qiyusong/climcorr_preprocessing/`

| stage | all-levels arm | nostrat arm |
|---|---|---|
| raw h5 (stride 1) | `spcam_uvq_2iter/{year}/` | `spcam_uvq_nostrat_2iter/{year}/` |
| val raw (stride 3) | `spcam_uvq_2iter_1989_sub3/` | `spcam_uvq_nostrat_2iter_1989_sub3/` |
| norm aggregate (stride 23) | `spcam_uvq_2iter_5yr_sub23/` | `spcam_uvq_nostrat_2iter_5yr_sub23/` |
| processed train | `spcam_uvq_2iter_processed/{year}/` | `spcam_uvq_nostrat_2iter_processed/{year}/` |
| processed val | `spcam_uvq_2iter_1989_sub3_processed/` | `spcam_uvq_nostrat_2iter_1989_sub3_processed/` |

Norm stat tags -> `preprocessing/normalization/{inputs,outputs}/*_spcam_uvq_2iter_5yr_sub23.npy`
and `*_spcam_uvq_nostrat_2iter_5yr_sub23.npy`.

expname / wandb: `spcam_uvq_alllev_ab` and `spcam_uvq_nostrat_ab`, wandb project
`ClimCorr_spcam` (NB: the factorized-vertical run was mislabelled `ClimCorr_camnudge` in
its launch script and only landed in the right project because I overrode it at sync time).

## Open questions

- 1980 may contain replay spin-up. Norm stats use 1980-1984; check whether 1980 statistics
  look anomalous vs 1985 before trusting. Not yet checked.
- Sample count per epoch will differ from the old run (9 yr here vs 13 yr before), so val
  numbers are NOT comparable to the 0.1357 of factorized_vertical_d256_depth8_spcam_v2_dc.
  Different period (1980s vs 2000s), different forcing (no T), different everything.

## Log

- 2026-08-11: verified source data, found landmines 1 and 2, fixed naming, wrote this file.
Tue Aug 11 02:57:46 EDT 2026: stage-1 submitted
Tue Aug 11 04:03:26 EDT 2026: stage-3 launched (38228079 38228080 38228089 38228090)

## 2026-08-11 overnight: training speed investigation

Measured (not assumed): factorized-vertical is **~14x slower per epoch** than the swinv2
baseline despite being 8x smaller in parameters.

| | factorized-vertical d256 | swinv2 rope d1024 |
|---|---|---|
| params | 12.65 M | 101.09 M |
| batch/rank | 2 | 4 |
| rate | 3.66 s/it | 1.96 it/s |
| epoch | 2:24:49 | 10:04 |
| per sample | ~1.83 s | ~0.13 s |

Parameter count is the wrong currency: the architecture batches the horizontal SwinV2 over
the level axis, so each block runs on B*L*H*W ~ 808k tokens of dim 256, twice (horizontal +
vertical), x depth 8, and does two full-tensor permute+contiguous copies per block
(~776 MB each at B=2).

Current state: no AMP/autocast, no torch.compile; only `set_float32_matmul_precision("high")`.
Gradient checkpointing ON for both block types.

First bench (A40, wrong hardware): baseline 8.606 s/it bs2, peak 35.5 GiB;
ckpt=False OOMs at 48 GiB; bf16 crashes in SDPA with
"CUDA error: invalid configuration argument" -- flash kernel cannot launch with the
vertical block's B*H*W = 31,104 sequences x 8 heads. Fix: disable flash SDPA
(mem-efficient/math handle it). A40 being 2.35x slower than the A100 run is itself a hint
the model is memory-bandwidth bound (A40 696 GB/s vs A100 2039 GB/s).

Levers to quantify on A100: gradient checkpointing off, bf16 autocast, larger batch,
torch.compile, and the cost of the permute/contiguous shuffles alone.

### Result: 2.9x speedup, numerics unchanged

Profiled (H200, bs2) instead of guessing. Kernel self-time: `aten::copy_` 23.8%,
elementwise 12.2%, layer-norm fwd+bwd ~22%, mul/div/add ~18%, **all matmuls only ~8%**.
The model is memory-bandwidth bound, not compute bound. That explains why an 8x smaller
model is 14x slower than the big Swin (which does fewer, larger, compute-bound matmuls),
and why the A40 was 2.35x slower than the A100 (bandwidth ratio ~2.9x).

Hypotheses killed by measurement:
- permute/contiguous shuffles: 0.010 s per 8 blocks = 0.6% of a 1.79 s iteration. NOT the problem.
- bf16 alone: only 1.18x (it accelerates the 8% that is matmul).
- gradient checkpointing is NOT removable: OOM at bs2 even on a 141 GB H200.
- `torch.compile` on the whole model: OOM -- compiling *through* torch.utils.checkpoint
  defeats checkpointing and retains all activations.

**What works: compile each block INSIDE the checkpoint boundary** (`checkpoint(compiled_blk)`),
i.e. wrap `m.horiz[i]` and `m.vert[i]` individually, plus bf16 autocast.

| config (H200, bs2) | s/it | s/sample | speedup | peak |
|---|---|---|---|---|
| current: fp32, no compile | 1.797 | 0.898 | 1.00x | 35.5 GiB |
| bf16 only | 1.520 | 0.760 | 1.18x | 26.2 GiB |
| bf16 + compile vert only | 1.278 | 0.639 | 1.40x | 26.2 GiB |
| bf16 + compile horiz only | 0.853 | 0.427 | 2.09x | 25.0 GiB |
| **bf16 + compile both** | **0.619** | **0.310** | **2.90x** | **23.3 GiB** |
| bf16 + compile both, max-autotune | 0.617 | 0.309 | 2.91x | 27.7 GiB (no gain, more mem) |

Numerics: 30 identical steps, fp32/no-compile final loss 0.428307 vs bf16/compiled 0.428308
(rel diff 0.000%), trajectory correlation 0.999996. Speed-only change.

Also required: `torch.backends.cuda.enable_flash_sdp(False)` and `enable_cudnn_sdp(False)` --
both kernels fail to launch on the vertical block's B*H*W = 31,104 sequences
("invalid configuration argument" / cuDNN mha_graph execute failure).

Projected: 2:24:49/epoch -> ~50 min/epoch on 4x A100. Still ~5x the swinv2 baseline (10:04)
but viable. Untested: whether the same treatment helps the swinv2 baseline (likely less --
it is more compute-bound).

## 2026-08-11 12:4x: A/B launched
Jobs: alllev 38333482, nostrat 38333486 (WANDB_MODE=online; offline was a leftover). Scripts ~/ClimCorr_trial/ab_{alllev,nostrat}.sh
Identical except dataset paths / expname / norm-stat paths (verified by diff).
factorized-vertical d256 depth8, bs2, epochs=12, cosine_warmup T_0=13 (full decay),
lr 3e-3 soap huber, amp_bf16=True compile_blocks=True (2.80x measured on A100).
batch size chosen as 2 because the model is memory-bound: s/sample is flat from bs2 to bs4,
so an epoch costs the same wall-clock at any batch and smaller batch buys more optimizer
steps for free.
WANDB_MODE=offline -> must `wandb sync --project ClimCorr_spcam <rundir>` afterwards.
Scoring still TODO: levels 4-25, denormalised (arms use different norm stats, so raw val
loss is not comparable between them).

### Memory-for-speed: measured, declined
Checkpointing costs ~31% (one extra forward), but buying it back needs far more memory than
the headroom suggests:
  ckpt both (current) 0.307 s/sample, 25.0 GiB   <- chosen
  ckpt horiz only     0.271 (1.13x), 69.5 GiB    <- fits A100-80 at 87%, no DDP margin
  ckpt vert only      0.270 (1.14x), 86.8 GiB    <- A100 OOM
  ckpt none           0.234 (1.31x), 132.9 GiB   <- H200 only
Declined: benchmark is single-process, real runs are DDP x4 (extra gradient buckets), and the
job constraint is [a100|h100|h200] so a config that fits H200 but not A100 fails at random,
hours in. 1.13x is not worth that. If wanted: pin --constraint=h200 + ckpt horiz only.
Wed Aug 12 18:07:59 EDT 2026: A/B relaunched 38747841 38747852 -- compile_blocks=False (torch.compile + checkpoint
  breaks under DDP: CheckpointError 'different number of tensors saved'; the 2.8x bench was single-GPU)
Fri Aug 14 03:40:51 EDT 2026: A/B ran 8/12 epochs then died on a NCCL ALLREDUCE timeout (600s) -- both jobs,
different nodes, same elapsed time => likely a filesystem stall at checkpoint write, not
hardware. epoch8: alllev 0.0926, nostrat 0.0854 (NOT comparable: nostrat has 38 zero target
channels vs 26). Resumed as 39130754/39130755 via restart_full_ckpt=True.
Mon Aug 17 21:49:47 EDT 2026: holylfs06 OSTs back — all 7 affected files read cleanly, checkpoint.0.8.pt intact.
alllev resumed as 39919675 (epochs 9-12). No files need regenerating; the outage left nothing corrupt.
