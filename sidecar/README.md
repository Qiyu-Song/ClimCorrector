# NN inference sidecar for CAM

> **Wire protocol: [PROTOCOL.md](PROTOCOL.md)** — normative. The Fortran half lives in a
> separate repo: `Qiyu-Song/CAM`, branch `conv_corrector_nn`,
> `src/physics/cam/corrector.F90` (`nn_sidecar_exchange`). Both must change together.

Runs the state-dependent corrector's neural network in a **persistent external
process on its own cores** instead of inside CAM, exchanging arrays over a
filesystem handshake.

**Measured: 143.2 → 25.4 s per simulated day (5.6×)** on the 2-day f19 test case
(128 MPI ranks, 32 sidecar threads).

## Why it helps

Only `masterproc` holds the gathered global field, so the in-process path ran the
same inference redundantly on every rank and each one was effectively
single-threaded. Threading it in place makes things *worse* — masterproc's threads
compete with the other ~47 ranks sharing its node (measured 177 vs 143 s/sim-day).
A separate process on dedicated cores gets the parallelism for free, and also drops
the per-rank copies of the model weights.

## Pieces

| file | role |
|---|---|
| `infer_server.py` | the persistent server: loads the TorchScript model once, polls for requests |
| `run_server.slurm` | launcher; set `SIDECAR_THREADS` |
| CAM `corrector.F90` | `nn_sidecar_exchange`, gated on `$CORRECTOR_SIDECAR_DIR` (Qiyu-Song/CAM, `conv_corrector_nn`) |

## Use

```bash
sbatch --export=ALL,SIDECAR_THREADS=16 run_server.slurm   # wait for "waiting for requests"
# in the CESM case:
./xmlchange --file env_mach_specific.xml ...   # or add to <environment_variables>:
#   <env name="CORRECTOR_SIDECAR_DIR">/n/holylfs06/.../sidecar_exch</env>
./case.setup --reset && ./xmlchange BUILD_COMPLETE=TRUE
./case.submit
```

Unset `CORRECTOR_SIDECAR_DIR` to fall back to the original in-process path — the
CAM change is inert when the variable is absent.

## Three traps, each of which cost hours

1. **The exchange directory must be on Lustre, never NFS.** `/n/netscratch` is NFS
   v3; its directory attribute cache pinned every round trip at exactly 30 s (60 s
   on the first call) regardless of how fast inference was, hiding the entire
   speedup. `/n/holylfs06` is Lustre and has coherent metadata.
2. **Big-endian.** CESM builds with `-convert big_endian`. The server reads and
   writes `>f4`. Read it little-endian and you get convincing garbage.
3. **Flush-to-zero.** ifort compiles CAM with `-ftz`, so its process flushes
   denormals. The server calls `torch.set_flush_denormal(True)` to match. Without
   it the two paths agree for several calls and then silently diverge the first
   time an intermediate underflows.

## Reproducibility

- `SIDECAR_THREADS=1` is **bitwise identical** to the original in-process run
  (all corrector calls exact; history files 40/40 on both days).
- **Any thread count ≥2 gives one single alternative trajectory** — 2/4/8/16/32 are
  bitwise identical to *each other*, and differ from serial by a fixed roundoff-sized
  amount (parallel GEMM reduction order). This was verified equivalent to injecting a
  deliberate 1-ulp perturbation, which reproduces the same divergence to ~1%.
- Divergence is therefore binary (serial vs parallel), **not** graded by thread count.
  After 2 days: T RMS 0.088 K, OLR RMS 6.4 W/m², but the median gridpoint differs by
  0.5 W/m² — half the precipitation difference comes from 0.25% of gridpoints where a
  convective threshold flipped.

Use 1 thread to regression-test against existing runs; use many threads for new
science and ensembles.

## Diagnostics

`SIDECAR_ARCHIVE=1` keeps a copy of the first request; `SIDECAR_PERTURB=1` injects a
1-ulp perturbation into the first response (for measuring CAM's sensitivity);
`SIDECAR_FTZ=0` disables denormal flushing. `NN_Data_Save=.true.` in `user_nl_cam`
dumps the full per-call input/output to netCDF, which is what pinned down the FTZ bug.
