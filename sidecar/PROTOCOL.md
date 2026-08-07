# Sidecar wire protocol (normative)

This file is the **single source of truth**. The Fortran half lives in a different
repository — `Qiyu-Song/CAM`, branch `conv_corrector_nn`, `src/physics/cam/corrector.F90`,
subroutine `nn_sidecar_exchange`. Change one side without the other and you get
silent corruption, not an error.

## Exchange

One request in flight, sequence number `NNNNNNNN` (8 digits, zero-padded, starts at 1
and **restarts at 1 every CESM run** — the server clears the directory at startup so a
stale response is never mistaken for a fresh one).

| step | actor | action |
|---|---|---|
| 1 | CAM | write `req_<seq>.bin`, close |
| 2 | CAM | create `req_<seq>.ready`  ← commit point |
| 3 | server | claim: unlink `req_<seq>.ready`, read `.bin`, unlink `req_<seq>.bin` |
| 4 | server | write `resp_<seq>.bin`, close |
| 5 | server | create `resp_<seq>.ready`  ← commit point |
| 6 | CAM | read `resp_<seq>.bin`, unlink both response files |

The `.ready` marker exists so a partially written `.bin` is never read. **Ownership is
split** — the server deletes the request pair, CAM deletes the response pair — so
neither side can unlink a file the other is about to open.

CAM aborts via `endrun` after `nn_sidecar_timeout` (default 900 s).

## Array format

| | request | response |
|---|---|---|
| Fortran shape | `(144, 96, 197, 1)` = (lon, lat, channel, 1) | `(144, 96, 104, 1)` |
| C / torch shape | `(1, 197, 96, 144)` | `(1, 104, 96, 144)` |
| dtype | **big-endian** float32 (`>f4`) | **big-endian** float32 |
| bytes | 10,893,312 | 5,750,784 |

Fortran column-major `(nlon,nlat,nchan,1)` is byte-identical to C-order
`(1,nchan,nlat,nlon)` — reversing the dimensions *is* the whole conversion. Neither
side transposes.

**Endianness is not optional.** CESM is compiled with ifort `-convert big_endian`, so
Fortran unformatted I/O is big-endian. Reading it as little-endian produces the right
byte count with ~0.4% NaN and values spanning ±3.4e38 — indistinguishable from
uninitialised memory at a glance.

## Environment contract

`CORRECTOR_SIDECAR_DIR` — must be on a **coherent-metadata filesystem (Lustre)**. On
NFS v3 the client's directory attribute cache pins every round trip at 30 s
(`acdirmin`) / 60 s on the first call (`acdirmax`), independent of inference cost.

The server must run with `torch.set_flush_denormal(True)` to match ifort's `-ftz`, or
results diverge from the in-process path the first time an intermediate underflows.

## Numerical contract

- 1 thread → **bitwise identical** to in-process CAM.
- ≥2 threads → one fixed alternative trajectory (all thread counts identical to each
  other), roundoff-equivalent to a 1-ulp perturbation.

## Compatibility

| ClimCorrector `sidecar/` | CAM `conv_corrector_nn` |
|---|---|
| initial version | corrector.F90 with `nn_sidecar_exchange` (adds `+147/-3` over `8723c3d9`) |

Model: TorchScript **wrapper** export (normalization baked in), 197 in / 104 out.
Changing `variable_subsets` or re-exporting the wrapper changes `NIN`/`NOUT` here.
