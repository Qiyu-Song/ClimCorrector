"""Persistent inference sidecar for CAM's nnCorrector.

Protocol (filesystem handshake, one request in flight):
  CAM  writes  req_<seq>.bin   then creates  req_<seq>.ready
  here writes  resp_<seq>.bin  then creates  resp_<seq>.ready
The .ready marker is the commit point, so a partially-written .bin is never read.
File ownership is split so neither side can unlink a file the other is about to
open: the server deletes the request pair, CAM deletes the response pair.

All arrays are big-endian float32: CESM is compiled with ifort -convert big_endian,
so Fortran unformatted I/O is big-endian. Reading it as little-endian yields
plausible-looking garbage (right byte count, ~0.4%% NaN) that mimics uninitialised
memory -- see README.

Env: CORRECTOR_SIDECAR_DIR (required), SIDECAR_THREADS, SIDECAR_MODEL,
     SIDECAR_MAX_IDLE (seconds with no request before exiting).
Touch <dir>/stop to shut down cleanly.
"""
import os, sys, time, glob, numpy as np, torch

D        = os.environ["CORRECTOR_SIDECAR_DIR"]
NTHREADS = int(os.environ.get("SIDECAR_THREADS", "1"))
MODEL    = os.environ.get("SIDECAR_MODEL",
    "/n/home03/qiyusong/saved_models_wrapper/swinv2_polepadding_dim1024_depth8_mlp4_rope_lr3e-3_cosine_camnudge_uvq_v2_dc_ckpt_epoch_8_metric_0.1247.pt")
MAX_IDLE = float(os.environ.get("SIDECAR_MAX_IDLE", "3600"))
NIN, NOUT, NLAT, NLON = 197, 104, 96, 144

torch.set_num_threads(NTHREADS)
# CAM is compiled by ifort with -ftz, so its process runs with flush-to-zero /
# denormals-are-zero set. Match it, or intermediates that underflow are handled
# differently here and the two paths disagree by ~1 ulp on some inputs.
if os.environ.get("SIDECAR_FTZ", "1") == "1":
    torch.set_flush_denormal(True)
    print("[sidecar] flush_denormal enabled (matching CAM's -ftz)", flush=True)
os.makedirs(D, exist_ok=True)
print(f"[sidecar] torch {torch.__version__} threads={torch.get_num_threads()} dir={D}", flush=True)
# Clear any leftovers from a previous run. CAM restarts its sequence counter at 1
# each run, so a stale resp_00000001.* would be mistaken for a fresh answer.
for _f in glob.glob(os.path.join(D, "req_*")) + glob.glob(os.path.join(D, "resp_*")) \
        + glob.glob(os.path.join(D, "stop")) + glob.glob(os.path.join(D, "error")):
    os.unlink(_f)
    print(f"[sidecar] cleared stale {os.path.basename(_f)}", flush=True)
model = torch.jit.load(MODEL, map_location="cpu"); model.eval()
# warm up so the first real request isn't paying lazy-init costs
with torch.no_grad():
    model(torch.zeros(1, NIN, NLAT, NLON))
print("[sidecar] model loaded and warmed; waiting for requests", flush=True)
open(os.path.join(D, "server_ready"), "w").close()

nreq, last = 0, time.time()
while True:
    if os.path.exists(os.path.join(D, "stop")):
        print(f"[sidecar] stop file seen; served {nreq} requests", flush=True); break
    ready = sorted(glob.glob(os.path.join(D, "req_*.ready")))
    if not ready:
        if time.time() - last > MAX_IDLE:
            print(f"[sidecar] idle {MAX_IDLE}s; exiting after {nreq} requests", flush=True); break
        time.sleep(0.02); continue

    rp  = ready[0]
    seq = os.path.basename(rp)[4:-6]
    req = os.path.join(D, f"req_{seq}.bin")
    t0  = time.perf_counter()
    # Claim the request before reading it. Without this the same request is
    # served repeatedly until CAM gets around to deleting it -- double work, and
    # a race where we read the .bin while CAM is unlinking it.
    try:
        a = np.fromfile(req, dtype=">f4").astype(np.float32)   # CESM: -convert big_endian
        # We own the request files: claim (.ready) then consume (.bin). CAM owns
        # and deletes only the response files. Split ownership means neither side
        # can ever unlink a file the other is about to open.
        if os.environ.get("SIDECAR_ARCHIVE") and nreq == 0:
            os.makedirs(os.path.join(D, "..", "archive"), exist_ok=True)
            a.tofile(os.path.join(D, "..", "archive", f"req_{seq}.bin"))
            print(f"[sidecar] archived request {seq}", flush=True)
        os.unlink(rp)
        os.unlink(req)
    except FileNotFoundError:
        print(f"[sidecar] seq {seq}: vanished mid-read, skipping", flush=True)
        continue
    exp = NIN * NLAT * NLON
    if a.size != exp:
        print(f"[sidecar] ERROR seq {seq}: got {a.size} floats, expected {exp}", flush=True)
        open(os.path.join(D, "error"), "w").write(f"bad size {a.size}\n"); break
    # Fortran (144,96,197,1) written in column-major order is byte-identical to
    # C-order (1,197,96,144) -- reversing the dims is the whole conversion.
    x  = torch.from_numpy(a.reshape(1, NIN, NLAT, NLON))
    t1 = time.perf_counter()
    with torch.no_grad():
        y = model(x)
    t2 = time.perf_counter()
    # Controlled experiment: perturb ONE output value by 1 ulp on the first call
    # only, to measure how fast CAM amplifies a roundoff-sized seed. Must happen
    # before the response is written.
    if os.environ.get("SIDECAR_PERTURB") and nreq == 0:
        yp = y.numpy().copy()
        idx = np.unravel_index(np.argmax(np.abs(yp)), yp.shape)   # largest value: 1 ulp is meaningful there
        before = float(yp[idx])
        yp[idx] = np.nextafter(np.float32(before), np.float32(np.inf * np.sign(before or 1.0)))
        print(f"[sidecar] PERTURB seq {seq} at {idx}: {before!r} -> {float(yp[idx])!r} "
              f"(delta {float(yp[idx])-before:.3e}, rel {abs(float(yp[idx])-before)/abs(before):.3e})", flush=True)
        y = torch.from_numpy(yp)
    np.ascontiguousarray(y.numpy(), dtype=np.float32).astype(">f4").tofile(os.path.join(D, f"resp_{seq}.bin"))
    open(os.path.join(D, f"resp_{seq}.ready"), "w").close()
    t3 = time.perf_counter()
    nreq += 1; last = time.time()
    yn = y.numpy()
    print(f"[sidecar] seq {seq}: read {t1-t0:.3f}s  infer {t2-t1:.3f}s  write {t3-t2:.3f}s", flush=True)
    print(f"[sidecar]   IN  min={a.min():+.4e} max={a.max():+.4e} mean={a.mean():+.4e} "
          f"nzero={int((a==0).sum())}/{a.size}", flush=True)
    print(f"[sidecar]   OUT min={yn.min():+.4e} max={yn.max():+.4e} mean={yn.mean():+.4e} "
          f"nzero={int((yn==0).sum())}/{yn.size}", flush=True)
print("[sidecar] exit", flush=True)
