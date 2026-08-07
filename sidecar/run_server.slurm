#!/bin/bash
# Persistent NN inference sidecar for CAM's nnCorrector.
# Launch BEFORE the CESM job, and keep it alive for the whole run.
#   sbatch --export=ALL,SIDECAR_THREADS=16 run_server.slurm
#SBATCH -J nnsidecar
#SBATCH -p huce_cascade,shared,serial_requeue
#SBATCH -N 1
#SBATCH -c 20                 # >= SIDECAR_THREADS + a few
#SBATCH --mem=48G
#SBATCH -t 1-00:00
#SBATCH -o sidecar.%j.log

# MUST be on a coherent-metadata filesystem (Lustre). On NFS the client caches
# directory entries for 30-60 s and every round trip stalls for exactly that long.
export CORRECTOR_SIDECAR_DIR=${CORRECTOR_SIDECAR_DIR:-/n/holylfs06/LABS/kuang_lab/Lab/qiyusong/ClimCorr/sidecar_exch}
export SIDECAR_THREADS=${SIDECAR_THREADS:-16}
export SIDECAR_MAX_IDLE=${SIDECAR_MAX_IDLE:-3600}
export SIDECAR_MODEL=${SIDECAR_MODEL:-/n/home03/qiyusong/saved_models_wrapper/swinv2_polepadding_dim1024_depth8_mlp4_rope_lr3e-3_cosine_camnudge_uvq_v2_dc_ckpt_epoch_8_metric_0.1247.pt}

mkdir -p "$CORRECTOR_SIDECAR_DIR"
echo "node=$(hostname) threads=$SIDECAR_THREADS dir=$CORRECTOR_SIDECAR_DIR"
PY=/n/holylfs06/LABS/kuang_lab/Lab/qiyusong/mamba_envs/torch_cpu_env/bin/python
$PY -u "$(dirname "$0")/infer_server.py"
