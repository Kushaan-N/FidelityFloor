#!/bin/bash
# Generic FidelityFloor stage runner on Unity L40S (free Modal fallback).
#   sbatch scripts/unity_run.sbatch smoke
#   sbatch scripts/unity_run.sbatch determinism
#   sbatch scripts/unity_run.sbatch shard --state-id 0
#   sbatch --array=0-29%10 scripts/unity_run.sbatch shard   # F1: state from array id
#SBATCH -J fidelityfloor
#SBATCH -p gpu-preempt
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /scratch4/workspace/knaskar_umass_edu-fidelityfloor/logs/%x_%A_%a.out

set -euo pipefail
WS=/scratch4/workspace/knaskar_umass_edu-fidelityfloor
mkdir -p "$WS/logs" "$WS/outputs" "$WS/xdg-cache" "$WS/xdg-data"

module load python/3.11.7
export XDG_CACHE_HOME=$WS/xdg-cache
export XDG_DATA_HOME=$WS/xdg-data
export FF_OUTPUT_ROOT=$WS/outputs
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=N

cd /work/pi_sniekum_umass_edu/kn/FidelityFloor
STAGE=${1:?usage: sbatch unity_run.sbatch <stage> [args...]}
shift || true
EXTRA=()
if [[ "$STAGE" == "shard" && -n "${SLURM_ARRAY_TASK_ID:-}" && "$*" != *"--state-id"* ]]; then
    EXTRA=(--state-id "$SLURM_ARRAY_TASK_ID")
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
exec $WS/venv-isaac/bin/python run_local.py "$STAGE" "$@" "${EXTRA[@]}"
