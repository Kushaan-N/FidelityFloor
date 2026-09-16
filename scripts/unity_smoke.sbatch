#!/bin/bash
# P0/G0 smoke test on Unity L40S (free fallback, spec §0.1/§11).
# Submit:  sbatch scripts/unity_smoke.sbatch
#SBATCH -J ff-smoke
#SBATCH -p gpu-preempt
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /scratch4/workspace/knaskar_umass_edu-fidelityfloor/logs/smoke_%j.out

set -euo pipefail
WS=/scratch4/workspace/knaskar_umass_edu-fidelityfloor
mkdir -p "$WS/logs" "$WS/outputs" "$WS/xdg-cache" "$WS/xdg-data"

module load python/3.11.7

# Everything large (kit caches, shader cache, outputs) stays on scratch, not $HOME
export XDG_CACHE_HOME=$WS/xdg-cache
export XDG_DATA_HOME=$WS/xdg-data
export FF_OUTPUT_ROOT=$WS/outputs
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=N

cd /work/pi_sniekum_umass_edu/kn/FidelityFloor
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
$WS/venv-isaac/bin/python run_local.py smoke
echo "SMOKE_EXIT_$?"
