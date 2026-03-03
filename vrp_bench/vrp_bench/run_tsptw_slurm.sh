#!/bin/bash
#SBATCH --job-name=tsptw-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=tsptw_eval_%j.out
#SBATCH --error=tsptw_eval_%j.err
#
# Trillium CPU: run TSP-TW generate + evaluate.
# Submit from repo root: sbatch vrp_bench/run_tsptw_slurm.sh [options]
# Example (single size): sbatch vrp_bench/run_tsptw_slurm.sh --sizes 10000
# Example (multiple):   sbatch vrp_bench/run_tsptw_slurm.sh --sizes 10 20 50
# Output/error go to submission directory. Results: vrp_bench/eval_results/<timestamp>/, vrp_bench/data/tsp_tw/<timestamp>/

set -e
# Job cwd on compute node can be spool dir; use submission dir so we find the repo
cd "${SLURM_SUBMIT_DIR:-.}"
if [ -f main.py ]; then
  : # already in vrp_bench
elif [ -f vrp_bench/main.py ]; then
  cd vrp_bench
else
  echo "ERROR: main.py not found. Submit from repo root: sbatch vrp_bench/run_tsptw_slurm.sh" >&2
  exit 1
fi

# Load Python + scientific stack. If you use conda/venv, comment these out and activate it instead.
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack 2>/dev/null || true

# Install requirements (repo root = parent of vrp_bench)
REQ="../requirements.txt"
if [ ! -f "$REQ" ]; then
  REQ="${SLURM_SUBMIT_DIR:-.}/requirements.txt"
fi
if ! python -m pip install --user -r "$REQ" --quiet 2>/dev/null; then
  echo "ERROR: pip install failed (compute nodes have no network). On the login node run once:" >&2
  echo "  module load StdEnv/2023 python/3.11.5" >&2
  echo "  pip install --user -r $(cd .. && pwd)/requirements.txt" >&2
  exit 1
fi

# Pass through args (e.g. --sizes 10000 or --sizes 10 20 50)
python main.py "$@"
