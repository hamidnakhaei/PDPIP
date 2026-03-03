# VRP Benchmarks

TSP-TW (TSP with time windows) benchmark and solver evaluation (NN+2opt, Tabu Search, ACO, OR-Tools).

## Installation

From the repo root:

```bash
pip install -r requirements.txt
```

All commands below that run Python scripts must be run from the `vrp_bench` directory:

```bash
cd vrp_bench
```

## Data format: npz vs torch

The pipeline supports two dataset formats:

- **npz** — NumPy compressed archives (`.npz`). Original format; use with the legacy evaluation (`eval.py` + `vrp_evaluation.py`).
- **torch** — PyTorch tensors (`.pt`). More efficient for loading; use with the unified evaluator (`evaluate_unified.py`).

You can generate and evaluate with either format. Same solvers and metrics; only the storage format and which evaluation script you run differ.

| Key | **npz** (NumPy) | **torch** (PyTorch) |
|-----|------------------|----------------------|
| **locations** | `np.ndarray` shape `(N,)` dtype=object; each element shape `(n_nodes, 2)` | `torch.Tensor` shape `(N, n_nodes, 2)` dtype float64 |
| **demands** | `np.ndarray` shape `(N,)` dtype=object; each element shape `(n_nodes,)` | `torch.Tensor` shape `(N, n_nodes)` dtype float64 |
| **time_matrix** | `np.ndarray` shape `(N,)` dtype=object; each element shape `(n_nodes, n_nodes)` | `torch.Tensor` shape `(N, n_nodes, n_nodes)` dtype float64 |
| **time_windows** | `np.ndarray` shape `(N,)` dtype=object; each element shape `(n_nodes, 2)` | `torch.Tensor` shape `(N, n_nodes, 2)` dtype float64 |
| **appear_times** | `np.ndarray` shape `(N,)` dtype=object; each element shape `(n_nodes,)` *(optional; used for dynamic VRP variants, not for static TSP-TW in this repo)* | `torch.Tensor` shape `(N, n_nodes)` dtype float64 *(optional; TSP-TW datasets generated here omit this key and assume all nodes are available from time 0)* |
| **num_vehicles** | `np.ndarray` shape `(N,)` dtype=object; each element int (e.g. 1) | `torch.Tensor` shape `(N,)` dtype int64 |
| **vehicle_capacities** | `np.ndarray` shape `(N,)` dtype=object; each element list `[capacity]` (e.g. [1e9]) | `torch.Tensor` shape `(N,)` dtype int64 (capacity per instance) |
| **travel_times** | `np.ndarray` shape `(N,)` dtype=object; each element dict `{(i,j): time}` | `list` of length N; each element dict `{(i,j): time}` |
| **map_size** | Tuple or 0-dim array, e.g. `(1000, 1000)` | Python `list` or tuple, e.g. `[1000, 1000]` |
| **num_cities** | Python int or 0-dim array | Python `int` |
| **num_depots** | Python int or 0-dim array | Python `int` |

*N = number of instances; n_nodes = 1 + num_customers (depot + customers).*

## TSP-TW data generation

Generate instances (writes to `data/tsp_tw/<timestamp>/`). Choose the format with `--format`:

**NumPy (npz) — default:**

```bash
cd vrp_bench
python generate_tsp_tw_instances.py
# or explicitly:
python generate_tsp_tw_instances.py --format npz
```

Output: `tsp_tw_10.npz`, `tsp_tw_20.npz`, `tsp_tw_50.npz`.

**PyTorch (torch):**

```bash
cd vrp_bench
python generate_tsp_tw_instances.py --format torch
```

Output: `tsp_tw_10.pt`, `tsp_tw_20.pt`, `tsp_tw_50.pt` in the same timestamped folder.

## Local run

### Option 1: npz (original pipeline)

Generate npz data and run evaluation (all solvers) using the original evaluator:

```bash
cd vrp_bench
python main.py
```

Evaluation only, reusing existing npz data (uses latest `data/tsp_tw/<timestamp>/`):

```bash
cd vrp_bench
python main.py --skip-generate
```

### Option 2: torch (unified pipeline)

Generate torch data and run the unified evaluator:

```bash
cd vrp_bench
python main.py --format torch
```

Evaluation only on existing torch data (latest run):

```bash
cd vrp_bench
python evaluate_unified.py --format torch
```

Specify a run and/or sizes:

```bash
python evaluate_unified.py --format torch --run 2026-02-23_15-38-58 --sizes 10 20 50
python evaluate_unified.py --format npz --run 2026-02-23_12-00-00
```

Output is printed to the terminal. The LaTeX summary table is also saved to `eval_results/<run>_<format>/latex_tables.tex` (e.g. `eval_results/2026-02-23_15-38-58_torch/latex_tables.tex`).

## SLURM

From the repo root.

**Original pipeline (npz, generate + evaluate):**

```bash
sbatch vrp_bench/run_tsptw_slurm.sh
```

Runs `main.py` (generate npz data, then evaluate with `eval.py`).

**Unified pipeline (npz or torch, evaluation only):**

```bash
sbatch vrp_bench/run_evaluate_unified_slurm.sh          # npz, latest run
sbatch vrp_bench/run_evaluate_unified_slurm.sh torch    # torch, latest run
RUN=2026-02-23_15-38-58 sbatch vrp_bench/run_evaluate_unified_slurm.sh torch   # specific run
```

Runs `evaluate_unified.py`. Data must already exist under `data/tsp_tw/<timestamp>/` (generate locally or in a separate job).

Logs are written in the submission directory:

- Original: `tsptw_eval_<jobid>.out`, `tsptw_eval_<jobid>.err`
- Unified: `tsptw_unified_eval_<jobid>.out`, `tsptw_unified_eval_<jobid>.err`

For cluster-specific modules (e.g. Trillium), see comments in the respective script.

## Results

Same solvers and metrics for both methods; only paths differ.

- **Data**: `vrp_bench/data/tsp_tw/<timestamp>/`  
  - npz: `tsp_tw_10.npz`, `tsp_tw_20.npz`, `tsp_tw_50.npz`  
  - torch: `tsp_tw_10.pt`, `tsp_tw_20.pt`, `tsp_tw_50.pt`

- **Evaluation (original, npz)**: `vrp_bench/eval_results/<timestamp>/`  
  `latex_tables.tex`, `nn+2opt_results.json`, `tabu_search_results.json`, `aco_results.json`, `or-tools_results.json`

- **Evaluation (unified, npz or torch)**: `vrp_bench/eval_results/<run>_<format>/`  
  e.g. `eval_results/2026-02-23_15-38-58_torch/` — same JSON and LaTeX files.
