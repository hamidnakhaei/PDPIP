# TSP_TW API Reference

API reference for TSP with time windows (TSP_TW) code under the STSPTW project. All paths are relative to the STSPTW repository root.

---

## Table of contents

- [Overview and data flow](#overview-and-data-flow)
- [Data structures](#data-structures)
- [Instance and dataset generation](#instance-and-dataset-generation)
- [Supporting modules](#supporting-modules)
- [Solver base and result contract](#solver-base-and-result-contract)
- [Solver implementations](#solver-implementations)
- [Evaluation](#evaluation)
- [Entry points and scripts](#entry-points-and-scripts)

---

## Overview and data flow

TSP_TW in this codebase is the **Travelling Salesman Problem with time windows**: a single depot (index 0), single vehicle, no capacity (effectively demands: depot 0, customers 1; vehicle capacity 1e9). Instances use hard time windows and stochastic travel times (paper model in `travel_time_generator`). Generated data is written under `vrp_bench/data/tsp_tw/<timestamp>/` (e.g. `tsp_tw_10.npz`, `tsp_tw_20.pt`).

```mermaid
flowchart LR
  Gen[Instance generation]
  Data[Dataset npz or pt]
  Solver[Solver]
  Eval[Evaluation and metrics]
  Gen --> Data
  Data --> Solver
  Solver --> Eval
```

---

## Data structures

### Instance dict (single problem)

One TSP-TW instance as produced by [generate_tsp_tw_instance](../vrp_bench/generate_tsp_tw_instances.py). Used internally during generation; solvers consume the **dataset dict** (batched).

| Key | Type | Description |
|-----|------|-------------|
| `locations` | `np.ndarray` shape `(n_nodes, 2)` | (x, y) coordinates; index 0 is depot |
| `demands` | `np.ndarray` shape `(n_nodes,)` | Depot 0, customers 1 (TSP convention) |
| `vehicle_capacity` | `int` | 1e9 for TSP-TW |
| `travel_times` | `dict` | `(i, j) -> float` travel time in minutes |
| `time_windows` | `np.ndarray` shape `(n_nodes, 2)` | (start, end) in minutes from midnight; depot typically (0, 1440) |
| `appear_time` | `np.ndarray` shape `(n_nodes,)` | When each node appears (minutes from midnight). For static TSP-TW in this repo, this is `0` for all nodes and is used only internally during generation; solvers and evaluators ignore this field. |
| `map_instance` | `Map` | [city.Map](../vrp_bench/city.py) used to build the instance |

### Dataset dict (batched)

Batched format passed to solvers and saved as npz or torch. Same keys for both formats; shapes differ as below.

| Key | **npz** (NumPy) | **torch** (PyTorch) |
|-----|------------------|----------------------|
| `locations` | `np.ndarray` shape `(N,)` dtype=object; each element shape `(n_nodes, 2)` | `torch.Tensor` shape `(N, n_nodes, 2)` dtype float64 |
| `demands` | `np.ndarray` shape `(N,)` dtype=object; each element shape `(n_nodes,)` | `torch.Tensor` shape `(N, n_nodes)` dtype float64 |
| `time_matrix` | `np.ndarray` shape `(N,)` dtype=object; each element shape `(n_nodes, n_nodes)` | `torch.Tensor` shape `(N, n_nodes, n_nodes)` dtype float64 |
| `time_windows` | `np.ndarray` shape `(N,)` dtype=object; each element shape `(n_nodes, 2)` | `torch.Tensor` shape `(N, n_nodes, 2)` dtype float64 |
| `appear_times` | `np.ndarray` shape `(N,)` dtype=object; each element shape `(n_nodes,)` *(optional; older static TSP-TW runs may include this key, but evaluation scripts drop it and treat all nodes as present from time 0)* | `torch.Tensor` shape `(N, n_nodes)` dtype float64 *(optional; current TSP-TW generation does not write this key; solvers ignore it if present)* |
| `num_vehicles` | `np.ndarray` shape `(N,)` dtype=object; each element int (e.g. 1) | `torch.Tensor` shape `(N,)` dtype int64 |
| `vehicle_capacities` | `np.ndarray` shape `(N,)` dtype=object; each element list `[capacity]` (e.g. [1e9]) | `torch.Tensor` shape `(N,)` dtype int64 (capacity per instance) |
| `travel_times` | `np.ndarray` shape `(N,)` dtype=object; each element dict `{(i,j): time}` | `list` of length N; each element dict `{(i,j): time}` |
| `map_size` | Tuple or 0-dim array, e.g. `(1000, 1000)` | Python `list` or tuple, e.g. `[1000, 1000]` |
| `num_cities` | Python int or 0-dim array | Python `int` |
| `num_depots` | Python int or 0-dim array | Python `int` |

*N = number of instances; n_nodes = 1 + num_customers (depot + customers).*

---

## Instance and dataset generation

**File:** [vrp_bench/generate_tsp_tw_instances.py](../vrp_bench/generate_tsp_tw_instances.py)

### Constants

- **`SIZES`** — List of customer counts per dataset file. Default: `[10, 20, 30, 40, 50]`.

### Functions

#### `get_time_matrix(n_nodes, travel_times)`

Builds an `(n_nodes, n_nodes)` time matrix from a travel-times dict.

- **Parameters:** `n_nodes: int`, `travel_times: Dict` (keys `(i, j)`, values `float`).
- **Returns:** `np.ndarray` shape `(n_nodes, n_nodes)`, dtype `np.float64`.

#### `generate_tsp_tw_instance(num_customers, num_cities=None, instance=None, use_paper_time_ratio=True)`

Generates one TSP-TW instance: single depot, locations, time windows, no capacity.

- **Parameters:** `num_customers: int`; `num_cities: int | None` (default `max(1, num_customers // 50)`); `instance: Dict | None` (if provided, reuse map/locations and add travel times/time windows); `use_paper_time_ratio: bool` (60% residential / 40% commercial time windows).
- **Returns:** Instance dict (see [Data structures](#data-structures)).

#### `generate_tsp_tw_dataset(num_customers, num_cities=None, precision=np.float64, use_paper_time_ratio=True, num_instances=None)`

Generates a dataset of TSP-TW instances (same structure as TWCVRP for [vrp_base](../vrp_bench/vrp_base.py)).

- **Parameters:** `num_customers: int`; `num_cities: int | None`; `precision` (e.g. `np.float64`); `use_paper_time_ratio: bool`; `num_instances: int | None` (default from [constants.NUM_INSTANCES](../vrp_bench/constants.py)).
- **Returns:** Dataset dict with keys listed in [Data structures](#data-structures) (values as arrays/lists).

#### `main(format=None)`

CLI entry: generates TSP-TW datasets and writes to `vrp_bench/data/tsp_tw/<timestamp>/`. Uses `SIZES` for file names (e.g. `tsp_tw_10.npz`, `tsp_tw_20.pt`).

- **Parameters:** `format: str | None` — `'npz'` or `'torch'`; if `None`, reads from `--format` (default `'npz'`).
- **Returns:** Run timestamp string (e.g. `'2026-02-25_12-00-00'`).

---

## Supporting modules

### common — [vrp_bench/common.py](../vrp_bench/common.py)

Data generation, I/O, and visualization used by TSP-TW.

| Symbol | Signature | Description |
|--------|------------|-------------|
| `generate_base_instance` | `(num_customers, map_size, num_cities, num_depots, demand_range, is_dynamic=False) -> Dict` | Builds base instance with map, locations, demands, appear_time; used by TSP-TW instance generation. |
| `save_dataset` | `(dataset: Dict, filename: str) -> None` | Saves dataset as compressed npz. |
| `load_dataset` | `(filename: str) -> Dict` | Loads npz into a dict. |
| `save_dataset_torch` | `(dataset: Dict, filename: str) -> None` | Saves dataset as PyTorch file (tensors + picklable fields). |
| `load_dataset_torch` | `(filename: str, map_location=None) -> Dict` | Loads dataset saved with `save_dataset_torch`. |
| `_dataset_to_torch` | `(dataset: Dict) -> Dict[str, Any]` | Converts dataset dict with numpy arrays to dict of tensors (internal). |
| `visualize_instance` | `(dataset: Dict, index: int = 0) -> None` | Shows map for one instance (uses [city.map_drawer](../vrp_bench/city.py)). |

### constants — [vrp_bench/constants.py](../vrp_bench/constants.py)

TSP_TW-relevant constants.

| Constant | Value / meaning |
|----------|------------------|
| `MAP_SIZE` | `(1000, 1000)` |
| `NUM_INSTANCES` | `10` (default instances per size when generating) |
| `PAPER_SEED` | `42` (reproducibility) |
| `REALIZATIONS_PER_MAP` | `5` (new map every N instances in dataset generation) |
| `OR_TOOLS_TIME_LIMIT_SECONDS` | `300` |
| `TABU_TIME_LIMIT_SECONDS` | `300` |
| `MAX_STAGNANT_ITERATIONS` | `1000` |
| `NUM_STOCHASTIC_REALIZATIONS` | `5` |
| `CUSTOMER` / `DEPOT` | `"customer"` / `"depot"` (location types) |

### real_twcvrp — [vrp_bench/real_twcvrp.py](../vrp_bench/real_twcvrp.py)

Time-window generation used by TSP-TW (shared with TWCVRP).

| Symbol | Signature | Description |
|--------|------------|-------------|
| `generate_time_window` | `(use_paper_ratio: bool = False) -> Tuple[int, int]` | Returns `(start_time, end_time)` in minutes from midnight. Paper ratio: 60% residential, 40% commercial. |

### time_windows_generator — [vrp_bench/time_windows_generator.py](../vrp_bench/time_windows_generator.py)

| Symbol | Signature | Description |
|--------|------------|-------------|
| `sample_time_window` | `(customer_type) -> Tuple[float, float]` | Samples (start, end) in minutes. `customer_type`: 0 = residential, 1 = commercial. |

### travel_time_generator — [vrp_bench/travel_time_generator.py](../vrp_bench/travel_time_generator.py)

Paper travel-time model (Section 2.1, Eq.1–11): T(a,b,t) = D(a,b)/V + delay (time factor, lognormal, accidents).

| Symbol | Signature | Description |
|--------|------------|-------------|
| `get_distances` | `(map: Map) -> Dict[Tuple[int,int], float]` | Pairwise Euclidean distances from map locations. |
| `sample_travel_time` | `(a, b, distances, current_time, velocity=1) -> float` | Stochastic travel time (minutes) including delay. |
| `deterministic_travel_time` | `(a, b, distances, velocity=1) -> float` | D(a,b)/V only; used for current segment in simulation. |

### city — [vrp_bench/city.py](../vrp_bench/city.py)

Map and location types used to generate instances.

| Symbol | Signature | Description |
|--------|------------|-------------|
| `Location` | `(x, y, type=CUSTOMER)` | Dataclass; `distance(other) -> int`. |
| `City` | `(center, spread)` | `batch_sample(map_size, n) -> List[Location]`. |
| `Map` | `(size, num_cities, num_depots)` | `sample_locations(num_locations)`, `cluster_and_place_depots()`; attributes `cities`, `depots`, `locations`. |
| `map_drawer` | `(map: Map, img_size=(720, 720)) -> Image` | Returns PIL Image of the map. |

---

## Solver base and result contract

**File:** [vrp_bench/vrp_base.py](../vrp_bench/vrp_base.py)

### VRPSolverBase

Abstract base class for all VRP/TSP-TW solvers.

#### Constructor

```python
VRPSolverBase(data: Dict)
```

- **`data`:** Dataset dict with at least `locations`. For TSP_TW, optional keys include `demands`, `num_vehicles`, `vehicle_capacities`, `time_windows`, `time_matrix`, `appear_times`. TSP mode defaults: single vehicle, capacity 1e9, depot index 0.

#### Abstract method

- **`solve_instance(instance_idx: int, num_realizations: int = 3) -> Dict`** — Solve one instance; must be implemented by subclasses.

#### Methods

- **`solve_all_instances(num_realizations: int = 3) -> Tuple[Dict, List[Dict]]`** — Solves all instances; returns `(avg_results, all_results)` where each result dict has the shape below.

#### Result dict (per instance)

| Key | Type | Description |
|-----|------|-------------|
| `total_cost` | float | Total route cost (e.g. time). |
| `waiting_time` | float | Total waiting time. |
| `cvr` | float | Constraint violation rate (0–100%). |
| `feasibility` | float | 1 if feasible, 0 otherwise. |
| `runtime` | float | Solve time (seconds). |
| `robustness` | float | Metric from stochastic realizations. |
| `time_window_violations` | float | (Optional) Number of time-window violations. |

#### Helpers (for implementers)

| Method | Signature | Description |
|--------|------------|-------------|
| `get_depots_and_customers` | `(instance_idx: int) -> Tuple[np.ndarray, np.ndarray]` | Returns depot indices and customer indices. |
| `get_time_windows` | `(instance_idx: int) -> Dict[int, Tuple[float, float]]` | Time windows for each node. |
| `get_appear_times` | `(instance_idx: int) -> Dict[int, float]` | Appear time per node. |
| `_get_demands` | `(instance_idx: int) -> np.ndarray` | Demands for the instance. |
| `_get_num_vehicles` | `(instance_idx: int) -> int` | Number of vehicles. |
| `_get_vehicle_capacities` | `(instance_idx: int) -> np.ndarray` | Vehicle capacities. |
| `_check_feasibility` | `(routes, instance_idx) -> Tuple[float, float, int]` | Returns (cvr, feasibility, tw_violations). |
| `_create_empty_result` | `() -> Dict` | Default result for failed instances. |

---

## Solver implementations

Each solver accepts a dataset dict and implements the [Solver base](#solver-base-and-result-contract) contract. All are TSP_TW-compatible (single vehicle, time windows).

| Class | File | Description |
|-------|------|-------------|
| **ORToolsSolver** | [vrp_bench/or_tools_solver.py](../vrp_bench/or_tools_solver.py) | OR-Tools with time limit; fallback to NN2opt if unavailable. |
| **NN2optSolver** | [vrp_bench/nn_2opt_solver.py](../vrp_bench/nn_2opt_solver.py) | Nearest-neighbor + 2-opt. |
| **TabuSearchSolver** | [vrp_bench/tabu_search_solver.py](../vrp_bench/tabu_search_solver.py) | Tabu search with feasibility focus. |
| **ACOSolver** | [vrp_bench/aco_solver.py](../vrp_bench/aco_solver.py) | Ant colony optimization. |

**Constructor:** `SolverClass(data: Dict)`  
**Solve:** `solve_instance(instance_idx: int, num_realizations: int = 3) -> Dict` (same result shape as [Result dict](#result-dict-per-instance)).

---

## Evaluation

### VRPEvaluator — [vrp_bench/vrp_evaluation.py](../vrp_bench/vrp_evaluation.py)

Evaluation framework for VRP/TSP-TW solvers.

#### Constructor

```python
VRPEvaluator(
    base_path: str = "../../vrp_benchmark/",
    problem_set: str = "tsptw",
    results_dir: Optional[str] = None,
    tsp_tw_run: Optional[str] = None,
)
```

- For `problem_set="tsptw"`, data paths are `base_path + "tsp_tw/" + run + "/tsp_tw_" + size + ".npz"`. If `tsp_tw_run` is `None`, the latest timestamped run under `base_path/tsp_tw/` is used.

#### Methods

| Method | Signature | Description |
|--------|------------|-------------|
| `evaluate_solver` | `(solver_class, solver_name, sizes=[10,20,50,100], max_instances_per_file=10, num_realizations=1, use_paper_protocol=True) -> Dict` | Runs solver on benchmark; returns results dict with per-size and aggregated metrics. |
| `_get_latest_tsptw_run` | `(base_path: str) -> str` | Latest timestamped subdir under `base_path/tsp_tw/` (YYYY-MM-DD_HH-MM-SS), or `""`. |
| `generate_latex_tables` | `(results: Dict) -> None` | Prints LaTeX tables for the given results. |
| `_save_results` | `(results: Dict, solver_name: str) -> None` | Writes results to JSON under `results_dir`. |

### eval.main — [vrp_bench/eval.py](../vrp_bench/eval.py)

TSP-TW-only evaluation using **npz** data and [VRPEvaluator](#vrpevaluator).

```python
def main(tsptw_run=None):
```

- Uses latest `data/tsp_tw/<timestamp>/` if `tsptw_run` is `None`. Runs NN+2opt, Tabu Search, ACO, and OR-Tools; prints comparison and LaTeX tables; writes results to `eval_results/<timestamp>/`.

### evaluate_unified.main — [vrp_bench/evaluate_unified.py](../vrp_bench/evaluate_unified.py)

Unified TSP-TW evaluation supporting both **npz** and **torch** formats.

```python
def main(run: Optional[str] = None, format: Optional[str] = None):
```

- **`run`:** Timestamp subdir under `data/tsp_tw/` (e.g. `2026-02-23_15-38-58`); default: latest.
- **`format`:** `"npz"` or `"torch"`; determines which files are loaded (`.npz` vs `.pt`).
- Loads from `data/tsp_tw/<run>/`, converts to solver-compatible dict (see helpers below), runs all solvers, aggregates, and writes to `eval_results/<run>_<format>/`.

#### Helpers (evaluate_unified.py)

| Function | Description |
|----------|-------------|
| `_get_latest_tsptw_run(base_path)` | Same semantics as VRPEvaluator. |
| `_torch_dict_to_solver_dict(raw)` | Converts dict from `load_dataset_torch` to solver-compatible dict (numpy/lists). |
| `_convert_npz_to_dict(data)` | Converts np.load archive to dict. |
| `_limit_instances(data_dict, max_instances)` | Restricts to first `max_instances` instances. |

---

## Entry points and scripts

### main.py — [vrp_bench/main.py](../vrp_bench/main.py)

```python
def main():
```

- **CLI:** `python main.py [--skip-generate] [--format npz|torch] [--num-instances N]`
- Step 1: Unless `--skip-generate`, runs [generate_tsp_tw_instances.main](../vrp_bench/generate_tsp_tw_instances.py) and writes to `data/tsp_tw/<timestamp>/`.
- Step 2: If `--format torch`, runs [evaluate_unified.main](../vrp_bench/evaluate_unified.py); otherwise runs [eval.main](../vrp_bench/eval.py). Uses the generated (or latest) run for data.

### SLURM scripts

| Script | Path | Purpose |
|--------|------|---------|
| **run_tsptw_slurm.sh** | [vrp_bench/run_tsptw_slurm.sh](../vrp_bench/run_tsptw_slurm.sh) | Submits TSP-TW evaluation job; output/error in submission dir; results in `vrp_bench/eval_results/<timestamp>/`, data in `vrp_bench/data/tsp_tw/<timestamp>/`. |
| **run_evaluate_unified_slurm.sh** | [vrp_bench/run_evaluate_unified_slurm.sh](../vrp_bench/run_evaluate_unified_slurm.sh) | Submits unified evaluation; expects data to already exist under `data/tsp_tw/<timestamp>/`. |

---

*End of API reference. For usage and data format details, see [README.md](../README.md).*
