"""
Generate TSP with time windows (TSP-TW) instances for the benchmark.
Single depot (index 0), single vehicle, no capacity (demands: depot 0, rest 1; capacity 1e9).
Hard time windows; stochastic travel time via existing travel_time_generator.
Saves to data/tsp_tw/<timestamp>/ as npz or PyTorch .pt (--format npz|torch).
"""
import argparse
import os
import random
from datetime import datetime
from typing import Dict, List, Union

import numpy as np
from tqdm import tqdm

from common import generate_base_instance, save_dataset, save_dataset_torch
from constants import MAP_SIZE, NUM_INSTANCES, PAPER_SEED, REALIZATIONS_PER_MAP
from real_twcvrp import generate_time_window
from travel_time_generator import get_distances, sample_travel_time

# Node sizes: 10, 20, 30, 40, 50 (15 instances each for local run)
SIZES = [10, 20, 30, 40, 50]


def _normalize_sizes(sizes: Union[int, List[int], None]) -> List[int]:
    """Return a list of node sizes. int -> [int], list -> list, None -> SIZES."""
    if sizes is None:
        return SIZES
    if isinstance(sizes, int):
        return [sizes]
    return list(sizes)


def get_time_matrix(n_nodes: int, travel_times: Dict) -> np.ndarray:
    """Build (n_nodes, n_nodes) time matrix from travel_times dict keyed by (i,j)."""
    matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for (i, j), t in travel_times.items():
        matrix[i, j] = t
    return matrix


def generate_tsp_tw_instance(
    num_customers: int,
    num_cities: int = None,
    instance: Dict = None,
    use_paper_time_ratio: bool = True,
) -> Dict:
    """Generate one TSP-TW instance: single depot, locations, time windows, no capacity."""
    num_cities = num_cities if num_cities else max(1, num_customers // 50)
    num_depots = 1
    demand_range = (0, 0)  # TSP: no demand; we overwrite to depot=0, customers=1
    if instance is None:
        instance = generate_base_instance(
            num_customers,
            MAP_SIZE,
            num_cities,
            num_depots,
            demand_range,
            is_dynamic=False,
        )
    n_nodes = num_customers + num_depots
    # TSP: depot demand 0, all others 1 (for vrp_base depot/customer split)
    demands = np.zeros(n_nodes, dtype=np.float64)
    demands[0] = 0
    demands[1:] = 1
    instance["demands"] = demands
    instance["vehicle_capacity"] = int(1e9)

    distances = get_distances(instance["map_instance"])
    travel_times = {}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                current_time = random.randint(0, 1440)
                travel_times[(i, j)] = round(
                    sample_travel_time(i, j, distances, current_time), 2
                )
            else:
                travel_times[(i, j)] = 0.0
    instance["travel_times"] = travel_times

    time_windows = [
        generate_time_window(use_paper_ratio=use_paper_time_ratio)
        for i in range(n_nodes)
    ]
    time_windows[0] = (0, 1440)  # Depot
    instance["time_windows"] = np.array(time_windows, dtype=np.float64)
    return instance


def generate_tsp_tw_dataset(
    num_customers: int,
    num_cities: int = None,
    precision=np.float64,
    use_paper_time_ratio: bool = True,
    num_instances: int = None,
) -> Dict:
    """Generate a dataset of TSP-TW instances (same structure as TWCVRP for vrp_base)."""
    if num_instances is None:
        num_instances = NUM_INSTANCES
    num_cities = num_cities if num_cities else max(1, num_customers // 50)
    num_depots = 1
    n_nodes = num_customers + num_depots
    dataset = {
        "num_vehicles": [],
        "locations": [],
        "demands": [],
        "time_matrix": [],
        "time_windows": [],
        "vehicle_capacities": [],
        "travel_times": [],
        "map_size": MAP_SIZE,
        "num_cities": num_cities,
        "num_depots": num_depots,
    }
    instance = None
    for _ in tqdm(range(num_instances), desc=f"TSP-TW n={num_customers}"):
        #make new map or change noise from existing map
        if _ % REALIZATIONS_PER_MAP == 0:
            instance = generate_tsp_tw_instance(
                num_customers,
                num_cities=num_cities,
                instance=None,
                #set 60% as residential, set 40% as commercial -> time_windows_generator.py
                use_paper_time_ratio=use_paper_time_ratio,
            )
        else:
            instance = generate_tsp_tw_instance(
                num_customers,
                num_cities=num_cities,
                instance=instance,
                use_paper_time_ratio=use_paper_time_ratio,
            )
        #add new instance to dataset
        dataset["locations"].append(instance["locations"].astype(precision))
        dataset["demands"].append(instance["demands"].astype(precision))
        dataset["vehicle_capacities"].append([int(1e9)])
        dataset["num_vehicles"].append(1)
        time_matrix = get_time_matrix(n_nodes, instance["travel_times"])
        dataset["time_matrix"].append(time_matrix.astype(precision))
        dataset["time_windows"].append(instance["time_windows"].astype(precision))
        dataset["travel_times"].append(instance["travel_times"])
    return {k: np.array(v, dtype=object) for k, v in dataset.items()}


def main(format: str = None, num_instances: int = None, sizes: Union[int, List[int], None] = None):
    """Generate TSP-TW datasets. format: 'npz' | 'torch' (default from --format or 'npz').
    sizes: single int or list of node sizes (num_customers); default SIZES when None."""
    if format is None:
        parser = argparse.ArgumentParser(description="Generate TSP-TW datasets (npz or PyTorch format)")
        parser.add_argument(
            "--format",
            choices=["npz", "torch"],
            default="npz",
            help="Output format: npz (numpy compressed) or torch (PyTorch tensors). Default: npz",
        )
        parser.add_argument(
            "--num-instances",
            type=int,
            default=None,
            help="Number of instances per node size (default from constants.NUM_INSTANCES)",
        )
        parser.add_argument(
            "--sizes",
            type=int,
            nargs="*",
            default=None,
            help="Node sizes as one or more integers (e.g. 10000 or 10 20 50). Default: 10,20,30,40,50",
        )
        args = parser.parse_args()
        format = args.format
        if num_instances is None:
            num_instances = args.num_instances
        if sizes is None and args.sizes is not None:
            sizes = args.sizes if len(args.sizes) > 0 else None
    size_list = _normalize_sizes(sizes)
    use_torch = format == "torch"
    ext = "pt" if use_torch else "npz"

    np.random.seed(PAPER_SEED)
    random.seed(PAPER_SEED)
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = os.path.join(os.path.dirname(__file__), "data", "tsp_tw", run_timestamp)
    os.makedirs(base, exist_ok=True)
    for num_customers in size_list:
        n_inst = num_instances if num_instances is not None else NUM_INSTANCES
        dataset = generate_tsp_tw_dataset(num_customers, use_paper_time_ratio=True, num_instances=n_inst)
        path = os.path.join(base, f"tsp_tw_{num_customers}.{ext}")
        if use_torch:
            save_dataset_torch(dataset, path)
        else:
            save_dataset(dataset, path)
    n_inst = num_instances if num_instances is not None else NUM_INSTANCES
    print(f"Done. TSP-TW datasets written under {base} (sizes={size_list}, {n_inst} instances per file, format={format}).")
    return run_timestamp


if __name__ == "__main__":
    main()
