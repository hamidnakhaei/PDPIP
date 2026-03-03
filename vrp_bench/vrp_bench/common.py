import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from city import Map, Location
from constants import DEPOT, DYNAMIC_PERCENTAGE
import random


def generate_base_instance(
    num_customers: int,
    map_size: Tuple[int, int],
    num_cities: int,
    num_depots: int,
    demand_range: Tuple[int, int],
    is_dynamic: bool = False,
) -> Dict:
    map_instance = Map(map_size, num_cities, num_depots)
    map_instance.sample_locations(num_customers)
    map_instance.cluster_and_place_depots()
    locations = map_instance.locations

    demands = np.random.randint(
        demand_range[0], demand_range[1] + 1, size=num_customers + num_depots
    )
    # All depots have zero demand (first num_depots locations are depots)
    num_depots_actual = len(map_instance.depots)
    demands[:num_depots_actual] = 0
    appear_time = []
    if is_dynamic:
        num_dynamic_customers = int(num_customers * DYNAMIC_PERCENTAGE)
        dynamic_customers = random.sample(range(num_customers), num_dynamic_customers)

        for i in range(num_customers + num_depots):
            if i in dynamic_customers:
                appear_time.append(random.uniform(0, 1440))
            else:
                appear_time.append(0)
    else:
        appear_time = [0] * (num_customers + num_depots)

    return {
        "locations": np.array([(loc.x, loc.y) for loc in locations]),
        "demands": demands,
        "map_instance": map_instance,
        "vehicle_capacity": int(
            # (random.random() * 0.5 + 0.5) * max(demands) * num_customers
            sum(demands)
        ),
        "appear_time": np.array(appear_time),
    }


def save_dataset(dataset: Dict, filename: str):
    np.savez_compressed(filename, **dataset)


def load_dataset(filename: str) -> Dict:
    return dict(np.load(filename, allow_pickle=True))


def _dataset_to_torch(dataset: Dict) -> Dict[str, Any]:
    """Convert dataset with numpy arrays to dict of tensors (and non-tensor fields)."""
    out = {}
    # Stackable array keys -> single tensor each
    stack_keys = ["locations", "demands", "time_matrix", "time_windows", "appear_times"]
    for key in stack_keys:
        if key in dataset:
            arr = dataset[key]
            # Handle np.array(dtype=object) of per-instance arrays
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                elems = [np.asarray(arr[i]) for i in range(len(arr))]
                stacked = np.stack(elems, axis=0)
            else:
                stacked = np.asarray(arr)
            # Ensure numeric dtype (object -> float64 by default)
            if stacked.dtype == object:
                stacked = stacked.astype(np.float64)
            out[key] = torch.from_numpy(stacked)
    # List-like that become 1D tensors
    if "num_vehicles" in dataset:
        nv = np.asarray(dataset["num_vehicles"], dtype=np.int64)
        out["num_vehicles"] = torch.from_numpy(nv)
    if "vehicle_capacities" in dataset:
        # list of [cap] per instance -> (N,) or (N, 1)
        caps = np.array(
            [c[0] if isinstance(c, (list, np.ndarray)) else c for c in dataset["vehicle_capacities"]],
            dtype=np.int64,
        )
        out["vehicle_capacities"] = torch.from_numpy(caps.astype(np.int64))
    # Keep as-is (pickled by torch.save): travel_times (list of dicts), scalars
    if "travel_times" in dataset:
        tt = dataset["travel_times"]
        if isinstance(tt, np.ndarray):
            tt = tt.tolist()
        out["travel_times"] = tt
    for key in ("map_size", "num_cities", "num_depots"):
        if key in dataset:
            val = dataset[key]
            if isinstance(val, np.ndarray):
                # 0-dim array of scalar or small array
                if val.shape == ():
                    val = val.item()
                else:
                    val = val.tolist()
            out[key] = val
    return out


def save_dataset_torch(dataset: Dict, filename: str):
    """Save dataset in PyTorch format (tensors + picklable fields). Retains all information."""
    torch_dict = _dataset_to_torch(dataset)
    torch.save(torch_dict, filename)


def load_dataset_torch(filename: str, map_location=None) -> Dict:
    """Load dataset saved with save_dataset_torch. Returns dict with tensors and original fields."""
    try:
        return torch.load(filename, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(filename, map_location=map_location)


def visualize_instance(dataset: Dict, index: int = 0):
    from city import map_drawer

    locations = dataset["locations"][index]
    num_depots = dataset["num_depots"]
    map_instance = Map(
        dataset["map_size"], dataset["num_cities"], dataset["num_depots"]
    )
    map_instance.locations = [Location(loc[0], loc[1]) for loc in locations]
    map_instance.depots = [
        Location(loc[0], loc[1], DEPOT) for loc in locations[:num_depots]
    ]
    img = map_drawer(map_instance)
    img.show()
