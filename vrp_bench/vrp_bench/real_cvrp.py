import os
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from common import (
    generate_base_instance,
    save_dataset,
    load_dataset,
    visualize_instance,
)
from constants import (
    NUM_INSTANCES,
    DEMAND_RANGE,
    DEMAND_RANGE_PAPER,
    MAP_SIZE,
)

CAPACITIES = {
    10: 20.0,
    15: 25.0,
    20: 30.0,
    30: 33.0,
    40: 37.0,
    50: 40.0,
    60: 43.0,
    75: 45.0,
    100: 50.0,
    125: 55.0,
    150: 60.0,
    200: 70.0,
    500: 100.0,
    1000: 150.0,
}

# Paper Section 3: "vehicle capacity set as total demand ÷ number of vehicles"
def get_num_vehicles_paper(total_demand: int, num_customers: int, single_vehicle: bool) -> int:
    if single_vehicle:
        return 1
    # Multi-vehicle: balance demand and capacity; use a reasonable fleet size
    return max(1, (num_customers + 19) // 20)


def get_vehicle_capacity_paper(total_demand: int, num_vehicles: int) -> int:
    """Paper: capacity = total_demand ÷ num_vehicles (ceil so total capacity >= total_demand)."""
    return max(1, int(np.ceil(total_demand / num_vehicles)))


def generate_cvrp_instance(
        num_customers: int,
        num_cities: Optional[int] = None,
        num_depots: int = 3,
        is_dynamic: bool = False,
        demand_range: Optional[tuple] = None,
) -> Dict:
    """Generate a base CVRP instance with support for multiple depots."""
    num_cities = num_cities if num_cities else max(1, num_customers // 50)
    demand_range = demand_range or DEMAND_RANGE
    instance = generate_base_instance(
        num_customers, MAP_SIZE, num_cities, num_depots, demand_range, is_dynamic
    )
    return instance


def get_num_vehicles(instance, num_customers) -> int:
    """
    Compute the number of vehicles needed for the instance (legacy heuristic).
    """
    total_demand = np.sum(instance["demands"])
    cap = CAPACITIES.get(num_customers, CAPACITIES[1000])
    return max(1, int(np.ceil(total_demand / (2 * cap))))


def generate_cvrp_dataset(
        num_customers: int,
        num_cities: Optional[int] = None,
        num_depots: int = 1,
        precision=np.uint16,
        is_dynamic: bool = False,
        use_paper_demand: bool = False,
        single_depot_single_vehicle: bool = False,
) -> Dict:
    """
    Generate a CVRP dataset. Paper-aligned: use_paper_demand=True, and
    single_depot_single_vehicle=True for single_depot_single_vehicule_sumDemands variant.
    """
    num_cities = num_cities if num_cities else max(1, num_customers // 50)
    demand_range = DEMAND_RANGE_PAPER if use_paper_demand else DEMAND_RANGE
    dataset = {
        "locations": [],
        "demands": [],
        "num_vehicles": [],
        "vehicle_capacities": [],
        "appear_times": [],
        "map_size": MAP_SIZE,
        "num_cities": num_cities,
        "num_depots": num_depots,
    }

    for _ in tqdm(
            range(NUM_INSTANCES), desc=f"Generating {num_customers} customer instances"
    ):
        instance = generate_cvrp_instance(
            num_customers,
            num_cities,
            num_depots,
            is_dynamic=is_dynamic,
            demand_range=demand_range,
        )
        total_demand = int(np.sum(instance["demands"]))
        single_vehicle = single_depot_single_vehicle or (num_depots == 1 and not use_paper_demand)
        if use_paper_demand:
            num_vehicles = get_num_vehicles_paper(
                total_demand, num_customers, single_depot_single_vehicle
            )
            cap = get_vehicle_capacity_paper(total_demand, num_vehicles)
            dataset["vehicle_capacities"].append([cap] * num_vehicles)
            dataset["num_vehicles"].append(num_vehicles)
        else:
            num_vehicles = get_num_vehicles(instance, num_customers)
            dataset["vehicle_capacities"].append([2 * CAPACITIES[num_customers]])
            dataset["num_vehicles"].append(num_vehicles)

        dataset["locations"].append(instance["locations"].astype(precision))
        dataset["demands"].append(instance["demands"].astype(precision))
        dataset["appear_times"].append(instance["appear_time"])

    return {k: np.array(v, dtype=object) for k, v in dataset.items()}


def main():
    customer_counts = [10, 20, 50, 100, 200, 500, 1000]
    os.makedirs("../data/real_cvrp", exist_ok=True)
    for num_customers in tqdm(customer_counts):
        depots = max(1, num_customers // 50)
        dataset = generate_cvrp_dataset(
            num_customers, num_depots=depots, use_paper_demand=False
        )
        save_dataset(
            dataset,
            f"../data/real_cvrp/cvrp_{num_customers}_multi_depot_multi_vehicule_capacities.npz",
        )


if __name__ == "__main__":
    main()
    # dataset = load_dataset("../data/real_cvrp/cvrp_1000.npz")
    # visualize_instance(dataset)
