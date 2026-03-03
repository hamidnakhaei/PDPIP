"""
Generate SVRPBench datasets that match the paper (Section 3).
Produces 10 instances per (size, variant) for:
- CVRP: single_depot_single_vehicule_sumDemands, multi_depot
- TWCVRP: single_depot, depots_equal_city

Sizes: 10, 20, 50, 100, 200, 500, 1000.
Run with fixed seed (42) for reproducibility.
"""
import os
import random
import numpy as np
from tqdm import tqdm

from constants import NUM_INSTANCES, PAPER_SEED
from real_cvrp import generate_cvrp_dataset, save_dataset
from real_twcvrp import generate_twcvrp_dataset

SIZES = [10, 20, 50, 100, 200, 500, 1000]


def main():
    np.random.seed(PAPER_SEED)
    random.seed(PAPER_SEED)

    base = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(os.path.join(base, "real_cvrp"), exist_ok=True)
    os.makedirs(os.path.join(base, "real_twcvrp"), exist_ok=True)

    for num_customers in tqdm(SIZES, desc="Sizes"):
        num_cities = max(1, num_customers // 50)

        # CVRP: single_depot_single_vehicule_sumDemands (1 depot, 1 vehicle, paper demand/capacity)
        dataset_cvrp_single = generate_cvrp_dataset(
            num_customers,
            num_depots=1,
            use_paper_demand=True,
            single_depot_single_vehicle=True,
        )
        path_cvrp_single = os.path.join(
            base, "real_cvrp", f"cvrp_{num_customers}_single_depot_single_vehicule_sumDemands.npz"
        )
        save_dataset(dataset_cvrp_single, path_cvrp_single)

        # CVRP: multi_depot (num_depots = num_cities, paper demand/capacity)
        dataset_cvrp_multi = generate_cvrp_dataset(
            num_customers,
            num_depots=num_cities,
            use_paper_demand=True,
            single_depot_single_vehicle=False,
        )
        path_cvrp_multi = os.path.join(base, "real_cvrp", f"cvrp_{num_customers}_multi_depot.npz")
        save_dataset(dataset_cvrp_multi, path_cvrp_multi)

        # TWCVRP: single_depot (paper demand and 60/40 time windows)
        dataset_tw_single = generate_twcvrp_dataset(
            num_customers, num_depots=1,
            use_paper_demand=True, use_paper_time_ratio=True,
        )
        path_tw_single = os.path.join(
            base, "real_twcvrp", f"twvrp_{num_customers}_single_depot.npz"
        )
        save_dataset(dataset_tw_single, path_tw_single)

        # TWCVRP: depots_equal_city (one depot per city center)
        dataset_tw_city = generate_twcvrp_dataset(
            num_customers, num_depots=num_cities,
            use_paper_demand=True, use_paper_time_ratio=True,
        )
        path_tw_city = os.path.join(
            base, "real_twcvrp", f"twvrp_{num_customers}_depots_equal_city.npz"
        )
        save_dataset(dataset_tw_city, path_tw_city)

    print(f"Done. Datasets written under {base} ({NUM_INSTANCES} instances per file).")


if __name__ == "__main__":
    main()
