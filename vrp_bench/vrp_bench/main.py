"""
Generate TSP-TW datasets (fixed seed) then run TSP-TW-only evaluation for all solvers.
Usage (from vrp_bench directory):
  python main.py [--skip-generate]

Datasets are written to data/tsp_tw/<timestamp>/ (tsp_tw_10.npz, ...). Evaluation
writes to eval_results/<timestamp>/ and uses the generated run (or latest) for data.
"""
import argparse
import os
import random
import re
import sys

import numpy as np

from constants import PAPER_SEED


def main():
    parser = argparse.ArgumentParser(description="SVRPBench paper reproduction pipeline")
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip dataset generation; use existing data/",
    )
    parser.add_argument(
        "--format",
        choices=["npz", "torch"],
        default="npz",
        help="Dataset format when generating: npz or torch (default: npz)",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=15,
        help="Number of instances per node size (default: 15)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=None,
        help="Node sizes: one int (e.g. 10000) or list (e.g. 10 20 50). Default: 10,20,30,40,50",
    )
    args = parser.parse_args()
    sizes = args.sizes if args.sizes else [10, 20, 30, 40, 50]

    np.random.seed(PAPER_SEED)
    random.seed(PAPER_SEED)

    base = os.path.join(os.path.dirname(__file__), "data")
    tsp_tw_dir = os.path.join(base, "tsp_tw")
    tsptw_run = None

    if not args.skip_generate:
        print("Step 1: Generating TSP-TW datasets (seed={}, sizes={})...".format(PAPER_SEED, sizes))
        from generate_tsp_tw_instances import main as generate_tsptw_main
        tsptw_run = generate_tsptw_main(format=args.format, num_instances=args.num_instances, sizes=sizes)
        print("Dataset generation done.\n")
    else:
        if not os.path.isdir(tsp_tw_dir):
            print("data/tsp_tw missing. Run without --skip-generate first.")
            sys.exit(1)
        pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
        subdirs = [d for d in os.listdir(tsp_tw_dir) if os.path.isdir(os.path.join(tsp_tw_dir, d)) and pattern.match(d)]
        if not subdirs:
            print("data/tsp_tw has no timestamped run. Run without --skip-generate first.")
            sys.exit(1)
        print("Skipping dataset generation (--skip-generate).\n")

    print("Step 2: Running evaluation (paper protocol: 10 instances, 1 realization)...")
    if args.format == "torch":
        from evaluate_unified import main as evaluate_main
        evaluate_main(run=tsptw_run, format="torch", sizes=sizes)
    else:
        from eval import main as eval_main
        eval_main(tsptw_run=tsptw_run, sizes=sizes)


if __name__ == "__main__":
    main()
