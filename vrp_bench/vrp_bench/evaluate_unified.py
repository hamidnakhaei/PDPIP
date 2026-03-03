"""
Unified TSP-TW evaluation: single file supporting both npz and torch dataset formats.
Run: python evaluate_unified.py [--format npz|torch] [--run TIMESTAMP]
Distinct from eval.py (original pipeline); use eval.py + vrp_evaluation.py for the legacy flow.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np

from common import load_dataset_torch
from nn_2opt_solver import NN2optSolver
from tabu_search_solver import TabuSearchSolver
from aco_solver import ACOSolver
from or_tools_solver import ORToolsSolver, OR_TOOLS_AVAILABLE
from constants import PAPER_SEED


# ---------------------------------------------------------------------------
# Helpers: run resolution, data loading, conversion
# ---------------------------------------------------------------------------

def _get_latest_tsptw_run(base_path: str) -> str:
    """Return latest timestamped subdir under base_path/tsp_tw (YYYY-MM-DD_HH-MM-SS), or empty string."""
    tsp_tw_dir = os.path.join(base_path.rstrip(os.sep), "tsp_tw")
    if not os.path.isdir(tsp_tw_dir):
        return ""
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
    subdirs = [d for d in os.listdir(tsp_tw_dir) if os.path.isdir(os.path.join(tsp_tw_dir, d)) and pattern.match(d)]
    return max(subdirs) if subdirs else ""


def _torch_dict_to_solver_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert dict from load_dataset_torch to solver-compatible dict (numpy/lists)."""
    out = {}
    # Tensor keys -> list of numpy arrays per instance
    for key in ("locations", "demands", "time_matrix", "time_windows", "appear_times"):
        if key not in raw:
            continue
        t = raw[key]
        if hasattr(t, "numpy"):  # torch.Tensor
            n = t.shape[0]
            out[key] = [t[i].numpy().copy() for i in range(n)]
        else:
            out[key] = t
    if "num_vehicles" in raw:
        nv = raw["num_vehicles"]
        out["num_vehicles"] = nv.numpy() if hasattr(nv, "numpy") else np.array(nv)
    if "vehicle_capacities" in raw:
        vc = raw["vehicle_capacities"]
        caps = vc.tolist() if hasattr(vc, "tolist") else list(vc)
        out["vehicle_capacities"] = [[int(c)] for c in caps]
    for key in ("travel_times", "map_size", "num_cities", "num_depots"):
        if key in raw:
            out[key] = raw[key]
    return out


def _convert_npz_to_dict(data) -> Dict:
    """Convert numpy archive (np.load) to dictionary."""
    return {key: data[key] for key in data.files}


def _is_empty_dataset(data_dict: Dict) -> bool:
    if "locations" not in data_dict:
        return True
    locs = data_dict["locations"]
    if isinstance(locs, np.ndarray):
        return locs.shape[0] == 0
    if isinstance(locs, list):
        return len(locs) == 0
    return True


def _limit_instances(data_dict: Dict, max_instances: int) -> Dict:
    """Limit to first max_instances; support both list-of-arrays and numpy arrays."""
    if "locations" not in data_dict:
        return data_dict
    locs = data_dict["locations"]
    if isinstance(locs, list):
        num_instances = min(max_instances, len(locs))
    elif isinstance(locs, np.ndarray):
        num_instances = min(max_instances, locs.shape[0])
    else:
        return data_dict

    limited = {}
    for key, value in data_dict.items():
        if isinstance(value, list) and len(value) >= num_instances:
            limited[key] = value[:num_instances]
        elif isinstance(value, np.ndarray) and len(value.shape) > 0 and value.shape[0] >= num_instances:
            limited[key] = value[:num_instances]
        else:
            limited[key] = value
    return limited


# ---------------------------------------------------------------------------
# Aggregation (same structure as vrp_evaluation)
# ---------------------------------------------------------------------------

def _empty_metrics() -> Dict:
    return {
        "total_cost": 0,
        "waiting_time": 0,
        "cvr": 0,
        "feasibility": 0,
        "runtime": 0,
        "robustness": 0,
        "time_window_violations": 0,
    }


def _aggregate_results(all_results: List[Dict]) -> Dict:
    if not all_results:
        return _empty_metrics()
    metrics = {
        "total_cost": float(np.mean([r["metrics"]["total_cost"] for r in all_results])),
        "waiting_time": float(np.mean([r["metrics"].get("waiting_time", 0) for r in all_results])),
        "cvr": float(np.mean([r["metrics"]["cvr"] for r in all_results])),
        "feasibility": float(np.mean([r["metrics"]["feasibility"] for r in all_results])),
        "runtime": float(np.mean([r["metrics"]["runtime"] for r in all_results])),
        "robustness": float(np.mean([r["metrics"]["robustness"] for r in all_results])),
    }
    tw_violations = [r["metrics"].get("time_window_violations", 0) for r in all_results]
    if any(tw > 0 for tw in tw_violations):
        metrics["time_window_violations"] = float(np.mean(tw_violations))
    return metrics


def _aggregate_size_results(results_by_size: Dict[str, List[Dict]]) -> Dict:
    size_metrics = {}
    size_categories = {"small": [10, 20, 50], "medium": [100, 200], "large": [500, 1000]}
    for category, results_list in results_by_size.items():
        if results_list:
            size_metrics[category] = {
                "feasibility": float(np.mean([r["feasibility"] for r in results_list])),
                "cost": float(np.mean([r["total_cost"] for r in results_list])),
                "cvr": float(np.mean([r["cvr"] for r in results_list])),
                "runtime": float(np.mean([r["runtime"] for r in results_list])),
            }
            if any("time_window_violations" in r for r in results_list):
                size_metrics[category]["time_window_violations"] = float(
                    np.mean([r.get("time_window_violations", 0) for r in results_list])
                )
    return size_metrics


# ---------------------------------------------------------------------------
# Single-file evaluator: load by format, run solver, aggregate
# ---------------------------------------------------------------------------

SIZE_CATEGORIES = {"small": [10, 20, 50], "medium": [100, 200], "large": [500, 1000]}


def evaluate_solver(
    solver_class,
    solver_name: str,
    base_path: str,
    run: str,
    data_format: str,
    results_dir: Optional[str],
    sizes: List[int],
    max_instances_per_file: int,
    num_realizations: int,
    use_paper_protocol: bool,
) -> Dict:
    """Evaluate one solver on TSP-TW; data_format is 'npz' or 'torch'."""
    all_results = []
    results_by_size = defaultdict(list)
    ext = "pt" if data_format == "torch" else "npz"
    prefix = os.path.join(base_path, "tsp_tw", run, "tsp_tw_") if run else os.path.join(base_path, "tsp_tw", "tsp_tw_")

    if use_paper_protocol:
        max_instances_per_file = 10
        num_realizations = 1

    instance_count = 0
    for size in sizes:
        data_path = prefix + str(size) + "." + ext
        if not os.path.exists(data_path):
            continue
        try:
            if data_format == "torch":
                raw = load_dataset_torch(data_path)
                data_dict = _torch_dict_to_solver_dict(raw)
            else:
                data = np.load(data_path, allow_pickle=True)
                data_dict = _convert_npz_to_dict(data)

            # TSP-TW: ignore dynamic appear times entirely (all nodes exist from time 0)
            # Older datasets may still contain an 'appear_times' key; drop it before
            # passing data to solvers so they operate purely with time windows.
            data_dict.pop("appear_times", None)
            limited_data = _limit_instances(data_dict, max_instances_per_file)
            if _is_empty_dataset(limited_data):
                continue
            solver = solver_class(limited_data)
            avg_results, _ = solver.solve_all_instances(num_realizations)
            result_entry = {
                "problem": f"tsp_tw_{size}",
                "size": size,
                "type": "",
                "metrics": avg_results,
                "problem_type": "tsp_tw",
            }
            all_results.append(result_entry)
            for category, size_list in SIZE_CATEGORIES.items():
                if size in size_list:
                    results_by_size[category].append(avg_results)
            instance_count += 1
            print(f"Completed {solver_name} - size {size}: Cost={avg_results['total_cost']:.1f}, "
                  f"Runtime={avg_results['runtime']*1000:.1f}ms")
        except Exception as e:
            print(f"Error processing {data_path}: {e}")
            continue

    print(f"Completed evaluation: {instance_count} problem-type combinations processed")
    overall_metrics = _aggregate_results(all_results)
    size_metrics = _aggregate_size_results(results_by_size)
    tsp_tw_metrics = _aggregate_results(all_results) if all_results else _empty_metrics()
    results = {
        "solver": solver_name,
        "overall": overall_metrics,
        "tsp_tw": tsp_tw_metrics,
        "by_size": size_metrics,
        "detailed": all_results,
    }
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, f"{solver_name.lower().replace(' ', '_')}_results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
    print("Evaluation completed. Results saved.")
    return results


# ---------------------------------------------------------------------------
# LaTeX and comparison (from eval.py)
# ---------------------------------------------------------------------------

def generate_docx_tables(results: Dict) -> None:
    print("\n% =========================================================")
    print("% TABLES FROM DOCX ATTACHMENT")
    print("% =========================================================")
    solvers = []
    metrics_data = []
    for solver_name, result in results.items():
        if result and "overall" in result:
            solvers.append(solver_name)
            metrics_data.append(result["overall"])
    print("\n% Table 1: Performance comparison of baseline methods")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Performance comparison of baseline methods (mean over all test")
    print("instances and 1 stochastic realization). Lower is better for Cost, CVR,")
    print("Runtime, and Robustness; higher is better for Feasibility Rate.}")
    print("\\label{tab:main_results}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Total Cost} & \\textbf{CVR (\\%)} &")
    print("\\textbf{Feasibility} & \\textbf{Runtime (s)} & \\textbf{Robustness} \\\\")
    print("\\midrule")
    for solver, metrics in zip(solvers, metrics_data):
        print(f"{solver} & {metrics.get('total_cost', 0):.1f} & {metrics.get('cvr', 0):.1f} & "
              f"{metrics.get('feasibility', 0):.3f} & {metrics.get('runtime', 0):.3f} & {metrics.get('robustness', 0):.1f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("\n% Table 2: Feasibility rate by instance scale")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Feasibility rate by instance scale.}")
    print("\\label{tab:feasibility_by_scale}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Small} & \\textbf{Medium} & \\textbf{Large} \\\\")
    print("\\midrule")
    for solver_name, result in results.items():
        if result and "by_size" in result:
            by_size = result["by_size"]
            small_feas = by_size.get("small", {}).get("feasibility", 0)
            medium_feas = by_size.get("medium", {}).get("feasibility", 0)
            large_feas = by_size.get("large", {}).get("feasibility", 0)
            print(f"{solver_name} & {small_feas:.3f} & {medium_feas:.3f} & {large_feas:.3f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("\n% Table 3: Cost robustness (variance) by method")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Cost robustness (variance) by method.}")
    print("\\label{tab:robustness_variance}")
    print("\\begin{tabular}{lc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Cost Variance} \\\\")
    print("\\midrule")
    for solver, metrics in zip(solvers, metrics_data):
        print(f"{solver} & {metrics.get('robustness', 0):.1f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("\n% Table 4: Average runtime by method")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Average runtime (in seconds) by method.}")
    print("\\label{tab:runtime_summary}")
    print("\\begin{tabular}{lc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Runtime (s)} \\\\")
    print("\\midrule")
    for solver, metrics in zip(solvers, metrics_data):
        print(f"{solver} & {metrics.get('runtime', 0):.3f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def generate_detailed_tables(results: Dict) -> None:
    print("\n% =========================================================")
    print("% DETAILED TABLES BY PROBLEM TYPE AND SIZE")
    print("% =========================================================")
    print("\n% Table 5: Detailed Performance by Instance Size")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Detailed Performance Analysis by Instance Size}")
    print("\\label{tab:detailed_size_performance}")
    print("\\begin{tabular}{lcccccccccccc}")
    print("\\toprule")
    print("& \\multicolumn{4}{c}{\\textbf{Small (≤50)}} & \\multicolumn{4}{c}{\\textbf{Medium (100-200)}} & \\multicolumn{4}{c}{\\textbf{Large (≥500)}} \\\\")
    print("\\cmidrule(r){2-5} \\cmidrule(r){6-9} \\cmidrule(l){10-13}")
    print("\\textbf{Method} & \\textbf{Cost} & \\textbf{CVR} & \\textbf{Feas} & \\textbf{RT} & \\textbf{Cost} & \\textbf{CVR} & \\textbf{Feas} & \\textbf{RT} & \\textbf{Cost} & \\textbf{CVR} & \\textbf{Feas} & \\textbf{RT} \\\\")
    print("\\midrule")
    for solver_name, result in results.items():
        if result and "by_size" in result:
            by_size = result["by_size"]
            small = by_size.get("small", {})
            medium = by_size.get("medium", {})
            large = by_size.get("large", {})
            print(f"{solver_name} & {small.get('cost', 0):.1f} & {small.get('cvr', 0):.1f} & {small.get('feasibility', 0):.3f} & {small.get('runtime', 0)*1000:.1f} & "
                  f"{medium.get('cost', 0):.1f} & {medium.get('cvr', 0):.1f} & {medium.get('feasibility', 0):.3f} & {medium.get('runtime', 0)*1000:.1f} & "
                  f"{large.get('cost', 0):.1f} & {large.get('cvr', 0):.1f} & {large.get('feasibility', 0):.3f} & {large.get('runtime', 0)*1000:.1f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def generate_individual_solver_tables(results: Dict) -> None:
    print("\n% =========================================================")
    print("% INDIVIDUAL SOLVER ANALYSIS TABLES")
    print("% =========================================================")
    for solver_name, result in results.items():
        if not result:
            continue
        print(f"\n% Table: {solver_name} Detailed Performance")
        print("\\begin{table}[ht]")
        print("\\centering")
        print(f"\\caption{{{solver_name} - Detailed Performance Breakdown}}")
        print(f"\\label{{tab:{solver_name.lower().replace(' ', '_').replace('+', 'plus')}_breakdown}}")
        print("\\begin{tabular}{lcccccc}")
        print("\\toprule")
        print("\\textbf{Configuration} & \\textbf{Size} & \\textbf{Cost} & \\textbf{CVR} & \\textbf{Feas} & \\textbf{Runtime} & \\textbf{TW Violations} \\\\")
        print("\\midrule")
        if "detailed" in result:
            for detail in result["detailed"]:
                metrics = detail["metrics"]
                tw_violations = metrics.get("time_window_violations", 0)
                print(f"{detail['type'].replace('_', ' ')} & {detail['size']} & "
                      f"{metrics['total_cost']:.1f} & {metrics['cvr']:.1f} & "
                      f"{metrics['feasibility']:.3f} & {metrics['runtime']*1000:.1f} & {tw_violations:.2f} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")


def aggregate_metrics(results_list: List[Dict]) -> Dict:
    if not results_list:
        return {}
    metrics = ["total_cost", "cvr", "feasibility", "runtime", "robustness"]
    aggregated = {}
    for metric in metrics:
        values = [r["metrics"].get(metric, 0) for r in results_list]
        aggregated[metric] = sum(values) / len(values) if values else 0
    tw_violations = [r["metrics"].get("time_window_violations", 0) for r in results_list]
    if any(tw > 0 for tw in tw_violations):
        aggregated["time_window_violations"] = sum(tw_violations) / len(tw_violations)
    return aggregated


def generate_comparative_analysis_tables(results: Dict) -> None:
    print("\n% =========================================================")
    print("% COMPARATIVE ANALYSIS TABLES")
    print("% =========================================================")
    solvers = list(results.keys())
    print("\n% Table: Pairwise Solver Comparison Matrix")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Pairwise Solver Performance Comparison (\\% Cost Improvement)}")
    print("\\label{tab:pairwise_comparison}")
    print(f"\\begin{{tabular}}{{l{'c'*len(solvers)}}}")
    print("\\toprule")
    print(" & " + " & ".join([f"\\textbf{{{s}}}" for s in solvers]) + " \\\\")
    print("\\midrule")
    for i, solver1 in enumerate(solvers):
        row = [f"\\textbf{{{solver1}}}"]
        for j, solver2 in enumerate(solvers):
            if i == j:
                row.append("--")
            else:
                result1 = results.get(solver1, {}).get("overall", {})
                result2 = results.get(solver2, {}).get("overall", {})
                cost1, cost2 = result1.get("total_cost", 0), result2.get("total_cost", 0)
                if cost1 > 0 and cost2 > 0:
                    improvement = ((cost1 - cost2) / cost1) * 100
                    row.append(f"{improvement:+.1f}")
                else:
                    row.append("--")
        print(" & ".join(row) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("\n% Table: Constraint Violation Breakdown (TSP-TW)")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Constraint Violation Analysis by Solver (TSP-TW)}")
    print("\\label{tab:constraint_violations}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Total CVR (\\%)} & \\textbf{Feasibility} & \\textbf{TW Violations} \\\\")
    print("\\midrule")
    for solver_name, result in results.items():
        if result and "overall" in result:
            overall = result["overall"]
            tsp_tw = result.get("tsp_tw", {})
            print(f"{solver_name} & {overall.get('cvr', 0):.1f} & {overall.get('feasibility', 0):.3f} & {tsp_tw.get('time_window_violations', 0):.2f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def generate_scalability_tables(results: Dict) -> None:
    print("\n% =========================================================")
    print("% SCALABILITY ANALYSIS TABLES")
    print("% =========================================================")
    print("\n% Table: Runtime Scalability Analysis")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Runtime Scalability Analysis (Growth Factors)}")
    print("\\label{tab:runtime_scalability}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Small} & \\textbf{Small→Med} & \\textbf{Medium} & \\textbf{Med→Large} & \\textbf{Large} & \\textbf{Overall} \\\\")
    print("\\textbf{} & \\textbf{RT (ms)} & \\textbf{Factor} & \\textbf{RT (ms)} & \\textbf{Factor} & \\textbf{RT (ms)} & \\textbf{Factor} \\\\")
    print("\\midrule")
    for solver_name, result in results.items():
        if result and "by_size" in result:
            by_size = result["by_size"]
            small = by_size.get("small", {})
            medium = by_size.get("medium", {})
            large = by_size.get("large", {})
            small_rt = small.get("runtime", 0) * 1000
            medium_rt = medium.get("runtime", 0) * 1000
            large_rt = large.get("runtime", 0) * 1000
            small_to_med = medium_rt / small_rt if small_rt > 0 else 0
            med_to_large = large_rt / medium_rt if medium_rt > 0 else 0
            overall_factor = large_rt / small_rt if small_rt > 0 else 0
            print(f"{solver_name} & {small_rt:.1f} & {small_to_med:.2f}x & {medium_rt:.1f} & {med_to_large:.2f}x & {large_rt:.1f} & {overall_factor:.2f}x \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("\n% Table: Cost Efficiency Analysis")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Cost Efficiency by Instance Size (Cost per Millisecond)}")
    print("\\label{tab:cost_efficiency}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Small} & \\textbf{Medium} & \\textbf{Large} & \\textbf{Efficiency Trend} \\\\")
    print("\\midrule")
    for solver_name, result in results.items():
        if result and "by_size" in result:
            by_size = result["by_size"]
            small = by_size.get("small", {})
            medium = by_size.get("medium", {})
            large = by_size.get("large", {})
            small_eff = small.get("cost", 0) / (small.get("runtime", 0) * 1000) if small.get("runtime", 0) > 0 else 0
            medium_eff = medium.get("cost", 0) / (medium.get("runtime", 0) * 1000) if medium.get("runtime", 0) > 0 else 0
            large_eff = large.get("cost", 0) / (large.get("runtime", 0) * 1000) if large.get("runtime", 0) > 0 else 0
            if large_eff > medium_eff and medium_eff > small_eff:
                trend = "Improving"
            elif large_eff < medium_eff and medium_eff < small_eff:
                trend = "Degrading"
            else:
                trend = "Mixed"
            print(f"{solver_name} & {small_eff:.1f} & {medium_eff:.1f} & {large_eff:.1f} & {trend} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("\n% Table: Overall Scalability Summary")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Overall Scalability Summary by Solver}")
    print("\\label{tab:scalability_summary}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Runtime} & \\textbf{Cost} & \\textbf{Feasibility} & \\textbf{Robustness} & \\textbf{Overall} & \\textbf{Rank} \\\\")
    print("\\textbf{} & \\textbf{Scale} & \\textbf{Scale} & \\textbf{Scale} & \\textbf{Scale} & \\textbf{Score} & \\textbf{} \\\\")
    print("\\midrule")
    scalability_scores = []
    for solver_name, result in results.items():
        if result and "by_size" in result:
            by_size = result["by_size"]
            small = by_size.get("small", {})
            large = by_size.get("large", {})
            rt_scale = large.get("runtime", 0) / small.get("runtime", 0) if small.get("runtime", 0) > 0 else float("inf")
            cost_scale = large.get("cost", 0) / small.get("cost", 0) if small.get("cost", 0) > 0 else float("inf")
            feas_scale = (1 - large.get("feasibility", 0)) / (1 - small.get("feasibility", 0)) if small.get("feasibility", 0) < 1 else 1
            rob_scale = large.get("robustness", 0) / small.get("robustness", 0) if small.get("robustness", 0) > 0 else 1
            overall_score = rt_scale * 0.4 + cost_scale * 0.3 + feas_scale * 0.2 + rob_scale * 0.1
            scalability_scores.append((solver_name, rt_scale, cost_scale, feas_scale, rob_scale, overall_score))
    scalability_scores.sort(key=lambda x: x[5])
    for rank, (solver_name, rt_scale, cost_scale, feas_scale, rob_scale, overall_score) in enumerate(scalability_scores, 1):
        print(f"{solver_name} & {rt_scale:.2f} & {cost_scale:.2f} & {feas_scale:.2f} & {rob_scale:.2f} & {overall_score:.2f} & {rank} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def print_final_comparison(results: Dict) -> None:
    print("\nOverall Performance Comparison:")
    print("-" * 100)
    data = []
    solver_names = []
    for solver_name, result in results.items():
        if result and "overall" in result:
            solver_names.append(solver_name)
            data.append(result["overall"])
    if not data:
        print("No results to compare")
        return
    print(f"{'Metric':<15} ", end="")
    for name in solver_names:
        print(f"{name:<15} ", end="")
    print("Best")
    print("-" * 100)
    metrics = ["total_cost", "cvr", "feasibility", "runtime", "robustness"]
    for metric in metrics:
        values = []
        print(f"{metric:<15} ", end="")
        for solver_data in data:
            value = solver_data.get(metric, 0)
            if metric == "runtime":
                value *= 1000
            values.append(value)
            print(f"{value:<15.1f} ", end="")
        best_idx = values.index(min(values)) if metric in ["total_cost", "cvr", "runtime", "robustness"] else values.index(max(values))
        print(solver_names[best_idx])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(run: Optional[str] = None, format: Optional[str] = None, sizes: Optional[List[int]] = None):
    import random
    parser = argparse.ArgumentParser(description="Unified TSP-TW evaluation (npz or torch format)")
    parser.add_argument("--format", choices=["npz", "torch"], default="npz", help="Dataset format: npz or torch")
    parser.add_argument("--run", type=str, default=None, help="TSP-TW run timestamp (e.g. 2026-02-23_15-38-58); default: latest")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 50], help="Instance sizes to evaluate")
    # Allow programmatic call: main(run=..., format=..., sizes=...) without parsing argv
    if run is not None or format is not None or sizes is not None:
        args = argparse.Namespace(
            format=format or "npz",
            run=run,
            sizes=sizes if sizes is not None else [10, 20, 50],
        )
    else:
        args = parser.parse_args()

    np.random.seed(PAPER_SEED)
    random.seed(PAPER_SEED)

    base_path = os.path.join(os.path.dirname(__file__), "data") + os.sep
    run_timestamp = args.run or _get_latest_tsptw_run(base_path)
    if not run_timestamp:
        print("No TSP-TW run found under data/tsp_tw/. Generate data first (e.g. python generate_tsp_tw_instances.py --format torch).")
        sys.exit(1)

    print("=" * 80)
    print("UNIFIED TSP-TW EVALUATION (format={}, run={})".format(args.format, run_timestamp))
    print("=" * 80)
    print("Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if OR_TOOLS_AVAILABLE:
        print("OR-Tools is available - using real OR-Tools solver")
        solver_name = "OR-Tools"
    else:
        print("OR-Tools not available - using fallback solver (NN+2opt)")
        solver_name = "OR-Tools (fallback)"

    start_time = time.time()
    eval_results_dir = os.path.join(os.path.dirname(__file__), "eval_results", run_timestamp + "_" + args.format)
    os.makedirs(eval_results_dir, exist_ok=True)

    config = {
        "test_sizes": args.sizes,
        "max_instances_per_file": 10,
        "num_realizations": 1,
        "use_paper_protocol": True,
    }
    solvers = [
        (NN2optSolver, "NN+2opt"),
        (TabuSearchSolver, "Tabu Search"),
        (ACOSolver, "ACO"),
        (ORToolsSolver, solver_name),
    ]
    results = {}
    for solver_class, name in solvers:
        print(f"\n{'='*80}")
        print(f"TESTING: {name} Solver")
        print("=" * 80)
        try:
            results[name] = evaluate_solver(
                solver_class=solver_class,
                solver_name=name,
                base_path=base_path,
                run=run_timestamp,
                data_format=args.format,
                results_dir=eval_results_dir,
                sizes=config["test_sizes"],
                max_instances_per_file=config["max_instances_per_file"],
                num_realizations=config["num_realizations"],
                use_paper_protocol=config["use_paper_protocol"],
            )
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            results[name] = None

    print(f"\n{'='*80}")
    print("COMPREHENSIVE COMPARISON - ALL SOLVERS")
    print("=" * 80)
    print_final_comparison(results)

    print(f"\n{'='*80}")
    print("GENERATING ALL LATEX TABLES")
    print("=" * 80)
    latex_path = os.path.join(eval_results_dir, "latex_tables.tex")
    old_stdout = sys.stdout
    try:
        with open(latex_path, "w") as latex_file:
            sys.stdout = latex_file
            generate_docx_tables(results)
            generate_detailed_tables(results)
            generate_individual_solver_tables(results)
            generate_comparative_analysis_tables(results)
            generate_scalability_tables(results)
    finally:
        sys.stdout = old_stdout
    print(f"LaTeX tables written to {latex_path}")

    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("TESTING COMPLETED")
    print("=" * 80)
    print(f"Total execution time: {total_time:.2f} seconds")
    if OR_TOOLS_AVAILABLE:
        print("Real OR-Tools was used in this evaluation.")
    else:
        print("OR-Tools fallback was used.")
    return results


if __name__ == "__main__":
    main()
