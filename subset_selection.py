from __future__ import annotations

"""
Run only the subset-selection stage for the policy search.

This script reuses the existing logic in
Optimization_Subset_Selection+KN.py, but it stops after subset
screening so you can see how long the subset stage takes by itself.
"""

import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd


def load_optimizer_module():
    """Load Optimization_Subset_Selection+KN.py by file path."""
    module_path = Path(__file__).with_name("Optimization_Subset_Selection+KN.py")
    spec = importlib.util.spec_from_file_location("optimization_subset_selection_kn", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    opt = load_optimizer_module()
    overall_start = time.perf_counter()
    all_results = []

    for search_timetable in opt._resolve_search_timetables():
        print("\n" + "=" * 80)
        opt._set_active_search(search_timetable)
        opt._require_policies()

        num_systems = len(opt.POLICIES)
        print("Candidate policy set")
        print(f"Timetable = {opt.ACTIVE_TIMETABLE}")
        print(f"Number of candidate systems = {num_systems}")
        opt._print_df_preview(opt.policy_table(range(1, num_systems + 1)))

        subset_start = time.perf_counter()
        subset, subset_df = opt.subset_crn(
            k=num_systems,
            alpha=opt.SUBSET_ALPHA,
            n=opt.SUBSET_INITIAL_REPS,
            seed=opt.BASE_SEED,
        )
        subset_elapsed = time.perf_counter() - subset_start

        kept_subset_df = opt.subset_result_table(subset, subset_df)

        print("\nSubset selection only")
        print(f"subset_reps = {opt.SUBSET_INITIAL_REPS}")
        print(f"subset_elapsed_seconds = {subset_elapsed:.2f}")
        print(f"subset_elapsed_minutes = {subset_elapsed / 60.0:.2f}")
        print(f"subset_elapsed_hours = {subset_elapsed / 3600.0:.2f}")

        print("\nSubset summary preview")
        opt._print_df_preview(subset_df)

        print("\nPolicies kept after subset screening")
        print(f"Number of systems kept after subset screening = {len(subset)}")
        print(kept_subset_df.to_string(index=False))

        all_results.append(
            {
                "timetable": opt.ACTIVE_TIMETABLE,
                "num_systems": num_systems,
                "num_kept": len(subset),
                "subset_elapsed_seconds": subset_elapsed,
                "best_subset_mean_H": float(subset_df["mean_H_subset"].min()),
            }
        )

    total_elapsed = time.perf_counter() - overall_start

    if all_results:
        print("\n" + "=" * 80)
        print("Subset-only runtime summary")
        summary_df = pd.DataFrame(all_results)
        summary_df["subset_elapsed_minutes"] = summary_df["subset_elapsed_seconds"] / 60.0
        summary_df["subset_elapsed_hours"] = summary_df["subset_elapsed_seconds"] / 3600.0
        print(summary_df.to_string(index=False))

    print(f"\nTotal wall-clock seconds = {total_elapsed:.2f}")
    print(f"Total wall-clock minutes = {total_elapsed / 60.0:.2f}")
    print(f"Total wall-clock hours = {total_elapsed / 3600.0:.2f}")


if __name__ == "__main__":
    main()
