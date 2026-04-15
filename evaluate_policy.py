"""
Evaluate saved policies and export per-replication simulation results.

Current behavior:
    - If run with no arguments, the script evaluates the default six selected
      policies (SAA2, KN+Subset, and Lin for R1/R2) using the current defaults.

Usage examples:
    # Run the default six-policy batch:
    python evaluate_policy.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from optimization_common import load_common_inputs, make_policy_name, resolve_timetable
from simulation_model import (
    apply_feasibility_to_qik,
    policy_from_qik,
    run_replications,
)

BASE_DIR = Path(__file__).resolve().parent

# Output folders to search for policy name → Qik mapping
OUTPUT_DIRS = [
    BASE_DIR / "optimization_subset_selection_kn_simplified_outputs5",
    BASE_DIR / "SAA2_output_folder",
    BASE_DIR / "optimization_lin_stage2_outputs",
]

EVAL_OUTPUT_DIR = BASE_DIR / "evaluate_policy"
DEFAULT_POLICY_NAMES = [
    "R1_840a3320_SAA2",
    "R2_7652d98c_SAA2",
    "R1_c5d534fd_SubsetKN",
    "R2_effc06e5_SubsetKN",
    "R1_3d503d0e_Lin",
    "R2_34b04524_Lin",
]


def _search_qik_by_policy_name(policy_name: str) -> tuple[str, np.ndarray] | None:
    """Search output folders for the weekly Qik matching a policy name."""
    timetable_name = policy_name.split("_")[0].upper()

    for output_dir in OUTPUT_DIRS:
        if not output_dir.exists():
            continue

        # SubsetKN: support both old and current result-file names
        subset_patterns = (
            "*_final_policy_results.csv",
            "*_subset_policy_results.csv",
        )
        seen_csv_paths: set[Path] = set()
        for pattern in subset_patterns:
            for csv_path in output_dir.glob(pattern):
                if csv_path in seen_csv_paths:
                    continue
                seen_csv_paths.add(csv_path)
                try:
                    df = pd.read_csv(csv_path)
                    if "policy" in df.columns and "weekly_qik_json" in df.columns:
                        match = df[df["policy"] == policy_name]
                        if not match.empty:
                            qik = np.array(json.loads(match.iloc[0]["weekly_qik_json"]), dtype=int)
                            return timetable_name, qik
                except Exception:
                    continue

        # Lin: summary JSON with policy_name and best_qik_json
        for json_path in output_dir.glob("*_summary.json"):
            try:
                with open(json_path) as f:
                    data = json.load(f)
                if data.get("policy_name") == policy_name:
                    qik = np.array(data["best_qik_json"], dtype=int)
                    return timetable_name, qik
            except Exception:
                continue

        # SAA / SAA2: summary.csv with policy_name, plus separate weekly_qik CSVs
        summary_path = output_dir / "summary.csv"
        if summary_path.exists():
            try:
                df = pd.read_csv(summary_path)
                if "policy_name" in df.columns:
                    match = df[df["policy_name"] == policy_name]
                    if not match.empty:
                        tt = match.iloc[0]["timetable"].strip().lower()
                        qik_csv = output_dir / f"{tt}_best_weekly_qik.csv"
                        if qik_csv.exists():
                            qik_df = pd.read_csv(qik_csv, index_col=0)
                            qik = qik_df.values.astype(int)
                            return timetable_name, qik
            except Exception:
                continue

    return None


def _resolve_policy_inputs(
    policy_name: str | None,
    timetable_name: str | None,
    qik_json: str | None,
) -> tuple[str, str, np.ndarray]:
    if qik_json is not None:
        if timetable_name is None:
            print("ERROR: --timetable is required when using --qik-json")
            sys.exit(1)
        resolved_timetable = timetable_name.strip().upper()
        weekly_qik = np.array(json.loads(qik_json), dtype=int)
        resolved_policy_name = policy_name or make_policy_name(resolved_timetable, "eval", weekly_qik)
        return resolved_timetable, resolved_policy_name, weekly_qik

    if policy_name is not None:
        result = _search_qik_by_policy_name(policy_name)
        if result is None:
            print(f"ERROR: Could not find policy '{policy_name}' in any output folder.")
            print("Searched:", [str(d) for d in OUTPUT_DIRS if d.exists()])
            sys.exit(1)
        resolved_timetable, weekly_qik = result
        return resolved_timetable, policy_name, weekly_qik

    print("ERROR: Provide either --policy-name or --qik-json")
    sys.exit(1)


def _evaluate_policy(
    policy_name: str,
    timetable_name: str,
    weekly_qik: np.ndarray,
    loaded_inputs,
    reps: int,
    seed: int,
    num_weeks: int,
    warmup_weeks: int,
) -> dict:
    timetable = resolve_timetable(timetable_name)
    weekly_qik = apply_feasibility_to_qik(weekly_qik, timetable)
    policy = policy_from_qik(weekly_qik, timetable)

    print(f"\nPolicy: {policy_name}")
    print(f"Timetable: {timetable_name}")
    print(f"Weekly Qik totals: {weekly_qik.sum(axis=1).tolist()}")
    print(f"Running {reps} replications (seed={seed}, weeks={num_weeks}, warmup={warmup_weeks})...")

    rep_df = run_replications(
        num_replications=reps,
        num_weeks=num_weeks,
        loaded_inputs=loaded_inputs,
        policy=policy,
        base_seed=seed,
        warmup_weeks=warmup_weeks,
    )

    mean_h = float(rep_df["H"].mean())
    std_h = float(rep_df["H"].std(ddof=1)) if len(rep_df) > 1 else 0.0
    mean_z1 = float(rep_df["Z1_wait_time"].mean())
    mean_z2 = float(rep_df["Z2_overtime"].mean())
    mean_z3 = float(rep_df["Z3_congestion"].mean())

    print(f"\nResults ({len(rep_df)} replications):")
    print(f"  Mean H  = {mean_h:.4f} +/- {std_h:.4f}")
    print(f"  Mean Z1 = {mean_z1:.4f} days")
    print(f"  Mean Z2 = {mean_z2:.4f} hours")
    print(f"  Mean Z3 = {mean_z3:.4f}")

    rep_out = rep_df.copy()
    rep_out.insert(0, "policy_name", policy_name)

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = policy_name.replace("/", "_")
    output_path = EVAL_OUTPUT_DIR / f"{safe_name}_eval.csv"
    rep_out.to_csv(output_path, index=False)
    print(f"\nSaved per-replication results to {output_path}")

    return {
        "policy_name": policy_name,
        "timetable": timetable_name,
        "mean_H": mean_h,
        "std_H": std_h,
        "mean_Z1": mean_z1,
        "mean_Z2": mean_z2,
        "mean_Z3": mean_z3,
        "output_path": str(output_path),
        "replications_df": rep_out,
    }


def _run_default_policy_set(args, loaded_inputs) -> None:
    print("No policy arguments provided. Running the default policy set:")
    for name in DEFAULT_POLICY_NAMES:
        print(f"  - {name}")

    results = []
    all_replications = []
    for policy_name in DEFAULT_POLICY_NAMES:
        timetable_name, resolved_policy_name, weekly_qik = _resolve_policy_inputs(
            policy_name=policy_name,
            timetable_name=None,
            qik_json=None,
        )
        result = _evaluate_policy(
            policy_name=resolved_policy_name,
            timetable_name=timetable_name,
            weekly_qik=weekly_qik,
            loaded_inputs=loaded_inputs,
            reps=args.reps,
            seed=args.seed,
            num_weeks=args.num_weeks,
            warmup_weeks=args.warmup_weeks,
        )
        results.append(result)
        all_replications.append(result["replications_df"])

    summary_df = pd.DataFrame(
        [
            {
                "policy_name": result["policy_name"],
                "timetable": result["timetable"],
                "reps": int(args.reps),
                "seed": int(args.seed),
                "num_weeks": int(args.num_weeks),
                "warmup_weeks": int(args.warmup_weeks),
                "mean_H": result["mean_H"],
                "std_H": result["std_H"],
                "mean_Z1": result["mean_Z1"],
                "mean_Z2": result["mean_Z2"],
                "mean_Z3": result["mean_Z3"],
                "output_path": result["output_path"],
            }
            for result in results
        ]
    ).sort_values(["mean_H", "policy_name"], ascending=[True, True], ignore_index=True)

    combined_df = pd.concat(all_replications, ignore_index=True)
    summary_path = EVAL_OUTPUT_DIR / "default_policy_set_summary.csv"
    combined_path = EVAL_OUTPUT_DIR / "default_policy_set_all_replications.csv"
    summary_df.to_csv(summary_path, index=False)
    combined_df.to_csv(combined_path, index=False)

    print("\nDefault policy set summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved default policy summary to {summary_path}")
    print(f"Saved combined replications to {combined_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a policy by name or Qik JSON.")
    parser.add_argument("--policy-name", default=None, help="Policy name to look up")
    parser.add_argument("--timetable", default=None, help="Timetable (R1 or R2), required with --qik-json")
    parser.add_argument("--qik-json", default=None, help="Weekly Qik as JSON 2x40 array")
    parser.add_argument("--reps", type=int, default=100, help="Number of replications")
    parser.add_argument("--seed", type=int, default=20123, help="Base seed")
    parser.add_argument("--num-weeks", type=int, default=180, help="Simulation weeks")
    parser.add_argument("--warmup-weeks", type=int, default=20, help="Warmup weeks")
    args = parser.parse_args()
    loaded_inputs = load_common_inputs()
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.policy_name is None and args.qik_json is None:
        _run_default_policy_set(args, loaded_inputs)
        return

    timetable_name, policy_name, weekly_qik = _resolve_policy_inputs(
        policy_name=args.policy_name,
        timetable_name=args.timetable,
        qik_json=args.qik_json,
    )
    _evaluate_policy(
        policy_name=policy_name,
        timetable_name=timetable_name,
        weekly_qik=weekly_qik,
        loaded_inputs=loaded_inputs,
        reps=args.reps,
        seed=args.seed,
        num_weeks=args.num_weeks,
        warmup_weeks=args.warmup_weeks,
    )


if __name__ == "__main__":
    main()
