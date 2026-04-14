from __future__ import annotations

"""
Warm-up analysis for the IR outpatient scheduling model.

Strategy
--------
This script follows the deletion-point idea described in
"Design and Analysis of Simulation Experiments.pdf" for steady-state simulation:

1. Determine a deletion point d so initialization bias is effectively removed.
2. Estimate the response-versus-index curve E(Y_i) and look for where it
   appears to stop changing.
3. After choosing d, use a measured run length m that is at least about 10d.

This model exposes H only as an end-of-run summary, not as an internal weekly
time series. To build the Y_i sequence for H, this script runs:

    warmup_weeks = 0, 1, 2, ..., MAX_ANALYSIS_WEEKS - 1

with MEASURED_WEEKS_PER_POINT measured weeks after each deletion point. For a
fixed seed, this gives a deleted-observation H value on the same underlying
sample path. Averaging those values across replications estimates the E(Y_i)
curve from the book's step-1 idea.

Notes
-----
1. Warm-up is a time/observation deletion point, not a replication count.
   The x-axis is elapsed week.
2. All weekly observations with numeric H are used in the deletion-point
   estimate, including weeks with zero completed patients.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

import Policy_defined
from input_loader import load_all_ir_inputs
from simulation_model import run_replications

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# USER SETTINGS
# =========================================================
ARRIVAL_JSON_PATH = "arrival_model_params.json"
SERVICE_JSON_PATH = "services rate.json"
RAW_DATA_PATH = "df_selected.xlsx"

POLICY_TIMETABLE = "R1"  # "R1" or "R2"
DAILY_QIK: list[list[int]] | None = None

BASE_SEED = 123
NUM_REPLICATIONS = 20
MAX_ANALYSIS_WEEKS = 50
MEASURED_WEEKS_PER_POINT = 4

SUGGESTED_WARMUP_WEEK = 20

OUTPUT_DIR = Path("doe_output")
OBSERVATION_CSV_NAME = "warmup_h_observations.csv"
SUMMARY_CSV_NAME = "warmup_h_summary.csv"
PLOT_FILE_NAME = "warmup_h_pdf_style_plot.png"


# =========================================================
# DATA OBJECTS
# =========================================================
@dataclass
class WarmupRecommendation:
    warmup_week: int | None
    note: str


# =========================================================
# HELPERS
# =========================================================
def build_policy():
    timetable_name = str(POLICY_TIMETABLE).strip().upper()

    if timetable_name == "R1":
        if DAILY_QIK is None:
            return Policy_defined.example_policy_R1()
        return Policy_defined.build_bruteforce_policy_R1(DAILY_QIK)

    if timetable_name == "R2":
        if DAILY_QIK is None:
            return Policy_defined.example_policy_R2()
        return Policy_defined.build_bruteforce_policy_R2(DAILY_QIK)

    raise ValueError("POLICY_TIMETABLE must be 'R1' or 'R2'.")


def analyze_single_deleted_week(
    *,
    loaded_inputs,
    policy,
    replication: int,
    warmup_weeks: int,
    measured_weeks: int,
    seed: int,
) -> dict:
    rep_df = run_replications(
        num_replications=1,
        num_weeks=measured_weeks,
        loaded_inputs=loaded_inputs,
        policy=policy,
        base_seed=seed,
        warmup_weeks=warmup_weeks,
    )

    row = rep_df.iloc[0]
    completed_count = int(row["completed_count"])
    raw_h = float(row["H"])
    h_for_analysis = raw_h
    used_for_analysis = not pd.isna(h_for_analysis)

    return {
        "replication": replication,
        "seed": seed,
        "warmup_weeks": warmup_weeks,
        "elapsed_week": warmup_weeks + 1,
        "measured_weeks": measured_weeks,
        "scheduled_count": int(row["scheduled_count"]),
        "completed_count": completed_count,
        "unscheduled_count": int(row["unscheduled_count"]),
        "unscheduled_count_total": int(row["unscheduled_count_total"]),
        "Z1_wait_time": float(row["Z1_wait_time"]),
        "Z2_overtime": float(row["Z2_overtime"]),
        "Z3_congestion": float(row["Z3_congestion"]),
        "H_raw": raw_h,
        "H_analysis": h_for_analysis,
        "used_for_analysis": used_for_analysis,
    }


def collect_deleted_week_observations(
    *,
    loaded_inputs,
    policy,
    num_replications: int,
    max_analysis_weeks: int,
    measured_weeks: int,
    base_seed: int,
) -> pd.DataFrame:
    rows = []

    for rep in range(1, num_replications + 1):
        seed = base_seed + rep - 1
        print(f"Running replication {rep}/{num_replications} with seed {seed}")

        for warmup_weeks in range(max_analysis_weeks):
            rows.append(
                analyze_single_deleted_week(
                    loaded_inputs=loaded_inputs,
                    policy=policy,
                    replication=rep,
                    warmup_weeks=warmup_weeks,
                    measured_weeks=measured_weeks,
                    seed=seed,
                )
            )

    return pd.DataFrame(rows)


def summarize_warmup_curve(observations_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        observations_df.groupby("elapsed_week", as_index=False)
        .agg(
            mean_H=("H_analysis", "mean"),
            std_H=("H_analysis", "std"),
            valid_replications=("used_for_analysis", "sum"),
            total_replications=("replication", "nunique"),
            mean_completed_count=("completed_count", "mean"),
            mean_unscheduled_count=("unscheduled_count", "mean"),
            mean_raw_H=("H_raw", "mean"),
        )
        .sort_values("elapsed_week", ignore_index=True)
    )

    summary_df["dropped_replications"] = (
        summary_df["total_replications"] - summary_df["valid_replications"]
    )

    return summary_df


def fixed_warmup_recommendation() -> WarmupRecommendation:
    return WarmupRecommendation(
        warmup_week=SUGGESTED_WARMUP_WEEK,
        note=f"Suggested warm-up week is fixed at {SUGGESTED_WARMUP_WEEK} weeks.",
    )


def plot_warmup_curve(
    summary_df: pd.DataFrame,
    recommendation: WarmupRecommendation,
    plot_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        summary_df["elapsed_week"],
        summary_df["mean_H"],
        marker="o",
        linewidth=2.0,
        label="Estimated E(Y_i) for H",
    )

    if recommendation.warmup_week is not None:
        ax.axvline(
            recommendation.warmup_week,
            color="red",
            linestyle="--",
            linewidth=1.8,
            label=f"Suggested deletion point = week {recommendation.warmup_week}",
        )

    ax.set_title("PDF-Style Deletion-Point Analysis for H")
    ax.set_xlabel("Elapsed Week")
    ax.set_ylabel("Mean H")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def print_configuration(policy) -> None:
    print("Warm-up analysis configuration")
    print(f"Policy timetable = {POLICY_TIMETABLE}")
    print(f"Policy name = {policy.timetable.name}")
    print(f"Daily Qik override supplied = {DAILY_QIK is not None}")
    print(f"Replications = {NUM_REPLICATIONS}")
    print(f"Max analysis weeks = {MAX_ANALYSIS_WEEKS}")
    print(f"Measured weeks per point = {MEASURED_WEEKS_PER_POINT}")
    print(f"Fixed suggested warm-up week = {SUGGESTED_WARMUP_WEEK}")
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loaded_inputs = load_all_ir_inputs(
        arrival_json_path=ARRIVAL_JSON_PATH,
        service_json_path=SERVICE_JSON_PATH,
        raw_data_path=RAW_DATA_PATH,
    )
    policy = build_policy()

    print_configuration(policy)

    observations_df = collect_deleted_week_observations(
        loaded_inputs=loaded_inputs,
        policy=policy,
        num_replications=NUM_REPLICATIONS,
        max_analysis_weeks=MAX_ANALYSIS_WEEKS,
        measured_weeks=MEASURED_WEEKS_PER_POINT,
        base_seed=BASE_SEED,
    )
    summary_df = summarize_warmup_curve(observations_df)
    recommendation = fixed_warmup_recommendation()

    observation_csv_path = OUTPUT_DIR / OBSERVATION_CSV_NAME
    summary_csv_path = OUTPUT_DIR / SUMMARY_CSV_NAME
    plot_path = OUTPUT_DIR / PLOT_FILE_NAME

    observations_df.to_csv(observation_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    plot_warmup_curve(summary_df, recommendation, plot_path)

    print("\nWeekly warm-up summary")
    print(
        summary_df[
            [
                "elapsed_week",
                "mean_H",
                "valid_replications",
                "dropped_replications",
                "mean_completed_count",
            ]
        ].to_string(index=False)
    )

    dropped_total = int((~observations_df["used_for_analysis"]).sum())
    print("\nRecommendation")
    print(recommendation.note)
    print(f"Suggested warm-up week = {recommendation.warmup_week}")
    print(f"Dropped zero-completion weekly observations = {dropped_total}")

    if recommendation.warmup_week is not None:
        recommended_run_length = 10 * recommendation.warmup_week
        print(
            f"Rule-of-thumb production run length after calibration: "
            f"at least about {recommended_run_length} measured weeks."
        )

    print("\nOutputs")
    print(f"Observations CSV = {observation_csv_path}")
    print(f"Summary CSV = {summary_csv_path}")
    print(f"Plot = {plot_path}")


if __name__ == "__main__":
    main()
