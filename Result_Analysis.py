from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EVALUATE_POLICY_DIR = Path("evaluate_policy")
ANALYSIS_OUTPUT_DIR = Path("result_analysis_outputs")
Z_VALUE_95 = 1.96


def _ensure_output_dir() -> Path:
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return ANALYSIS_OUTPUT_DIR


def _method_from_policy_name(policy_name: str) -> str:
    if str(policy_name).endswith("_SAA2"):
        return "SAA"
    if str(policy_name).endswith("_SubsetKN"):
        return "KN+Subset"
    if str(policy_name).endswith("_Lin"):
        return "Lin_Stage2"
    return "Unknown"


def _timetable_from_policy_name(policy_name: str) -> str:
    return str(policy_name).split("_")[0].upper()


def _policy_label(policy_name: str) -> str:
    method = _method_from_policy_name(policy_name)
    timetable = _timetable_from_policy_name(policy_name)
    return f"{method} - {timetable}"


def _load_eval_tables() -> dict[str, pd.DataFrame]:
    if not EVALUATE_POLICY_DIR.exists():
        raise FileNotFoundError(f"Evaluation folder not found: {EVALUATE_POLICY_DIR}")

    policy_tables: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(EVALUATE_POLICY_DIR.glob("*_eval.csv")):
        df = pd.read_csv(csv_path)
        if "policy_name" not in df.columns or "H" not in df.columns:
            continue
        unique_policy_names = df["policy_name"].dropna().astype(str).unique()
        if len(unique_policy_names) != 1:
            raise ValueError(f"Expected exactly one policy_name in {csv_path}, found {len(unique_policy_names)}.")
        policy_name = unique_policy_names[0]
        policy_tables[policy_name] = df.copy()

    if not policy_tables:
        raise ValueError(f"No per-policy evaluation CSV files were found in {EVALUATE_POLICY_DIR}.")
    return policy_tables


def _build_summary_table(policy_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for policy_name, df in policy_tables.items():
        h_values = df["H"].to_numpy(dtype=float)
        n = int(len(h_values))
        mean_h = float(h_values.mean())
        std_h = float(h_values.std(ddof=1)) if n > 1 else 0.0
        se_h = std_h / np.sqrt(n) if n > 0 else 0.0
        ci_half_width = Z_VALUE_95 * se_h

        rows.append(
            {
                "policy_name": policy_name,
                "policy": _policy_label(policy_name),
                "method": _method_from_policy_name(policy_name),
                "timetable": _timetable_from_policy_name(policy_name),
                "n_replications": n,
                "mean_H": mean_h,
                "std_H": std_h,
                "SE_H": se_h,
                "CI95_lower_H": mean_h - ci_half_width,
                "CI95_upper_H": mean_h + ci_half_width,
                "CI95_width_H": 2.0 * ci_half_width,
                "mean_Z1": float(df["Z1_wait_time"].mean()) if "Z1_wait_time" in df.columns else np.nan,
                "mean_Z2": float(df["Z2_overtime"].mean()) if "Z2_overtime" in df.columns else np.nan,
                "mean_Z3": float(df["Z3_congestion"].mean()) if "Z3_congestion" in df.columns else np.nan,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(
        ["mean_H", "policy_name"], ascending=[True, True], ignore_index=True
    )
    return summary_df


def _save_summary_table(summary_df: pd.DataFrame) -> Path:
    output_path = _ensure_output_dir() / "evaluate_policy_H_summary.csv"
    summary_df.to_csv(output_path, index=False)
    return output_path


def _save_box_plot(summary_df: pd.DataFrame, policy_tables: dict[str, pd.DataFrame]) -> Path:
    output_path = _ensure_output_dir() / "evaluate_policy_H_boxplot.png"
    ordered_policy_names = summary_df["policy_name"].tolist()
    labels = summary_df["policy"].tolist()
    box_data = [policy_tables[policy_name]["H"].to_numpy(dtype=float) for policy_name in ordered_policy_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(box_data, tick_labels=labels, patch_artist=True)
    ax.set_xlabel("Policy")
    ax.set_ylabel("Objective H")
    ax.set_title("Box Plot of H Across 100 Replications")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _save_mean_ci_plot(summary_df: pd.DataFrame) -> Path:
    output_path = _ensure_output_dir() / "evaluate_policy_H_mean_CI95.png"

    x = np.arange(len(summary_df))
    y = summary_df["mean_H"].to_numpy(dtype=float)
    yerr = (summary_df["CI95_upper_H"] - summary_df["mean_H"]).to_numpy(dtype=float)
    labels = summary_df["policy"].tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        color="blue",
        ecolor="blue",
        elinewidth=1.5,
        capsize=5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Objective H")
    ax.set_title("Mean H with 95% Confidence Intervals")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:
    policy_tables = _load_eval_tables()
    summary_df = _build_summary_table(policy_tables)
    summary_path = _save_summary_table(summary_df)
    box_plot_path = _save_box_plot(summary_df, policy_tables)
    mean_ci_plot_path = _save_mean_ci_plot(summary_df)

    display_df = summary_df.copy()
    for col in [
        "mean_H",
        "std_H",
        "SE_H",
        "CI95_lower_H",
        "CI95_upper_H",
        "CI95_width_H",
        "mean_Z1",
        "mean_Z2",
        "mean_Z3",
    ]:
        display_df[col] = display_df[col].round(4)

    print("H Summary From evaluate_policy Folder:")
    print(display_df.to_string(index=False))
    print(f"\nSaved summary table to {summary_path}")
    print(f"Saved H box plot to {box_plot_path}")
    print(f"Saved H mean/CI plot to {mean_ci_plot_path}")


if __name__ == "__main__":
    main()
