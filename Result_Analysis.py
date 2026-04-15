from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EVALUATE_POLICY_DIR = Path("evaluate_policy")
ANALYSIS_OUTPUT_DIR = Path("result_analysis_outputs")
Z_VALUE_95 = 1.96
BASELINE_POLICY_NAME = "R1_840a3320_SAA2"
TITLE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 16
TICK_LABEL_FONT_SIZE = 15
SHOW_PLOTS = True
Z1_NORMALIZER = 28.0
Z2_NORMALIZER = 2.5
Z3_NORMALIZER = 2.0
MERGED_PLOT_GROUPS = [
    (
        ("R1_3d503d0e_Lin", "R1_c5d534fd_SubsetKN"),
        "Lin_Stage2 / KN+Subset - R1",
    ),
]


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


def _format_policy_tick_label(label: str) -> str:
    return label.replace(" / ", " /\n").replace(" - ", "\n")


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


def _build_plot_entries(summary_df: pd.DataFrame, policy_tables: dict[str, pd.DataFrame]) -> list[dict]:
    entries: list[dict] = []
    used_policy_names: set[str] = set()

    for policy_names, label in MERGED_PLOT_GROUPS:
        if all(policy_name in summary_df["policy_name"].values for policy_name in policy_names):
            base_policy_name = policy_names[0]
            base_row = summary_df.loc[summary_df["policy_name"] == base_policy_name].iloc[0]
            entries.append(
                {
                    "policy_name": base_policy_name,
                    "label": label,
                    "row": base_row,
                    "h_values": policy_tables[base_policy_name]["H"].to_numpy(dtype=float),
                }
            )
            used_policy_names.update(policy_names)

    for row in summary_df.itertuples(index=False):
        if row.policy_name in used_policy_names:
            continue
        current_row = summary_df.loc[summary_df["policy_name"] == row.policy_name].iloc[0]
        entries.append(
            {
                "policy_name": row.policy_name,
                "label": row.policy,
                "row": current_row,
                "h_values": policy_tables[row.policy_name]["H"].to_numpy(dtype=float),
            }
        )

    return entries


def _save_box_plot(summary_df: pd.DataFrame, policy_tables: dict[str, pd.DataFrame]) -> Path:
    output_path = _ensure_output_dir() / "evaluate_policy_H_boxplot.png"
    plot_entries = _build_plot_entries(summary_df, policy_tables)
    labels = [entry["label"] for entry in plot_entries]
    box_data = [entry["h_values"] for entry in plot_entries]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(box_data, tick_labels=labels, patch_artist=True)
    ax.set_xlabel("Policy", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Objective H", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title("Objective Value H Across 100 Replications", fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    return output_path


def _save_mean_ci_plot(summary_df: pd.DataFrame, policy_tables: dict[str, pd.DataFrame]) -> Path:
    output_path = _ensure_output_dir() / "evaluate_policy_H_mean_CI95.png"

    plot_entries = _build_plot_entries(summary_df, policy_tables)
    x = np.arange(len(plot_entries))
    y = np.array([float(entry["row"].mean_H) for entry in plot_entries], dtype=float)
    yerr = np.array(
        [float(entry["row"].CI95_upper_H) - float(entry["row"].mean_H) for entry in plot_entries],
        dtype=float,
    )
    labels = [entry["label"] for entry in plot_entries]

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
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_FONT_SIZE)
    ax.set_xlabel("Policy", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Objective H", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title("Mean Objective Value H with 95% Confidence Intervals", fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    return output_path


def _save_normalized_z_bar_plot(summary_df: pd.DataFrame, policy_tables: dict[str, pd.DataFrame]) -> Path:
    output_path = _ensure_output_dir() / "evaluate_policy_Z_normalized_bar.png"

    plot_entries = _build_plot_entries(summary_df, policy_tables)
    labels = [entry["label"] for entry in plot_entries]
    tick_labels = [_format_policy_tick_label(label) for label in labels]
    z1_values = np.array([float(entry["row"].mean_Z1) / Z1_NORMALIZER for entry in plot_entries], dtype=float)
    z2_values = np.array([float(entry["row"].mean_Z2) / Z2_NORMALIZER for entry in plot_entries], dtype=float)
    z3_values = np.array([float(entry["row"].mean_Z3) / Z3_NORMALIZER for entry in plot_entries], dtype=float)

    x = np.arange(len(plot_entries))
    width = 0.24

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x - width, z1_values, width, label=f"Z1 / {Z1_NORMALIZER:g}")
    ax.bar(x, z2_values, width, label=f"Z2 / {Z2_NORMALIZER:g}")
    ax.bar(x + width, z3_values, width, label=f"Z3 / {Z3_NORMALIZER:g}")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FONT_SIZE, pad=8)
    ax.set_xlabel("Policy", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Normalized Value", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title("Normalized Comparison of Z1, Z2, and Z3 Across Policies", fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    ax.legend(fontsize=TICK_LABEL_FONT_SIZE - 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    return output_path


def pairwise_comparison(
    summary_df: pd.DataFrame,
    policy_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    plot_entries = _build_plot_entries(summary_df, policy_tables)
    baseline_entry = next(
        (entry for entry in plot_entries if entry["policy_name"] == BASELINE_POLICY_NAME),
        None,
    )
    if baseline_entry is None:
        raise ValueError(f"Baseline policy {BASELINE_POLICY_NAME} was not found in the evaluation tables.")

    baseline_df = policy_tables[BASELINE_POLICY_NAME][["replication", "H"]].copy()
    baseline_df = baseline_df.rename(columns={"H": "H_baseline"})

    rows: list[dict] = []
    for entry in plot_entries:
        if entry["policy_name"] == BASELINE_POLICY_NAME:
            continue

        compare_df = pd.DataFrame(
            {
                "replication": policy_tables[entry["policy_name"]]["replication"].to_numpy(dtype=int),
                "H_compare": entry["h_values"],
            }
        )
        paired_df = baseline_df.merge(compare_df, on="replication", how="inner")
        diff_values = paired_df["H_compare"].to_numpy(dtype=float) - paired_df["H_baseline"].to_numpy(dtype=float)

        n = int(len(diff_values))
        mean_diff = float(diff_values.mean())
        std_diff = float(diff_values.std(ddof=1)) if n > 1 else 0.0
        se_diff = std_diff / np.sqrt(n) if n > 0 else 0.0
        ci_half_width = Z_VALUE_95 * se_diff
        ci_lower = mean_diff - ci_half_width
        ci_upper = mean_diff + ci_half_width

        rows.append(
            {
                "baseline_policy": baseline_entry["label"],
                "compared_policy": entry["label"],
                "n_replications": n,
                "mean_difference_H": mean_diff,
                "std_difference_H": std_diff,
                "SE_difference_H": se_diff,
                "CI95_lower_difference_H": ci_lower,
                "CI95_upper_difference_H": ci_upper,
                "CI95_width_difference_H": 2.0 * ci_half_width,
                "zero_in_CI": bool(ci_lower <= 0.0 <= ci_upper),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    policy_tables = _load_eval_tables()
    summary_df = _build_summary_table(policy_tables)
    summary_path = _save_summary_table(summary_df)
    box_plot_path = _save_box_plot(summary_df, policy_tables)
    mean_ci_plot_path = _save_mean_ci_plot(summary_df, policy_tables)
    normalized_z_plot_path = _save_normalized_z_bar_plot(summary_df, policy_tables)
    diff_df = pairwise_comparison(summary_df, policy_tables)
    diff_path = _ensure_output_dir() / "pairwise_comparison.csv"
    diff_df.to_csv(diff_path, index=False)

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

    display_diff_df = diff_df.copy()
    for col in [
        "mean_difference_H",
        "std_difference_H",
        "SE_difference_H",
        "CI95_lower_difference_H",
        "CI95_upper_difference_H",
        "CI95_width_difference_H",
    ]:
        display_diff_df[col] = display_diff_df[col].round(4)

    print("H Summary From evaluate_policy Folder:")
    print(display_df.to_string(index=False))
    print("\nPaired H Difference Versus SAA - R1 (other policy minus SAA - R1):")
    print(display_diff_df.to_string(index=False))
    print(f"\nSaved summary table to {summary_path}")
    print(f"Saved pairwise comparison table to {diff_path}")
    print(f"Saved H box plot to {box_plot_path}")
    print(f"Saved H mean/CI plot to {mean_ci_plot_path}")
    print(f"Saved normalized Z bar plot to {normalized_z_plot_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
