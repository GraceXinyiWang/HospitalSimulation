import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from arrival_rate import preprocess_arrival_data


INPUT_EXCEL_PATH = "df_selected.xlsx"
OUTPUT_JSON_PATH = "services rate.json"
PLOT_DIR = "serivce_rate_plot"
ALPHA = 0.05
BINS = 30
SHOW_PLOTS = False
SAVE_PLOTS = True


def sanitize_filename(text):
    text = str(text).strip().replace("|", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "plot"


def ensure_plot_dir(plot_dir):
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)
    return plot_path


def pretty_duration_label(duration_col, time_unit=""):
    label_map = {
        "Procedure_duration_hours": "Procedure Duration",
        "Preparation_duration_days": "Preparation Duration",
        "LateTime_minutes": "Late Time",
    }
    base = label_map.get(duration_col, duration_col.replace("_", " ").strip().title())
    return f"{base} ({time_unit})" if time_unit else base


def pretty_group_label(group_key):
    if group_key == "All":
        return "All Cases"

    parts = []
    for item in str(group_key).split("|"):
        item = item.strip()
        if "=" in item:
            key, value = item.split("=", 1)
            key = key.strip().replace("_", " ").title()
            value = value.strip().replace("_", " ").title()
            parts.append(f"{key}: {value}")
        else:
            parts.append(item.replace("_", " ").title())
    return " | ".join(parts)


def fit_service_time_distribution(
    df,
    duration_col,
    group_cols=None,
    max_value=None,
    min_positive=True,
    bins=30,
    alpha=0.05,
    show_plots=False,
    save_plots=True,
    plot_dir=None,
    plot_prefix=None,
    time_unit="",
):
    """
    Fit Exponential, Gamma, Weibull, and Lognormal distributions to one duration column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    duration_col : str
        Duration column to fit.
    group_cols : None, str, or list[str]
        Grouping columns. If None, fit on all rows together.
    max_value : float or None
        Remove observations above this threshold.
    min_positive : bool
        Keep only values > 0 when True.
    bins : int
        Number of histogram bins for plots.
    alpha : float
        Significance level used for the KS decision.
    show_plots : bool
        Whether to display plots on screen.
    save_plots : bool
        Whether to save plots to disk.
    plot_dir : str or Path or None
        Folder used for saved plots.
    plot_prefix : str or None
        Prefix added to saved plot filenames.
    time_unit : str
        Unit label used in printed summaries and the returned dataframe.

    Returns
    -------
    best_results_df : pandas.DataFrame
        Best distribution for each group after sorting by KS_D then KS_p_value.
    all_results_df : pandas.DataFrame
        All fitted distribution results for all groups.
    cleaned_df : pandas.DataFrame
        Dataframe after cleaning rules are applied.
    outliers_df : pandas.DataFrame
        Rows removed because duration > max_value.
    fit_params_dict : dict
        Fitted parameter dictionary for every group and every candidate distribution.
    """
    df_work = df.copy()

    if max_value is not None:
        outliers_df = df_work[df_work[duration_col] > max_value].copy()
    else:
        outliers_df = df_work.iloc[0:0].copy()

    mask = df_work[duration_col].notna()

    if min_positive:
        mask &= df_work[duration_col] > 0

    if max_value is not None:
        mask &= df_work[duration_col] <= max_value

    df_work = df_work.loc[mask].copy()

    print(f"Number kept: {len(df_work)}")
    print(f"Number dropped as outliers: {len(outliers_df)}")

    if group_cols is None:
        grouping_cols = []
        groups = [("All", df_work)]
    else:
        grouping_cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
        groups = list(df_work.groupby(grouping_cols, dropna=False))

    best_results = []
    all_results = []
    fit_params_dict = {}

    for group_name, group_df in groups:
        if not isinstance(group_name, tuple):
            group_name = (group_name,)

        if grouping_cols:
            group_key = " | ".join(
                f"{col}={val}" for col, val in zip(grouping_cols, group_name)
            )
        else:
            group_key = "All"

        service_data = group_df[duration_col].dropna().astype(float)

        if len(service_data) < 5:
            print(f"\n{group_key}: not enough data to fit.")
            continue

        print("\n" + "=" * 90)
        print(f"Group: {group_key}")
        print(f"n = {len(service_data)}")
        print(f"Mean duration ({time_unit}): {service_data.mean():.4f}")
        print(f"Std duration ({time_unit}): {service_data.std():.4f}")
        print(f"Min duration ({time_unit}): {service_data.min():.4f}")
        print(f"Max duration ({time_unit}): {service_data.max():.4f}")

        mean_duration = service_data.mean()
        inverse_mean_rate = 1 / mean_duration
        print(f"Average service rate (1/{time_unit}): {inverse_mean_rate:.4f}")

        exp_loc, exp_scale = stats.expon.fit(service_data, floc=0)
        gamma_a, gamma_loc, gamma_scale = stats.gamma.fit(service_data, floc=0)
        weibull_c, weibull_loc, weibull_scale = stats.weibull_min.fit(service_data, floc=0)
        lognorm_sigma, lognorm_loc, lognorm_scale = stats.lognorm.fit(service_data, floc=0)

        candidate_specs = [
            ("Exponential", "expon", (exp_loc, exp_scale)),
            ("Gamma", "gamma", (gamma_a, gamma_loc, gamma_scale)),
            ("Weibull", "weibull_min", (weibull_c, weibull_loc, weibull_scale)),
            ("Lognormal", "lognorm", (lognorm_sigma, lognorm_loc, lognorm_scale)),
        ]

        fit_params_dict[group_key] = {
            "Exponential": {"loc": float(exp_loc), "scale": float(exp_scale)},
            "Gamma": {"shape": float(gamma_a), "loc": float(gamma_loc), "scale": float(gamma_scale)},
            "Weibull": {"shape": float(weibull_c), "loc": float(weibull_loc), "scale": float(weibull_scale)},
            "Lognormal": {"sigma": float(lognorm_sigma), "loc": float(lognorm_loc), "scale": float(lognorm_scale)},
        }

        rows = []
        for dist_name, scipy_name, params in candidate_specs:
            D, p_value = stats.kstest(service_data, scipy_name, args=params)
            decision = "Fail to reject" if p_value >= alpha else "Reject"

            row = {
                "group": group_key,
                "distribution": dist_name,
                "KS_D": float(D),
                "KS_p_value": float(p_value),
                "decision": decision,
                "n": int(len(service_data)),
                "mean_value": float(mean_duration),
                "inverse_mean_rate": float(inverse_mean_rate),
                "time_unit": time_unit,
            }

            if dist_name == "Exponential":
                row.update({
                    "param_1_name": "loc",
                    "param_1_value": float(exp_loc),
                    "param_2_name": "scale",
                    "param_2_value": float(exp_scale),
                    "param_3_name": None,
                    "param_3_value": None,
                })
            elif dist_name == "Gamma":
                row.update({
                    "param_1_name": "shape",
                    "param_1_value": float(gamma_a),
                    "param_2_name": "loc",
                    "param_2_value": float(gamma_loc),
                    "param_3_name": "scale",
                    "param_3_value": float(gamma_scale),
                })
            elif dist_name == "Weibull":
                row.update({
                    "param_1_name": "shape",
                    "param_1_value": float(weibull_c),
                    "param_2_name": "loc",
                    "param_2_value": float(weibull_loc),
                    "param_3_name": "scale",
                    "param_3_value": float(weibull_scale),
                })
            else:
                row.update({
                    "param_1_name": "sigma",
                    "param_1_value": float(lognorm_sigma),
                    "param_2_name": "loc",
                    "param_2_value": float(lognorm_loc),
                    "param_3_name": "scale",
                    "param_3_value": float(lognorm_scale),
                })

            rows.append(row)

        ks_df = pd.DataFrame(rows).sort_values(
            ["KS_D", "KS_p_value"], ascending=[True, False]
        ).reset_index(drop=True)

        print("\nKS test results:")
        print(ks_df[["distribution", "KS_D", "KS_p_value", "decision"]])

        best_results.append(ks_df.iloc[0].copy())
        all_results.append(ks_df)

        if show_plots or save_plots:
            plot_fitted_distributions(
                service_data=service_data,
                group_key=group_key,
                duration_col=duration_col,
                time_unit=time_unit,
                bins=bins,
                exp_params=(exp_loc, exp_scale),
                gamma_params=(gamma_a, gamma_loc, gamma_scale),
                weibull_params=(weibull_c, weibull_loc, weibull_scale),
                lognorm_params=(lognorm_sigma, lognorm_loc, lognorm_scale),
                show_plots=show_plots,
                save_plots=save_plots,
                plot_dir=plot_dir,
                plot_prefix=plot_prefix,
            )

    best_results_df = pd.DataFrame(best_results).reset_index(drop=True)
    all_results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    return best_results_df, all_results_df, df_work, outliers_df, fit_params_dict



def save_or_show_figure(fig, output_path, show_plots):
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)



def plot_fitted_distributions(
    service_data,
    group_key,
    duration_col,
    time_unit,
    bins,
    exp_params,
    gamma_params,
    weibull_params,
    lognorm_params,
    show_plots=False,
    save_plots=True,
    plot_dir=None,
    plot_prefix=None,
):
    base_name = sanitize_filename(f"{plot_prefix}_{group_key}" if plot_prefix else group_key)
    plot_path = ensure_plot_dir(plot_dir) if save_plots and plot_dir is not None else None
    measure_label = pretty_duration_label(duration_col, time_unit)
    group_label = pretty_group_label(group_key)

    fig = plt.figure(figsize=(12, 8))
    plt.hist(service_data, bins=bins, density=True, alpha=0.5, label="Observed data")

    x = np.linspace(service_data.min(), service_data.max(), 300)
    plt.plot(x, stats.expon.pdf(x, *exp_params), label="Exponential")
    plt.plot(x, stats.gamma.pdf(x, *gamma_params), label="Gamma")
    plt.plot(x, stats.weibull_min.pdf(x, *weibull_params), label="Weibull")
    plt.plot(x, stats.lognorm.pdf(x, *lognorm_params), label="Lognormal")

    plt.title(f"{measure_label} Distribution with Fitted Curves\n{group_label}")
    plt.xlabel(measure_label)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    fit_path = plot_path / f"{base_name}_fitted_distributions.png" if plot_path is not None else None
    save_or_show_figure(fig, fit_path, show_plots)

    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    sorted_data = np.sort(service_data)
    prob = (np.arange(1, len(service_data) + 1) - 0.5) / len(service_data)

    theoretical_exp = stats.expon.ppf(prob, *exp_params)
    axs[0].plot(theoretical_exp, sorted_data, "o", markersize=4)
    axs[0].plot(theoretical_exp, theoretical_exp, "r--")
    axs[0].set_title(f"Exponential Q-Q Plot\n{group_label}")

    theoretical_gamma = stats.gamma.ppf(prob, *gamma_params)
    axs[1].plot(theoretical_gamma, sorted_data, "o", markersize=4)
    axs[1].plot(theoretical_gamma, theoretical_gamma, "r--")
    axs[1].set_title(f"Gamma Q-Q Plot\n{group_label}")

    theoretical_weibull = stats.weibull_min.ppf(prob, *weibull_params)
    axs[2].plot(theoretical_weibull, sorted_data, "o", markersize=4)
    axs[2].plot(theoretical_weibull, theoretical_weibull, "r--")
    axs[2].set_title(f"Weibull Q-Q Plot\n{group_label}")

    theoretical_lognorm = stats.lognorm.ppf(prob, *lognorm_params)
    axs[3].plot(theoretical_lognorm, sorted_data, "o", markersize=4)
    axs[3].plot(theoretical_lognorm, theoretical_lognorm, "r--")
    axs[3].set_title(f"Lognormal Q-Q Plot\n{group_label}")

    for ax in axs:
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Empirical Quantiles")

    fig.suptitle(f"{measure_label} Q-Q Comparison", y=1.02, fontsize=14)
    plt.tight_layout()
    qq_path = plot_path / f"{base_name}_qq.png" if plot_path is not None else None
    save_or_show_figure(fig, qq_path, show_plots)



def plot_preparation_percentages(category_percentage_table, plot_dir, show_plots=False, save_plots=True):
    if category_percentage_table.empty:
        return

    plot_path = ensure_plot_dir(plot_dir) if save_plots and plot_dir is not None else None

    fig = plt.figure(figsize=(10, 6))
    category_percentage_table.plot(kind="bar", stacked=True)
    plt.title("Preparation Category Percentages by Classification")
    plt.xlabel("Classification")
    plt.ylabel("Percentage")
    plt.legend(title="category_prepared", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    output_path = plot_path / "preparation_category_percentages.png" if plot_path is not None else None
    save_or_show_figure(fig, output_path, show_plots)



def row_to_fit_payload(row):
    payload = {
        "distribution": row["distribution"],
        "parameters": {},
        "KS_D": float(row["KS_D"]),
        "KS_p_value": float(row["KS_p_value"]),
        "decision": row["decision"],
        "n": int(row["n"]),
        "mean_value": float(row["mean_value"]),
        "inverse_mean_rate": float(row["inverse_mean_rate"]),
        "time_unit": row["time_unit"],
    }

    payload["parameters"][row["param_1_name"]] = float(row["param_1_value"])
    payload["parameters"][row["param_2_name"]] = float(row["param_2_value"])
    if pd.notna(row["param_3_name"]):
        payload["parameters"][row["param_3_name"]] = float(row["param_3_value"])

    return payload



def build_nested_preparation_fits(best_results_df, accepted_only=True):
    preparation_fits = {}
    empirical_fallback_groups = []

    for _, row in best_results_df.iterrows():
        group_text = row["group"]
        parts = [p.strip() for p in group_text.split("|")]
        parsed = {}
        for part in parts:
            key, value = part.split("=", 1)
            parsed[key.strip()] = value.strip()

        classification = parsed["classification"]
        category_prepared = parsed["category_prepared"]

        if accepted_only and row["decision"] != "Fail to reject":
            empirical_fallback_groups.append({
                "classification": classification,
                "category_prepared": category_prepared,
                "reason": "All fitted candidate distributions were rejected by the KS p-value rule.",
            })
            continue

        preparation_fits.setdefault(classification, {})
        preparation_fits[classification][category_prepared] = row_to_fit_payload(row)

    return preparation_fits, empirical_fallback_groups



def dataframe_to_nested_percentages(df):
    nested = {}
    for classification, row in df.iterrows():
        class_name = classification[0] if isinstance(classification, tuple) else classification
        nested[class_name] = {
            str(col): float(value)
            for col, value in row.to_dict().items()
        }
    return nested



def main():
    ensure_plot_dir(PLOT_DIR)

    # ---------------------------------------------------------
    # Notebook order: read excel, create df_no_weekend
    # ---------------------------------------------------------
    df_no_weekend = preprocess_arrival_data(INPUT_EXCEL_PATH)

    # ---------------------------------------------------------
    # Notebook order: procedure duration and global filter
    # ---------------------------------------------------------
    df_no_weekend["Procedure_duration_hours"] = (
        (df_no_weekend["ScanStopF"] - df_no_weekend["ScanStartF"]).dt.total_seconds() / 3600
    )

    df_no_weekend = df_no_weekend[df_no_weekend["Procedure_duration_hours"] < 6].copy()

    procedure_best_df, procedure_all_df, procedure_cleaned_df, procedure_outliers_df, procedure_fit_params = (
        fit_service_time_distribution(
            df=df_no_weekend,
            duration_col="Procedure_duration_hours",
            group_cols=None,
            max_value=180,
            alpha=ALPHA,
            bins=BINS,
            show_plots=SHOW_PLOTS,
            save_plots=SAVE_PLOTS,
            plot_dir=PLOT_DIR,
            plot_prefix="procedure_duration_hours",
            time_unit="hours",
        )
    )

        

    df_no_weekend["Preparation_duration"] = (
        df_no_weekend["lastBooked"] - df_no_weekend["Ordered"]
    )

    df_no_weekend["Preparation_duration_days"] = (
        df_no_weekend["Preparation_duration"].dt.total_seconds() / 86400
    )
    prep_best_df, prep_all_df, prep_cleaned_df, prep_outliers_df, prep_fit_params = (
        fit_service_time_distribution(
            df=df_no_weekend,
            duration_col="Preparation_duration_days",
            group_cols=["classification", "category_prepared"],
            max_value=180,
            alpha=ALPHA,
            bins=BINS,
            show_plots=SHOW_PLOTS,
            save_plots=SAVE_PLOTS,
            plot_dir=PLOT_DIR,
            plot_prefix="preparation_duration_days",
            time_unit="days",
        )
    )

    category_percentage_table = (
        df_no_weekend.groupby(["classification", "category_prepared"])
        .size()
        .groupby(level=0)
        .apply(lambda x: 100 * x / x.sum())
        .unstack(fill_value=0)
    )
    plot_preparation_percentages(
        category_percentage_table=category_percentage_table,
        plot_dir=PLOT_DIR,
        show_plots=SHOW_PLOTS,
        save_plots=SAVE_PLOTS,
    )

    preparation_fits, empirical_fallback_groups = build_nested_preparation_fits(
        prep_best_df, accepted_only=True
    )

    # ---------------------------------------------------------
    # Notebook order: late time
    # ---------------------------------------------------------
    df_no_weekend["LateTime"] = (
        df_no_weekend["ScanStartF"].dt.hour - df_no_weekend["FinalScheduled"].dt.hour / 60
    )

    df_no_weekend["LateTime_minutes"] = (
        (df_no_weekend["ScanStartF"] - df_no_weekend["FinalScheduled"]).dt.total_seconds() / 60
    )

    df_no_weekend = df_no_weekend[df_no_weekend["LateTime_minutes"] > -10].copy()

    late_best_df, late_all_df, late_cleaned_df, late_outliers_df, late_fit_params = (
        fit_service_time_distribution(
            df=df_no_weekend,
            duration_col="LateTime_minutes",
            group_cols="classification",
            max_value=365,
            alpha=ALPHA,
            bins=BINS,
            show_plots=SHOW_PLOTS,
            save_plots=SAVE_PLOTS,
            plot_dir=PLOT_DIR,
            plot_prefix="late_time_minutes",
            time_unit="minutes",
        )
    )

    output = {
        "source_file": INPUT_EXCEL_PATH,
        "plot_folder": PLOT_DIR,
        "procedure_duration_hours": {
            "All": row_to_fit_payload(procedure_best_df.iloc[0])
        },
        "preparation_duration_days": preparation_fits,
        "preparation_category_percentages": dataframe_to_nested_percentages(category_percentage_table),
        "preparation_empirical_fallback_groups": empirical_fallback_groups,
        "late_time_minutes": {
            row["group"]: row_to_fit_payload(row)
            for _, row in late_best_df.iterrows()
        },
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"\nSaved final selected parameters to: {OUTPUT_JSON_PATH}")
    print(f"Saved plots to folder: {PLOT_DIR}")


if __name__ == "__main__":
    main()
