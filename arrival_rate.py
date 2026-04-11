import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import gamma, lognorm, chi2


# =========================================================
# DATA PREPROCESSING
# =========================================================
def preprocess_arrival_data(filepath="df_selected.xlsx"):
    df = pd.read_excel(filepath)
    df = df.copy()
    df["Ordered"] = pd.to_datetime(df["Ordered"])
    df["ScanStartF"] = pd.to_datetime(df["ScanStartF"])
    df["ScanStopF"] = pd.to_datetime(df["ScanStopF"])
    df["hour"] = df["Ordered"].dt.hour
    df["arrival_day"] = df["Ordered"].dt.floor("D")

    df = df[df["arrival_day"].dt.weekday < 5].copy()
    df["Procedure_duration_hours"] = (
        (df["ScanStopF"] - df["ScanStartF"]).dt.total_seconds() / 3600
    )
    df = df[df["Procedure_duration_hours"] < 6].copy()

    df["arrival_day"] = pd.to_datetime(df["arrival_day"])
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["classification_norm"] = df["classification"].astype(str).str.strip().str.lower()
    return df


# =========================================================
# SHARED BIN / COUNT HELPERS
# =========================================================
def make_bin_labels(start_hour=8, end_hour=17):
    bin_edges = list(range(start_hour, end_hour + 1, 1))
    bin_labels = [f"{h}:00-{h+1}:00" for h in range(start_hour, end_hour)]
    return bin_edges, bin_labels


def add_time_bins(df, start_hour=8, end_hour=17):
    bin_edges, bin_labels = make_bin_labels(start_hour, end_hour)

    df_selected = df[(df["hour"] >= start_hour) & (df["hour"] < end_hour)].copy()
    df_selected["time_bin"] = pd.cut(
        df_selected["hour"],
        bins=bin_edges,
        right=False,
        labels=bin_labels,
    )
    return df_selected, bin_labels


def get_all_weekdays(df):
    all_days = pd.date_range(
        start=df["arrival_day"].min(),
        end=df["arrival_day"].max(),
        freq="D",
    )
    return all_days[all_days.dayofweek < 5]


def build_counts_by_day_bin(df_selected, classification, all_weekdays, bin_labels):
    classification = classification.strip().lower()
    subset = df_selected[df_selected["classification_norm"] == classification].copy()

    counts_by_day_bin = (
        subset.groupby(["arrival_day", "time_bin"], observed=False)
        .size()
        .unstack(fill_value=0)
    )

    counts_by_day_bin.index = pd.to_datetime(counts_by_day_bin.index)
    counts_by_day_bin = counts_by_day_bin.reindex(index=all_weekdays, fill_value=0)
    counts_by_day_bin = counts_by_day_bin.reindex(columns=bin_labels, fill_value=0)
    counts_by_day_bin.index.name = "arrival_day"
    return counts_by_day_bin


# =========================================================
# NHPP / POISSON DIAGNOSTICS
# =========================================================
def poisson_cumulative_check(counts_by_day_bin):
    cumulative_counts = counts_by_day_bin.cumsum(axis=1)
    cumulative_counts.columns = [
        f"up_to_bin_{i+1}" for i in range(cumulative_counts.shape[1])
    ]

    lambda_bar = cumulative_counts.mean(axis=0)
    v_t = cumulative_counts.var(axis=0, ddof=1)
    ratio = np.where(lambda_bar > 0, v_t / lambda_bar, np.nan)

    result = pd.DataFrame({
        "t": cumulative_counts.columns,
        "Lambda_bar_t": lambda_bar.values,
        "V_t": v_t.values,
        "V_over_Lambda": ratio,
    })
    avg_ratio = np.nanmean(ratio)
    return result, avg_ratio, cumulative_counts


def bin_variance_mean_check(counts_by_day_bin):
    mean_bin = counts_by_day_bin.mean(axis=0)
    var_bin = counts_by_day_bin.var(axis=0, ddof=1)
    ratio_bin = np.where(mean_bin > 0, var_bin / mean_bin, np.nan)

    return pd.DataFrame({
        "time_bin": counts_by_day_bin.columns,
        "mean_bin": mean_bin.values,
        "var_bin": var_bin.values,
        "var_over_mean_bin": ratio_bin,
    })


def fit_nhpp_parameters(counts_by_day_bin):
    lambda_hat = counts_by_day_bin.mean(axis=0)
    cumulative_check, avg_ratio, cumulative_counts = poisson_cumulative_check(counts_by_day_bin)
    bin_check = bin_variance_mean_check(counts_by_day_bin)

    return {
        "model": "NHPP",
        "counts_by_day_bin": counts_by_day_bin,
        "lambda_hat": lambda_hat,
        "cumulative_check": cumulative_check,
        "avg_cumulative_ratio": avg_ratio,
        "cumulative_counts": cumulative_counts,
        "bin_check": bin_check,
    }


def assess_nhpp_by_classification(
    df_no_weekend,
    classifications,
    start_hour=8,
    end_hour=17,
    make_plots=True,
    plot_dir="arrival_rate_plot",
):
    df_selected, bin_labels = add_time_bins(df_no_weekend, start_hour, end_hour)
    all_weekdays = get_all_weekdays(df_no_weekend)

    results = {}
    for classification in classifications:
        counts_by_day_bin = build_counts_by_day_bin(
            df_selected=df_selected,
            classification=classification,
            all_weekdays=all_weekdays,
            bin_labels=bin_labels,
        )
        nhpp_result = fit_nhpp_parameters(counts_by_day_bin)
        results[classification] = nhpp_result

        if make_plots:
            plot_nhpp_diagnostics(classification, nhpp_result, plot_dir=plot_dir)

    return results


# =========================================================
# ANGIOGRAPHY NON-POISSON MODEL
# =========================================================
def fit_zero_inflated_gamma_theta(theta_hat):
    p_zero = np.mean(theta_hat == 0)
    theta_pos = theta_hat[theta_hat > 0].values

    a_raw, _, b_raw = gamma.fit(theta_pos, floc=0)
    mean_pos_raw = a_raw * b_raw
    target_mean_pos = 1.0 / (1.0 - p_zero)
    scale_factor = target_mean_pos / mean_pos_raw
    theta_pos_scaled = theta_pos * scale_factor

    a, _, b = gamma.fit(theta_pos_scaled, floc=0)
    mean_pos = a * b
    var_pos = a * (b ** 2)

    e_theta = (1 - p_zero) * mean_pos
    e_theta2 = (1 - p_zero) * (var_pos + mean_pos ** 2)
    var_theta = e_theta2 - e_theta ** 2

    return {
        "model": "Poisson-Gamma",
        "p_zero": p_zero,
        "shape": a,
        "scale": b,
        "E_theta": e_theta,
        "Var_theta": var_theta,
        "theta_scaled": theta_pos_scaled,
    }


def fit_zero_inflated_lognormal_theta(theta_hat):
    p_zero = np.mean(theta_hat == 0)
    theta_pos = theta_hat[theta_hat > 0].values

    s_raw, _, scale_raw = lognorm.fit(theta_pos, floc=0)
    mean_pos_raw = np.exp(np.log(scale_raw) + 0.5 * s_raw ** 2)
    target_mean_pos = 1.0 / (1.0 - p_zero)
    scale_factor = target_mean_pos / mean_pos_raw
    theta_pos_scaled = theta_pos * scale_factor

    s, _, scale = lognorm.fit(theta_pos_scaled, floc=0)
    mu = np.log(scale)
    sigma = s

    mean_pos = np.exp(mu + 0.5 * sigma ** 2)
    var_pos = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)

    e_theta = (1 - p_zero) * mean_pos
    e_theta2 = (1 - p_zero) * (var_pos + mean_pos ** 2)
    var_theta = e_theta2 - e_theta ** 2

    return {
        "model": "Poisson-Lognormal",
        "p_zero": p_zero,
        "mu": mu,
        "sigma": sigma,
        "E_theta": e_theta,
        "Var_theta": var_theta,
        "theta_scaled": theta_pos_scaled,
    }


def implied_moments_shared_theta(lambda_hat, e_theta, var_theta):
    implied_mean = lambda_hat.values * e_theta
    implied_var = lambda_hat.values * e_theta + (lambda_hat.values ** 2) * var_theta
    implied_ratio = np.where(implied_mean > 0, implied_var / implied_mean, np.nan)

    return pd.DataFrame({
        "time_bin": lambda_hat.index,
        "implied_mean": implied_mean,
        "implied_var": implied_var,
        "implied_ratio": implied_ratio,
    })


def simulate_poisson_gamma_days(n_days, lambda_hat, fit, random_state=None):
    rng = np.random.default_rng(random_state)
    out = np.zeros((n_days, len(lambda_hat)), dtype=int)

    for d in range(n_days):
        theta = 0.0 if rng.uniform() < fit["p_zero"] else rng.gamma(shape=fit["shape"], scale=fit["scale"])
        out[d, :] = rng.poisson(theta * lambda_hat.values)

    return pd.DataFrame(out, columns=lambda_hat.index)


def simulate_poisson_lognormal_days(n_days, lambda_hat, fit, random_state=None):
    rng = np.random.default_rng(random_state)
    out = np.zeros((n_days, len(lambda_hat)), dtype=int)

    for d in range(n_days):
        theta = 0.0 if rng.uniform() < fit["p_zero"] else rng.lognormal(mean=fit["mu"], sigma=fit["sigma"])
        out[d, :] = rng.poisson(theta * lambda_hat.values)

    return pd.DataFrame(out, columns=lambda_hat.index)


def chi_square_gof_from_simulation(observed_counts, simulated_counts, max_count=6):
    obs = np.asarray(observed_counts)
    sim = np.asarray(simulated_counts)

    def tabulate(arr):
        out = [np.sum(arr == k) for k in range(max_count)]
        out.append(np.sum(arr >= max_count))
        return np.array(out, dtype=float)

    obs_freq = tabulate(obs)
    sim_freq = tabulate(sim)
    sim_prob = sim_freq / sim_freq.sum()
    expected = sim_prob * obs_freq.sum()

    mask = expected >= 5
    stat = ((obs_freq[mask] - expected[mask]) ** 2 / expected[mask]).sum()
    df = mask.sum() - 1
    p_value = 1 - chi2.cdf(stat, df)

    return {
        "stat": stat,
        "df": df,
        "p_value": p_value,
        "obs_freq": obs_freq,
        "expected": expected,
    }


def run_gof_all_bins(observed_df, simulated_df, model_name, max_count=6):
    rows = []
    for col in observed_df.columns:
        res = chi_square_gof_from_simulation(
            observed_counts=observed_df[col].values,
            simulated_counts=simulated_df[col].values,
            max_count=max_count,
        )
        rows.append({
            "model": model_name,
            "time_bin": col,
            "chi2_stat": res["stat"],
            "df": res["df"],
            "p_value": res["p_value"],
        })
    return pd.DataFrame(rows)


def fit_arrival_models(
    df_no_weekend,
    target_class="angiography",
    start_hour=8,
    end_hour=17,
    n_sim_days=None,
    random_state=123,
    make_plots=True,
    plot_dir="arrival_rate_plot",
):
    df_selected, bin_labels = add_time_bins(df_no_weekend, start_hour, end_hour)
    all_weekdays = get_all_weekdays(df_no_weekend)
    counts_by_day_bin = build_counts_by_day_bin(
        df_selected=df_selected,
        classification=target_class,
        all_weekdays=all_weekdays,
        bin_labels=bin_labels,
    )

    obs_mean = counts_by_day_bin.mean(axis=0)
    obs_var = counts_by_day_bin.var(axis=0, ddof=1)
    obs_ratio = np.where(obs_mean > 0, obs_var / obs_mean, np.nan)

    lambda_hat = obs_mean.copy()
    daily_totals = counts_by_day_bin.sum(axis=1)
    theta_hat = daily_totals / lambda_hat.sum()

    summary_obs = pd.DataFrame({
        "time_bin": obs_mean.index,
        "obs_mean": obs_mean.values,
        "obs_var": obs_var.values,
        "obs_ratio": obs_ratio,
    })

    pg_fit = fit_zero_inflated_gamma_theta(theta_hat)
    pln_fit = fit_zero_inflated_lognormal_theta(theta_hat)

    pg_mom = implied_moments_shared_theta(lambda_hat, pg_fit["E_theta"], pg_fit["Var_theta"]).rename(
        columns={
            "implied_mean": "pg_mean",
            "implied_var": "pg_var",
            "implied_ratio": "pg_ratio",
        }
    )
    pln_mom = implied_moments_shared_theta(lambda_hat, pln_fit["E_theta"], pln_fit["Var_theta"]).rename(
        columns={
            "implied_mean": "pln_mean",
            "implied_var": "pln_var",
            "implied_ratio": "pln_ratio",
        }
    )

    comparison = pd.DataFrame({
        "time_bin": obs_mean.index,
        "obs_mean": obs_mean.values,
        "obs_var": obs_var.values,
        "obs_ratio": obs_ratio,
    }).merge(pg_mom, on="time_bin", how="left").merge(pln_mom, on="time_bin", how="left")

    n_obs_days = len(counts_by_day_bin)
    n_sim_days = max(5000, 50 * n_obs_days) if n_sim_days is None else n_sim_days

    sim_pg = simulate_poisson_gamma_days(n_days=n_sim_days, lambda_hat=lambda_hat, fit=pg_fit, random_state=random_state)
    sim_pln = simulate_poisson_lognormal_days(n_days=n_sim_days, lambda_hat=lambda_hat, fit=pln_fit, random_state=random_state)

    gof_pg = run_gof_all_bins(counts_by_day_bin, sim_pg, "Poisson-Gamma", max_count=6)
    gof_pln = run_gof_all_bins(counts_by_day_bin, sim_pln, "Poisson-Lognormal", max_count=6)
    gof_all = pd.concat([gof_pg, gof_pln], ignore_index=True)

    gof_summary = (
        gof_all.groupby("model")
        .agg(
            mean_p_value=("p_value", "mean"),
            median_p_value=("p_value", "median"),
            num_bins_p_gt_005=("p_value", lambda x: np.sum(x > 0.05)),
            num_bins_tested=("p_value", "count"),
        )
        .reset_index()
    )

    results = {
        "counts_by_day_bin": counts_by_day_bin,
        "lambda_hat": lambda_hat,
        "theta_hat": theta_hat,
        "summary_obs": summary_obs,
        "pg_fit": pg_fit,
        "pln_fit": pln_fit,
        "comparison": comparison,
        "sim_pg": sim_pg,
        "sim_pln": sim_pln,
        "gof_all": gof_all,
        "gof_summary": gof_summary,
    }

    if make_plots:
        plot_nonpoisson_diagnostics(target_class, results, plot_dir=plot_dir)

    return results


# =========================================================
# HIGH-LEVEL PIPELINE
# =========================================================
def analyze_arrival_rates(
    df_no_weekend,
    poisson_classes=("interventional", "angiography"),
    angiography_class="angiography",
    start_hour=8,
    end_hour=17,
    random_state=123,
    make_plots=True,
    plot_dir="arrival_rate_plot",
):
    poisson_results = assess_nhpp_by_classification(
        df_no_weekend=df_no_weekend,
        classifications=poisson_classes,
        start_hour=start_hour,
        end_hour=end_hour,
        make_plots=make_plots,
        plot_dir=plot_dir,
    )

    angiography_results = fit_arrival_models(
        df_no_weekend=df_no_weekend,
        target_class=angiography_class,
        start_hour=start_hour,
        end_hour=end_hour,
        random_state=random_state,
        make_plots=make_plots,
        plot_dir=plot_dir,
    )

    poisson_parameters = {
        classification: result["lambda_hat"]
        for classification, result in poisson_results.items()
    }

    return {
        "poisson_parameters": poisson_parameters,
        "poisson_results": poisson_results,
        "angiography_model": angiography_results,
    }


# =========================================================
# PLOTTING
# =========================================================
def _sanitize_filename(text):
    return (
        str(text).strip().lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "")
    )


def _save_plot(fig, plot_dir, filename):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / filename
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_nhpp_diagnostics(classification, result, plot_dir="arrival_rate_plot"):
    class_tag = _sanitize_filename(classification)

    counts_by_day_bin = result["counts_by_day_bin"]
    mean_counts = counts_by_day_bin.mean(axis=0)
    cumulative_check = result["cumulative_check"]
    bin_check = result["bin_check"]
    days_in_each_bin = (counts_by_day_bin > 0).sum(axis=0)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(mean_counts.index.astype(str), mean_counts.values, marker="o")
    plt.title(f"Mean Arrivals by Time Bin - {classification.title()}")
    plt.xlabel("Time Bin")
    plt.ylabel("Mean Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save_plot(fig, plot_dir, f"{class_tag}_mean_arrivals_by_time_bin.png")

    fig = plt.figure(figsize=(10, 6))
    plt.plot(cumulative_check["t"], cumulative_check["V_over_Lambda"], marker="o")
    plt.axhline(y=1.0, linestyle="--")
    plt.title(f"Cumulative Variance-to-Mean Ratio - {classification.title()}")
    plt.xlabel("Cumulative Time Bin")
    plt.ylabel("V(t) / Lambda_bar(t)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save_plot(fig, plot_dir, f"{class_tag}_cumulative_variance_to_mean_ratio.png")

    fig = plt.figure(figsize=(10, 6))
    plt.bar(days_in_each_bin.index.astype(str), days_in_each_bin.values, edgecolor="black")
    plt.title(f"Days with At Least One Arrival by Time Bin - {classification.title()}")
    plt.xlabel("Time Bin")
    plt.ylabel("Number of Days")
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save_plot(fig, plot_dir, f"{class_tag}_days_with_arrivals_by_time_bin.png")

    # fig = plt.figure(figsize=(10, 6))
    # plt.plot(bin_check["time_bin"].astype(str), bin_check["var_over_mean_bin"], marker="o")
    # plt.axhline(y=1.0, linestyle="--")
    # plt.title(f"Bin-Level Variance-to-Mean Ratio - {classification.title()}")
    # plt.xlabel("Time Bin")
    # plt.ylabel("Variance / Mean")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # _save_plot(fig, plot_dir, f"{class_tag}_bin_level_variance_to_mean_ratio.png")


def plot_nonpoisson_diagnostics(classification, result, plot_dir="arrival_rate_plot"):
    class_tag = _sanitize_filename(classification)
    comparison = result["comparison"]

    fig = plt.figure(figsize=(11, 5))
    plt.plot(comparison["time_bin"], comparison["obs_ratio"], marker="o", label="Observed")
    plt.plot(comparison["time_bin"], comparison["pg_ratio"], marker="s", label="Poisson-Gamma")
    plt.plot(comparison["time_bin"], comparison["pln_ratio"], marker="^", label="Poisson-Lognormal")
    plt.axhline(1.0, linestyle="--")
    plt.title(f"Variance-to-Mean Ratio by Bin - {classification.title()}")
    plt.xlabel("Time Bin")
    plt.ylabel("Variance / Mean")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    _save_plot(fig, plot_dir, f"{class_tag}_nonpoisson_variance_to_mean_comparison.png")


# =========================================================
# RUN DIRECTLY
# =========================================================
if __name__ == "__main__":
    plot_dir = "arrival_rate_plot"

    df_no_weekend = preprocess_arrival_data("df_selected.xlsx")

    results = analyze_arrival_rates(
        df_no_weekend=df_no_weekend,
        poisson_classes=("interventional", "angiography"),
        angiography_class="angiography",
        start_hour=8,
        end_hour=17,
        random_state=123,
        make_plots=True,
        plot_dir=plot_dir,
    )

    poisson_parameters = results["poisson_parameters"]
    angiography_model = results["angiography_model"]

    arrival_params = {
        "interventional_nhpp": {
            "model": "NHPP",
            "start_hour": 8,
            "end_hour": 17,
            "lambda_hat": {k: float(v) for k, v in poisson_parameters["interventional"].to_dict().items()}
        },
        "angiography_pln": {
            "model": "Poisson-Lognormal",
            "start_hour": 8,
            "end_hour": 17,
            "lambda_hat": {k: float(v) for k, v in angiography_model["lambda_hat"].to_dict().items()},
            "pln_fit": {
                "p_zero": float(angiography_model["pln_fit"]["p_zero"]),
                "mu": float(angiography_model["pln_fit"]["mu"]),
                "sigma": float(angiography_model["pln_fit"]["sigma"])
            }
        }
    }

    with open("arrival_model_params.json", "w") as f:
        json.dump(arrival_params, f, indent=4)
