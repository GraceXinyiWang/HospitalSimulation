import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


##### Data Preprocessing #####
df = pd.read_excel('df_selected.xlsx')
df_valid = df.copy()
df_valid['Ordered'] = pd.to_datetime(df_valid['Ordered'])
df_valid['hour'] = df_valid['Ordered'].dt.hour
df_valid['arrival_day'] = df_valid['Ordered'].dt.floor('D')

df_no_weekend = df_valid[df_valid['arrival_day'].dt.weekday < 5].copy() # Remove Outlier

df_no_weekend['Procedure_duration_hours'] = (
    (df_no_weekend['ScanStopF'] - df_no_weekend['ScanStartF']).dt.total_seconds() / 3600
)

df_no_weekend = df_no_weekend[df_no_weekend['Procedure_duration_hours'] < 6].copy()

########Define the function to calculate arrival rate########
# ==================================================
# Step 0: make sure your dataframe has these columns
# df_no_weekend should contain at least:
#   - arrival_day
#   - hour
#   - classification
# ==================================================



df_no_weekend['arrival_day'] = pd.to_datetime(df_no_weekend['arrival_day'])
df_no_weekend['hour'] = pd.to_numeric(df_no_weekend['hour'], errors='coerce')

# ==================================================
# Step 1: choose time window
# ==================================================
start_hour = 8
end_hour = 17

bin_edges = list(range(start_hour, end_hour + 1, 1))
bin_labels = [f"{h}:00-{h+1}:00" for h in range(start_hour, end_hour, 1)]

# ==================================================
# Step 2: keep only records inside selected hours
# ==================================================
df_selected = df_no_weekend[
    (df_no_weekend['hour'] >= start_hour) &
    (df_no_weekend['hour'] < end_hour)
].copy()

# Assign each arrival to a time bin
df_selected['time_bin'] = pd.cut(
    df_selected['hour'],
    bins=bin_edges,
    right=False,
    labels=bin_labels
)

# ==================================================
# Step 3: build the full set of weekdays
# IMPORTANT:
# We want the same day index even if one day has no arrivals
# inside the chosen time window
# ==================================================
all_days = pd.date_range(
    start=df_no_weekend['arrival_day'].min(),
    end=df_no_weekend['arrival_day'].max(),
    freq='D'
)

# Keep only weekdays if your df_no_weekend is weekday-only logic
all_days = all_days[all_days.dayofweek < 5]

# ==================================================
# Step 4: cumulative Poisson check function
# ==================================================
def poisson_cumulative_check(counts_by_day_bin):
    # cumulative counts across bins for each day
    cumulative_counts = counts_by_day_bin.cumsum(axis=1)

    # rename columns
    cumulative_columns = [f"up_to_bin_{i+1}" for i in range(cumulative_counts.shape[1])]
    cumulative_counts.columns = cumulative_columns

    # mean across days
    lambda_bar = cumulative_counts.mean(axis=0)

    # sample variance across days
    V_t = cumulative_counts.var(axis=0, ddof=1)

    # variance / mean ratio
    ratio = np.where(lambda_bar > 0, V_t / lambda_bar, np.nan)

    result = pd.DataFrame({
        "t": cumulative_counts.columns,
        "Lambda_bar_t": lambda_bar.values,
        "V_t": V_t.values,
        "V_over_Lambda": ratio
    })

    avg_ratio = np.nanmean(ratio)

    return result, avg_ratio, cumulative_counts

# ==================================================
# Step 5: run for each classification
# ==================================================
classifications = sorted(df_no_weekend['classification'].dropna().unique())

for classification in classifications:
    subset = df_selected[df_selected['classification'] == classification].copy()

    # Count arrivals by day and time bin
    counts_by_day_bin = (
        subset.groupby(['arrival_day', 'time_bin'])
        .size()
        .unstack(fill_value=0)
    )

    # make sure index is datetime
    counts_by_day_bin.index = pd.to_datetime(counts_by_day_bin.index)

    # IMPORTANT:
    # reindex to all weekdays so missing days stay as zero rows
    counts_by_day_bin = counts_by_day_bin.reindex(index=all_days, fill_value=0)

    # make sure all bins exist
    counts_by_day_bin = counts_by_day_bin.reindex(columns=bin_labels, fill_value=0)

    # skip if too few days
    if len(counts_by_day_bin) < 5:
        print(f"Skipping {classification}: not enough days ({len(counts_by_day_bin)})")
        continue

    # ----------------------------------------------
    # Plot mean count by bin
    # ----------------------------------------------
    mean_counts = counts_by_day_bin.mean(axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_counts.index.astype(str), mean_counts.values, marker='o')
    plt.title(f"Mean Number of Counts by Time Bin - {classification}")
    plt.xlabel("Time Bin")
    plt.ylabel("Mean Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------
    # Poisson cumulative check
    # ----------------------------------------------
    result, avg_ratio, cumulative_counts = poisson_cumulative_check(counts_by_day_bin)

    print("\n" + "=" * 90)
    print(f"Poisson cumulative check for classification: {classification}")
    print(result)
    print(f"\nAverage cumulative variance / mean ratio = {avg_ratio:.4f}")

    if 0.7 <= avg_ratio <= 1.4:
        print("Interpretation: roughly consistent with a Poisson-process assumption.")
    elif avg_ratio > 1.4:
        print("Interpretation: overdispersion; Poisson may be too simple.")
    else:
        print("Interpretation: underdispersion; arrivals may be more regular than Poisson.")

    # ----------------------------------------------
    # Plot cumulative variance/mean ratio
    # ----------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(result['t'], result['V_over_Lambda'], marker='o')
    plt.axhline(y=1.0, linestyle='--')
    plt.title(f"Cumulative Variance/Mean Ratio - {classification}")
    plt.xlabel("Cumulative Time Bin")
    plt.ylabel("V(t) / Lambda_bar(t)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------
    # Number of days with at least one arrival in each bin
    # ----------------------------------------------
    days_in_each_bin = (counts_by_day_bin > 0).sum(axis=0)

    plt.figure(figsize=(10, 6))
    plt.bar(days_in_each_bin.index.astype(str), days_in_each_bin.values, edgecolor='black')
    plt.title(f"Number of Days with At Least One Arrival in Each Time Bin - {classification}")
    plt.xlabel("Time Bin")
    plt.ylabel("Number of Days")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------
    # Optional: non-cumulative variance/mean by bin
    # ----------------------------------------------
    mean_bin = counts_by_day_bin.mean(axis=0)
    var_bin = counts_by_day_bin.var(axis=0, ddof=1)
    ratio_bin = np.where(mean_bin > 0, var_bin / mean_bin, np.nan)

    bin_result = pd.DataFrame({
        "time_bin": counts_by_day_bin.columns,
        "mean_bin": mean_bin.values,
        "var_bin": var_bin.values,
        "var_over_mean_bin": ratio_bin
    })

    print("\nNon-cumulative bin-by-bin variance/mean check:")
    print(bin_result)

    plt.figure(figsize=(10, 6))
    plt.plot(bin_result["time_bin"].astype(str), bin_result["var_over_mean_bin"], marker='o')
    plt.axhline(y=1.0, linestyle='--')
    plt.title(f"Non-Cumulative Variance/Mean Ratio by Bin - {classification}")
    plt.xlabel("Time Bin")
    plt.ylabel("Var / Mean")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import gamma, lognorm, chi2


# =========================================================
# 1. FIT POISSON-GAMMA DAY EFFECT
#    Theta_d = 0 with prob p_zero
#    Theta_d ~ Gamma(a, scale=b) on positive days
#    Then rescale so E[Theta] = 1
# =========================================================
def fit_zero_inflated_gamma_theta(theta_hat):
    p_zero = np.mean(theta_hat == 0)
    theta_pos = theta_hat[theta_hat > 0].values

    if len(theta_pos) < 2:
        raise ValueError("Not enough positive theta values for Gamma fit.")

    # Raw fit
    a_raw, loc_raw, b_raw = gamma.fit(theta_pos, floc=0)

    # Rescale positive part so unconditional E[Theta] = 1
    mean_pos_raw = a_raw * b_raw
    target_mean_pos = 1.0 / (1.0 - p_zero) if p_zero < 1 else np.nan
    scale_factor = target_mean_pos / mean_pos_raw

    theta_pos_scaled = theta_pos * scale_factor

    a, loc, b = gamma.fit(theta_pos_scaled, floc=0)

    mean_pos = a * b
    var_pos = a * (b ** 2)

    E_theta = (1 - p_zero) * mean_pos
    E_theta2 = (1 - p_zero) * (var_pos + mean_pos**2)
    Var_theta = E_theta2 - E_theta**2

    return {
        "model": "Poisson-Gamma",
        "p_zero": p_zero,
        "shape": a,
        "scale": b,
        "E_theta": E_theta,
        "Var_theta": Var_theta,
        "theta_scaled": theta_pos_scaled
    }


# =========================================================
# 2. FIT POISSON-LOGNORMAL DAY EFFECT
#    Theta_d = 0 with prob p_zero
#    Theta_d ~ Lognormal(mu, sigma^2) on positive days
#    Then rescale so E[Theta] = 1
# =========================================================
def fit_zero_inflated_lognormal_theta(theta_hat):
    p_zero = np.mean(theta_hat == 0)
    theta_pos = theta_hat[theta_hat > 0].values

    if len(theta_pos) < 2:
        raise ValueError("Not enough positive theta values for Lognormal fit.")

    # Raw lognormal fit
    s_raw, loc_raw, scale_raw = lognorm.fit(theta_pos, floc=0)

    mean_pos_raw = np.exp(np.log(scale_raw) + 0.5 * s_raw**2)
    target_mean_pos = 1.0 / (1.0 - p_zero) if p_zero < 1 else np.nan
    scale_factor = target_mean_pos / mean_pos_raw

    theta_pos_scaled = theta_pos * scale_factor

    s, loc, scale = lognorm.fit(theta_pos_scaled, floc=0)
    mu = np.log(scale)
    sigma = s

    mean_pos = np.exp(mu + 0.5 * sigma**2)
    var_pos = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)

    E_theta = (1 - p_zero) * mean_pos
    E_theta2 = (1 - p_zero) * (var_pos + mean_pos**2)
    Var_theta = E_theta2 - E_theta**2

    return {
        "model": "Poisson-Lognormal",
        "p_zero": p_zero,
        "mu": mu,
        "sigma": sigma,
        "E_theta": E_theta,
        "Var_theta": Var_theta,
        "theta_scaled": theta_pos_scaled
    }


# =========================================================
# 3. IMPLIED MOMENTS FOR SHARED-THETA MODELS
#    For Poisson mixture:
#      E[N_k] = lambda_k * E[Theta]
#      Var(N_k) = lambda_k * E[Theta] + lambda_k^2 * Var(Theta)
# =========================================================
def implied_moments_shared_theta(lambda_hat, E_theta, Var_theta):
    implied_mean = lambda_hat.values * E_theta
    implied_var = lambda_hat.values * E_theta + (lambda_hat.values ** 2) * Var_theta
    implied_ratio = np.where(implied_mean > 0, implied_var / implied_mean, np.nan)

    return pd.DataFrame({
        "time_bin": lambda_hat.index,
        "implied_mean": implied_mean,
        "implied_var": implied_var,
        "implied_ratio": implied_ratio
    })


# =========================================================
# 4. SIMULATORS
# =========================================================
def simulate_poisson_gamma_days(n_days, lambda_hat, fit, random_state=None):
    rng = np.random.default_rng(random_state)
    out = np.zeros((n_days, len(lambda_hat)), dtype=int)

    for d in range(n_days):
        if rng.uniform() < fit["p_zero"]:
            theta = 0.0
        else:
            theta = rng.gamma(shape=fit["shape"], scale=fit["scale"])

        out[d, :] = rng.poisson(theta * lambda_hat.values)

    return pd.DataFrame(out, columns=lambda_hat.index)


def simulate_poisson_lognormal_days(n_days, lambda_hat, fit, random_state=None):
    rng = np.random.default_rng(random_state)
    out = np.zeros((n_days, len(lambda_hat)), dtype=int)

    for d in range(n_days):
        if rng.uniform() < fit["p_zero"]:
            theta = 0.0
        else:
            theta = rng.lognormal(mean=fit["mu"], sigma=fit["sigma"])

        out[d, :] = rng.poisson(theta * lambda_hat.values)

    return pd.DataFrame(out, columns=lambda_hat.index)


# =========================================================
# 5. SIMULATION-BASED GOODNESS-OF-FIT
# =========================================================
def chi_square_gof_from_simulation(observed_counts, simulated_counts, max_count=6):
    """
    observed_counts, simulated_counts: 1D arrays
    Pool counts into categories 0,1,2,...,max_count-1, >=max_count
    """
    obs = np.asarray(observed_counts)
    sim = np.asarray(simulated_counts)

    def tabulate(arr):
        out = []
        for k in range(max_count):
            out.append(np.sum(arr == k))
        out.append(np.sum(arr >= max_count))
        return np.array(out, dtype=float)

    obs_freq = tabulate(obs)
    sim_freq = tabulate(sim)

    sim_prob = sim_freq / sim_freq.sum()
    expected = sim_prob * obs_freq.sum()

    mask = expected >= 5
    if mask.sum() < 2:
        return {
            "stat": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "obs_freq": obs_freq,
            "expected": expected,
            "note": "too few expected cells >=5"
        }

    stat = ((obs_freq[mask] - expected[mask]) ** 2 / expected[mask]).sum()
    df = mask.sum() - 1
    p_value = 1 - chi2.cdf(stat, df)

    return {
        "stat": stat,
        "df": df,
        "p_value": p_value,
        "obs_freq": obs_freq,
        "expected": expected,
        "note": ""
    }


def run_gof_all_bins(observed_df, simulated_df, model_name, max_count=6):
    rows = []
    for col in observed_df.columns:
        res = chi_square_gof_from_simulation(
            observed_counts=observed_df[col].values,
            simulated_counts=simulated_df[col].values,
            max_count=max_count
        )
        rows.append({
            "model": model_name,
            "time_bin": col,
            "chi2_stat": res["stat"],
            "df": res["df"],
            "p_value": res["p_value"],
            "note": res["note"]
        })
    return pd.DataFrame(rows)


# =========================================================
# 6. MAIN FIT FUNCTION
# =========================================================
def fit_arrival_models(
    df_no_weekend,
    target_class="angiography",
    start_hour=8,
    end_hour=17,
    n_sim_days=None,
    random_state=123,
    make_plots=True
):
    df_work = df_no_weekend.copy()

    # Make sure arrival_day is datetime-like for reindexing
    df_work["arrival_day"] = pd.to_datetime(df_work["arrival_day"])

    # Build time bins
    bin_edges = list(range(start_hour, end_hour + 1, 1))
    bin_labels = [f"{h}:00-{h+1}:00" for h in range(start_hour, end_hour)]

    df_selected = df_work[
        (df_work["hour"] >= start_hour) &
        (df_work["hour"] < end_hour)
    ].copy()

    df_selected["time_bin"] = pd.cut(
        df_selected["hour"],
        bins=bin_edges,
        right=False,
        labels=bin_labels
    )

    # Filter target class
    target_class = str(target_class).strip().lower()
    filtered_df = df_selected[
        df_selected["classification"].astype(str).str.strip().str.lower() == target_class
    ].copy()

    # Build counts by day and bin
    counts_by_day_bin = (
        filtered_df.groupby(["arrival_day", "time_bin"])
        .size()
        .unstack(fill_value=0)
    )

    counts_by_day_bin = counts_by_day_bin.reindex(columns=bin_labels, fill_value=0)

    # Reindex to all weekdays in full range
    all_days = pd.date_range(
        start=df_work["arrival_day"].min(),
        end=df_work["arrival_day"].max(),
        freq="D"
    )
    all_weekdays = all_days[all_days.dayofweek < 5]

    counts_by_day_bin = counts_by_day_bin.reindex(all_weekdays, fill_value=0)
    counts_by_day_bin.index.name = "arrival_day"

    print("Counts by day and time bin:")
    print(counts_by_day_bin.head())

    # Observed moments
    obs_mean = counts_by_day_bin.mean(axis=0)
    obs_var = counts_by_day_bin.var(axis=0, ddof=1)
    obs_ratio = np.where(obs_mean > 0, obs_var / obs_mean, np.nan)

    lambda_hat = obs_mean.copy()
    daily_totals = counts_by_day_bin.sum(axis=1)
    expected_daily_total = lambda_hat.sum()

    if expected_daily_total <= 0:
        raise ValueError(f"Expected daily total is zero. No {target_class} arrivals available.")

    theta_hat = daily_totals / expected_daily_total

    summary_obs = pd.DataFrame({
        "bin": bin_labels,
        "obs_mean": obs_mean.values,
        "obs_var": obs_var.values,
        "obs_ratio": obs_ratio
    })

    print("\nObserved summary:")
    print(summary_obs)

    # Fit models
    pg_fit = fit_zero_inflated_gamma_theta(theta_hat)
    pln_fit = fit_zero_inflated_lognormal_theta(theta_hat)

    print("\nPoisson-Gamma fit:")
    print(pg_fit)

    print("\nPoisson-Lognormal fit:")
    print(pln_fit)

    # Implied moments
    pg_mom = implied_moments_shared_theta(lambda_hat, pg_fit["E_theta"], pg_fit["Var_theta"])
    pg_mom = pg_mom.rename(columns={
        "implied_mean": "pg_mean",
        "implied_var": "pg_var",
        "implied_ratio": "pg_ratio"
    })

    pln_mom = implied_moments_shared_theta(lambda_hat, pln_fit["E_theta"], pln_fit["Var_theta"])
    pln_mom = pln_mom.rename(columns={
        "implied_mean": "pln_mean",
        "implied_var": "pln_var",
        "implied_ratio": "pln_ratio"
    })

    comparison = pd.DataFrame({
        "time_bin": obs_mean.index,
        "obs_mean": obs_mean.values,
        "obs_var": obs_var.values,
        "obs_ratio": obs_ratio
    })

    comparison = comparison.merge(pg_mom, on="time_bin", how="left")
    comparison = comparison.merge(pln_mom, on="time_bin", how="left")

    print("\nMoment comparison:")
    print(comparison)

    # Number of simulation days
    n_obs_days = len(counts_by_day_bin)
    if n_sim_days is None:
        n_sim_days = max(5000, 50 * n_obs_days)

    # Simulate
    sim_pg = simulate_poisson_gamma_days(
        n_days=n_sim_days,
        lambda_hat=lambda_hat,
        fit=pg_fit,
        random_state=random_state
    )

    sim_pln = simulate_poisson_lognormal_days(
        n_days=n_sim_days,
        lambda_hat=lambda_hat,
        fit=pln_fit,
        random_state=random_state
    )

    # GOF
    gof_pg = run_gof_all_bins(counts_by_day_bin, sim_pg, "Poisson-Gamma", max_count=6)
    gof_pln = run_gof_all_bins(counts_by_day_bin, sim_pln, "Poisson-Lognormal", max_count=6)

    gof_all = pd.concat([gof_pg, gof_pln], ignore_index=True)

    print("\nGoodness-of-fit by bin:")
    print(gof_all)

    gof_summary = (
        gof_all.groupby("model")
        .agg(
            mean_p_value=("p_value", "mean"),
            median_p_value=("p_value", "median"),
            num_bins_p_gt_005=("p_value", lambda x: np.sum(x > 0.05)),
            num_bins_tested=("p_value", lambda x: x.notna().sum())
        )
        .reset_index()
    )

    print("\nGOF summary:")
    print(gof_summary)

    

    # Plots
    if make_plots:
        plt.figure(figsize=(11, 5))
        plt.plot(comparison["time_bin"], comparison["obs_ratio"], marker="o", label="Observed")
        plt.plot(comparison["time_bin"], comparison["pg_ratio"], marker="s", label="Poisson-Gamma")
        plt.plot(comparison["time_bin"], comparison["pln_ratio"], marker="^", label="Poisson-Lognormal")
        plt.axhline(1.0, linestyle="--")
        plt.title("Variance-to-Mean Ratio by Bin - Model Comparison")
        plt.xlabel("Time Bin")
        plt.ylabel("Variance / Mean")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

        example_bin = bin_labels[3] if len(bin_labels) > 3 else bin_labels[0]

    # Return all useful outputs
    return {
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
        "gof_summary": gof_summary
    }


# =========================================================
# 7. OPTIONAL: RUN DIRECTLY
# =========================================================
if __name__ == "__main__":
    results = fit_arrival_models(
        df_no_weekend=df_no_weekend,
        target_class="angiography",
        start_hour=8,
        end_hour=17,
        make_plots=True
    )

    # Example: access returned values
    lambda_hat = results["lambda_hat"]
    pg_fit = results["pg_fit"]
    pln_fit = results["pln_fit"]

    print("\nSaved parameters for reuse:")
    print("lambda_hat:")
    print(lambda_hat)
    print("\npg_fit:")
    print(pg_fit)
    print("\npln_fit:")
    print(pln_fit)