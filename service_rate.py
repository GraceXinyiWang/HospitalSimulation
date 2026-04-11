import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from arrival_rate import preprocess_arrival_data











df_no_weekend = preprocess_arrival_data("df_selected.xlsx")



# --------------------------------------------------
# 1. Create procedure duration in hours
# --------------------------------------------------
df_no_weekend = df_no_weekend.copy()

df_no_weekend['Procedure_duration'] = (
    df_no_weekend['ScanStopF'] - df_no_weekend['ScanStartF']
)

df_no_weekend['Procedure_duration_hours'] = (
    df_no_weekend['Procedure_duration'].dt.total_seconds() / 3600
)

# keep outliers separately
outliers = df_no_weekend[
    df_no_weekend['Procedure_duration_hours'] > 6
].copy()

# drop durations > 6 hours
df_no_weekend = df_no_weekend[
    df_no_weekend['Procedure_duration_hours'].notna() &
    (df_no_weekend['Procedure_duration_hours'] > 0) &
    (df_no_weekend['Procedure_duration_hours'] <= 6)
].copy()

print("Number of dropped outliers (>6 hours):", len(outliers))

# --------------------------------------------------
# 2. Function to fit one classification
# --------------------------------------------------
def fit_service_by_class(service_data, classification_name):
    service_data = pd.Series(service_data).dropna()
    service_data = service_data[service_data > 0]

    if len(service_data) < 5:
        print(f"\n{classification_name}: not enough data to fit.")
        return None

    print("\n" + "=" * 90)
    print(f"Classification: {classification_name}")
    print(f"Number of observations: {len(service_data)}")
    print(f"Mean duration (hours): {service_data.mean():.4f}")
    print(f"Std duration (hours): {service_data.std():.4f}")
    print(f"Min duration (hours): {service_data.min():.4f}")
    print(f"Max duration (hours): {service_data.max():.4f}")

    mean_service_time = service_data.mean()
    mu = 1 / mean_service_time
    print(f"Average service rate (patients/hour): {mu:.4f}")

    # --------------------------------------------------
    # Histogram
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(service_data, bins=30, density=True, alpha=0.6)
    plt.title(f'Histogram of Procedure Duration (Hours) - {classification_name}')
    plt.xlabel('Procedure Duration (Hours)')
    plt.ylabel('Density')
    plt.show()

    # --------------------------------------------------
    # Fit distributions
    # --------------------------------------------------
    exp_loc, exp_scale = stats.expon.fit(service_data, floc=0)
    exp_params = (exp_loc, exp_scale)

    gamma_a, gamma_loc, gamma_scale = stats.gamma.fit(service_data, floc=0)
    gamma_params = (gamma_a, gamma_loc, gamma_scale)

    weibull_c, weibull_loc, weibull_scale = stats.weibull_min.fit(service_data, floc=0)
    weibull_params = (weibull_c, weibull_loc, weibull_scale)

    lognorm_sigma, lognorm_loc, lognorm_scale = stats.lognorm.fit(service_data, floc=0)
    lognorm_params = (lognorm_sigma, lognorm_loc, lognorm_scale)

    print("Exponential params:", exp_params)
    print("Gamma params:", gamma_params)
    print("Weibull params:", weibull_params)
    print("Lognormal params:", lognorm_params)

    # --------------------------------------------------
    # Overlay fitted densities
    # --------------------------------------------------
    plt.figure(figsize=(12, 8))
    plt.hist(service_data, bins=30, density=True, alpha=0.5, label='Data')

    x = np.linspace(service_data.min(), service_data.max(), 300)

    plt.plot(x, stats.expon.pdf(x, *exp_params), label='Exponential')
    plt.plot(x, stats.gamma.pdf(x, *gamma_params), label='Gamma')
    plt.plot(x, stats.weibull_min.pdf(x, *weibull_params), label='Weibull')
    plt.plot(x, stats.lognorm.pdf(x, *lognorm_params), label='Lognormal')

    plt.title(f'Procedure Duration (Hours) with Fitted Distributions - {classification_name}')
    plt.xlabel('Procedure Duration (Hours)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # --------------------------------------------------
    # Q-Q plots
    # --------------------------------------------------
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    sorted_data = np.sort(service_data)
    prob = (np.arange(1, len(service_data) + 1) - 0.5) / len(service_data)

    theoretical_exp = stats.expon.ppf(prob, *exp_params)
    axs[0].plot(theoretical_exp, sorted_data, 'o', markersize=4)
    axs[0].plot(theoretical_exp, theoretical_exp, 'r--')
    axs[0].set_title(f"Q-Q Plot: Exponential\n{classification_name}")

    theoretical_gamma = stats.gamma.ppf(prob, *gamma_params)
    axs[1].plot(theoretical_gamma, sorted_data, 'o', markersize=4)
    axs[1].plot(theoretical_gamma, theoretical_gamma, 'r--')
    axs[1].set_title(f"Q-Q Plot: Gamma\n{classification_name}")

    theoretical_weibull = stats.weibull_min.ppf(prob, *weibull_params)
    axs[2].plot(theoretical_weibull, sorted_data, 'o', markersize=4)
    axs[2].plot(theoretical_weibull, theoretical_weibull, 'r--')
    axs[2].set_title(f"Q-Q Plot: Weibull\n{classification_name}")

    theoretical_lognorm = stats.lognorm.ppf(prob, *lognorm_params)
    axs[3].plot(theoretical_lognorm, sorted_data, 'o', markersize=4)
    axs[3].plot(theoretical_lognorm, theoretical_lognorm, 'r--')
    axs[3].set_title(f"Q-Q Plot: Lognormal\n{classification_name}")

    for ax in axs:
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Empirical Quantiles")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # KS tests
    # --------------------------------------------------
    def perform_ks_test(data, distribution_name, params):
        if distribution_name == 'Exponential':
            D, p_value = stats.kstest(data, 'expon', args=params)
        elif distribution_name == 'Gamma':
            D, p_value = stats.kstest(data, 'gamma', args=params)
        elif distribution_name == 'Weibull':
            D, p_value = stats.kstest(data, 'weibull_min', args=params)
        elif distribution_name == 'Lognormal':
            D, p_value = stats.kstest(data, 'lognorm', args=params)

        return D, p_value

    results = []
    for dist_name, params in [
        ('Exponential', exp_params),
        ('Gamma', gamma_params),
        ('Weibull', weibull_params),
        ('Lognormal', lognorm_params)
    ]:
        D, p_value = perform_ks_test(service_data, dist_name, params)
        results.append([dist_name, D, p_value])

    results_df = pd.DataFrame(results, columns=['distribution', 'KS_D', 'KS_p_value'])
    print("\nKS test results:")
    print(results_df.sort_values('KS_D'))

    return results_df

# --------------------------------------------------
# 3. Run for each classification
# --------------------------------------------------
all_results = {}

for classification in sorted(df_no_weekend['classification'].dropna().unique()):
    class_data = df_no_weekend.loc[
        df_no_weekend['classification'] == classification,
        'Procedure_duration_hours'
    ]
    all_results[classification] = fit_service_by_class(class_data, classification)