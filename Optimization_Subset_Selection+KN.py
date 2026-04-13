import inspect
import math

import numpy as np
import pandas as pd
from scipy import stats

import Policy_defined
from input_loader import load_all_ir_inputs
from simulation_model import run_replications


# Keep these settings near the top so the experiment is easy to adjust.
ARRIVAL_JSON_PATH = "arrival_model_params.json"
SERVICE_JSON_PATH = "services rate.json"
RAW_DATA_PATH = "df_selected.xlsx"

NUM_WEEKS = 52
WARMUP_WEEKS = 8
BASE_SEED = 123

SUBSET_ALPHA = 0.05
SUBSET_INITIAL_REPS = 2

KN_ALPHA = 0.05
KN_INITIAL_REPS = 10
KN_DELTA = 15.0

FINAL_EVAL_REPS = 5


def _discover_policy_builders():
    """Read all candidate policies from Policy_defined.py."""
    builders = []
    for name, func in inspect.getmembers(Policy_defined, inspect.isfunction):
        if name.startswith("example_policy_"):
            builders.append((name.replace("example_policy_", ""), func))

    builders.sort(key=lambda item: item[0])
    if not builders:
        raise ValueError("No example_policy_* functions were found in Policy_defined.py.")
    return builders


POLICIES = _discover_policy_builders()
LOADED_INPUTS = load_all_ir_inputs(
    arrival_json_path=ARRIVAL_JSON_PATH,
    service_json_path=SERVICE_JSON_PATH,
    raw_data_path=RAW_DATA_PATH,
)
GLOBAL_RNG = np.random.default_rng(BASE_SEED)


def policy_name(system_index):
    return POLICIES[system_index - 1][0]


def _resolve_policy_indices(k, policy_indices=None):
    if policy_indices is None:
        indices = list(range(1, k + 1))
    else:
        indices = list(policy_indices)

    if len(indices) != k:
        raise ValueError("k must match the number of candidate policy indices.")
    return indices


def MySim(system_index, n=1, seed=None, RandomSeed=None):
    """Simulation wrapper kept close to the original subset-selection pattern.

    Parameters
    ----------
    system_index:
        1-based policy index.
    n:
        Number of replications to run. If n > 1, returns a numpy array.
    seed / RandomSeed:
        Use the same base seed across systems when CRN is desired.
    """
    if system_index < 1 or system_index > len(POLICIES):
        raise IndexError(f"system_index must be between 1 and {len(POLICIES)}.")

    _, builder = POLICIES[system_index - 1]

    if RandomSeed is not None:
        num_replications = 1
        base_seed = int(RandomSeed)
    else:
        num_replications = int(n or 1)
        if seed is None:
            base_seed = int(GLOBAL_RNG.integers(1, 2**31 - 1))
        else:
            base_seed = int(seed)

    rep_df = run_replications(
        num_replications=num_replications,
        num_weeks=NUM_WEEKS,
        loaded_inputs=LOADED_INPUTS,
        policy=builder(),
        base_seed=base_seed,
        warmup_weeks=WARMUP_WEEKS,
    )
    values = rep_df["H"].to_numpy(dtype=float)

    if RandomSeed is not None or num_replications == 1:
        return float(values[0])
    return values


def subset_crn(k, alpha, n, seed, policy_indices=None):
    """Common-random-number subset selection screen."""
    indices = _resolve_policy_indices(k, policy_indices)
    sample_matrix = np.column_stack(
        [np.asarray(MySim(system_index, n=n, seed=seed), dtype=float) for system_index in indices]
    )
    means = sample_matrix.mean(axis=0)

    if len(indices) == 1:
        summary_df = pd.DataFrame(
            {
                "system": indices,
                "policy": [policy_name(i) for i in indices],
                "mean_H_subset": means,
            }
        )
        return indices, summary_df

    t_value = stats.t.ppf(1.0 - alpha / max(len(indices) - 1, 1), df=max(n - 1, 1))
    keep_indices = []

    for col_i, system_i in enumerate(indices):
        eliminated = False
        for col_j, system_j in enumerate(indices):
            if system_i == system_j:
                continue
            diff = sample_matrix[:, col_i] - sample_matrix[:, col_j]
            s2_ij = float(np.var(diff, ddof=1)) if n > 1 else 0.0
            margin = t_value * math.sqrt(s2_ij / n) if n > 0 else 0.0
            if means[col_i] > means[col_j] + margin:
                eliminated = True
                break
        if not eliminated:
            keep_indices.append(system_i)

    if not keep_indices:
        keep_indices = [indices[int(np.argmin(means))]]

    summary_df = pd.DataFrame(
        {
            "system": indices,
            "policy": [policy_name(i) for i in indices],
            "mean_H_subset": means,
        }
    ).sort_values("mean_H_subset", ascending=True, ignore_index=True)

    return keep_indices, summary_df


def kn(k, alpha, n0, delta, policy_indices=None):
    """
    Perform the KN procedure without common random numbers.

    :param k: Number of systems
    :param alpha: Significance level
    :param n0: Initial sample size
    :param delta: Indifference-zone parameter
    :param policy_indices: Optional list of actual policy ids to compare
    """
    indices = _resolve_policy_indices(k, policy_indices)

    if k == 1:
        best_system = indices[0]
        summary_df = pd.DataFrame(
            [{"system": best_system, "policy": policy_name(best_system), "mean_H_final": float(MySim(best_system))}]
        )
        return {"Best": best_system, "n": 1, "Elim": np.zeros(1), "summary": summary_df}

    if n0 < 2:
        raise ValueError("n0 must be at least 2 for KN because pairwise variances are required.")

    II = np.arange(1, k + 1)
    Active = np.full(k, True)
    Elim = np.zeros(k)

    Yn0 = np.zeros((k, n0))
    for i in range(k):
        system_index = indices[i]
        for r in range(n0):
            Yn0[i, r] = MySim(system_index)

    S2 = np.zeros((k, k))
    for i in range(k):
        for l in range(i + 1, k):
            diff = Yn0[i, :] - Yn0[l, :]
            S2diff = float(np.var(diff, ddof=1))
            S2[i, l] = S2diff
            S2[l, i] = S2diff

    eta = 0.5 * ((2 * alpha / (k - 1)) ** (-2 / (n0 - 1)) - 1)
    h2 = 2 * eta * (n0 - 1)

    Ysum = Yn0.sum(axis=1)
    r = n0

    while np.sum(Active) > 1:
        r += 1
        ATemp = Active.copy()

        for i in II[Active]:
            system_index = indices[i - 1]
            Ysum[i - 1] += MySim(system_index)

        for i in II[Active]:
            for l in II[Active]:
                if i != l:
                    W = max(0.0, (delta / 2.0) * (h2 * S2[i - 1, l - 1] / delta**2 - r))

                    if Ysum[i - 1] > Ysum[l - 1] + W:
                        ATemp[i - 1] = False
                        Elim[i - 1] = r
                        break

        Active = ATemp

    best_local = int(II[Active][0])
    best_system = indices[best_local - 1]
    summary_df = pd.DataFrame(
        {
            "system": indices,
            "policy": [policy_name(i) for i in indices],
            "mean_H_final": Ysum / r,
        }
    ).sort_values("mean_H_final", ascending=True, ignore_index=True)

    return {"Best": best_system, "n": r, "Elim": Elim, "summary": summary_df}


def kn_crn(k, alpha, n0, delta, seed, policy_indices=None):
    """
    Perform the KN procedure with Common Random Numbers (CRN).

    :param k: Number of systems
    :param alpha: Significance level
    :param n0: Initial sample size
    :param delta: Indifference-zone parameter
    :param seed: Random seed used to generate common random numbers
    :param policy_indices: Optional list of actual policy ids to compare
    """
    indices = _resolve_policy_indices(k, policy_indices)

    if k == 1:
        best_system = indices[0]
        summary_df = pd.DataFrame(
            [{"system": best_system, "policy": policy_name(best_system), "mean_H_final": float(MySim(best_system))}]
        )
        return {"Best": best_system, "n": 1, "Elim": np.zeros(1), "summary": summary_df}

    if n0 < 2:
        raise ValueError("n0 must be at least 2 for KN because pairwise variances are required.")

    II = np.arange(1, k + 1)
    Active = np.full(k, True)
    Elim = np.zeros(k)

    Yn0 = np.zeros((k, n0))

    # Initial samples using CRN
    for i in range(k):
        system_index = indices[i]
        Yn0[i, :] = MySim(system_index, n0, seed)

    # Covariance matrix of initial samples
    Sigma = np.cov(Yn0, bias=False)

    eta = 0.5 * ((2 * alpha / (k - 1)) ** (-2 / (n0 - 1)) - 1)
    h2 = 2 * eta * (n0 - 1)

    Ysum = Yn0.sum(axis=1)
    r = n0

    while np.sum(Active) > 1:
        r += 1
        ATemp = Active.copy()

        # Sequential samples using same seed across systems
        for i in II[Active]:
            system_index = indices[i - 1]
            Ysum[i - 1] += MySim(system_index, RandomSeed=seed + r - 1)

        for i in II[Active]:
            for l in II[Active]:
                if i != l:
                    S2diff = Sigma[i - 1, i - 1] + Sigma[l - 1, l - 1] - 2 * Sigma[i - 1, l - 1]
                    W = max(0.0, (delta / 2.0) * (h2 * S2diff / delta**2 - r))

                    # Smaller H is better, so eliminate i when it is clearly worse.
                    if Ysum[i - 1] > Ysum[l - 1] + W:
                        ATemp[i - 1] = False
                        Elim[i - 1] = r
                        break

        Active = ATemp

    best_local = int(II[Active][0])
    best_system = indices[best_local - 1]
    summary_df = pd.DataFrame(
        {
            "system": indices,
            "policy": [policy_name(i) for i in indices],
            "mean_H_final": Ysum / r,
        }
    ).sort_values("mean_H_final", ascending=True, ignore_index=True)

    return {"Best": best_system, "n": r, "Elim": Elim, "summary": summary_df}


def _policy_table(policy_indices):
    rows = [{"system": i, "policy": policy_name(i)} for i in policy_indices]
    return pd.DataFrame(rows)


def main():
    num_systems = len(POLICIES)

    print("Candidate policy set")
    print(_policy_table(range(1, num_systems + 1)).to_string(index=False))

    subset, subset_df = subset_crn(
        k=num_systems,
        alpha=SUBSET_ALPHA,
        n=SUBSET_INITIAL_REPS,
        seed=BASE_SEED,
    )

    print("\nSubset selection with CRN")
    print(f"subset_reps = {SUBSET_INITIAL_REPS}")
    print(subset_df.to_string(index=False))
    print("\nPolicies kept after subset screening")
    print(_policy_table(subset).to_string(index=False))

    if len(subset) == 1:
        best_system = subset[0]
        kn_result = {
            "Best": best_system,
            "n": SUBSET_INITIAL_REPS,
            "Elim": np.zeros(1),
            "summary": pd.DataFrame(
                [
                    {
                        "system": best_system,
                        "policy": policy_name(best_system),
                        "mean_H_final": subset_df.loc[
                            subset_df["system"] == best_system, "mean_H_subset"
                        ].iloc[0],
                    }
                ]
            ),
        }
    else:
        kn_result = kn_crn(
            k=len(subset),
            alpha=KN_ALPHA,
            n0=KN_INITIAL_REPS,
            delta=KN_DELTA,
            seed=BASE_SEED + 10_000,
            policy_indices=subset,
        )
        best_system = kn_result["Best"]

    print("\nFinal KN result")
    print(kn_result["summary"].to_string(index=False))
    print(f"Best system = {kn_result['Best']}")
    print(f"Total samples used = {kn_result['n']}")

    final_eval_df = run_replications(
        num_replications=FINAL_EVAL_REPS,
        num_weeks=NUM_WEEKS,
        loaded_inputs=LOADED_INPUTS,
        policy=POLICIES[best_system - 1][1](),
        base_seed=BASE_SEED + 20_000,
        warmup_weeks=WARMUP_WEEKS,
    )

    print("\nSelected best policy")
    print(_policy_table([best_system]).to_string(index=False))
    print("\nFinal check on selected policy")
    print(
        final_eval_df[["replication", "Z1_wait_time", "Z2_overtime", "Z3_congestion", "H"]].to_string(
            index=False
        )
    )
    print(f"\nAverage H over final check = {final_eval_df['H'].mean():.4f}")


if __name__ == "__main__":
    main()
