from __future__ import annotations

"""
Subset selection with CRN + KN with CRN
for brute-force repeated-weekday Qik policies.

Policy structure
----------------
Each candidate policy consists of:
1. A timetable choice: R1 or R2
2. A daily Qik matrix of shape (2, 8)

The daily Qik is repeated Monday-Friday to form the weekly policy.
"""

import math
from dataclasses import dataclass
from functools import partial
from itertools import islice

import numpy as np
import pandas as pd
from scipy import stats

import Policy_defined
from input_loader import load_all_ir_inputs
from simulation_model import run_replications


# =========================================================
# USER SETTINGS
# =========================================================
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

SEARCH_TIMETABLE = "R1"   # "R1" or "R2"
MAX_QIK_VALUE = 3

# For debugging only. Set to None to use all candidates.
MAX_CANDIDATES = None


# =========================================================
# POLICY CANDIDATE OBJECT
# =========================================================
@dataclass
class CandidatePolicy:
    system_id: int
    name: str
    daily_qik: np.ndarray
    weekly_qik: np.ndarray
    build_policy: callable


# =========================================================
# LOAD INPUTS ONCE
# =========================================================
LOADED_INPUTS = load_all_ir_inputs(
    arrival_json_path=ARRIVAL_JSON_PATH,
    service_json_path=SERVICE_JSON_PATH,
    raw_data_path=RAW_DATA_PATH,
)


# =========================================================
# BUILD CANDIDATE POLICIES
# =========================================================
def _resolve_search_components():
    if SEARCH_TIMETABLE == "R1":
        timetable = Policy_defined.example_timetable_R1()
        build_policy_func = Policy_defined.build_bruteforce_policy_R1
    elif SEARCH_TIMETABLE == "R2":
        timetable = Policy_defined.example_timetable_R2()
        build_policy_func = Policy_defined.build_bruteforce_policy_R2
    else:
        raise ValueError("SEARCH_TIMETABLE must be 'R1' or 'R2'.")

    return timetable, build_policy_func


def _build_candidate_policies():
    timetable, build_policy_func = _resolve_search_components()

    generator = Policy_defined.generate_bruteforce_daily_qik_candidates(
        max_value=MAX_QIK_VALUE,
        timetable=timetable,
    )

    if MAX_CANDIDATES is not None:
        generator = islice(generator, MAX_CANDIDATES)

    candidates = []
    for idx, daily_qik in enumerate(generator, start=1):
        daily_qik = np.asarray(daily_qik, dtype=int).copy()
        weekly_qik = Policy_defined.build_weekly_qik_from_daily(daily_qik)

        candidates.append(
            CandidatePolicy(
                system_id=idx,
                name=f"{SEARCH_TIMETABLE}_cand_{idx}",
                daily_qik=daily_qik,
                weekly_qik=weekly_qik,
                build_policy=partial(build_policy_func, daily_qik=daily_qik),
            )
        )

    return candidates


POLICIES = _build_candidate_policies()


# =========================================================
# SMALL HELPERS
# =========================================================
def _require_policies():
    if len(POLICIES) == 0:
        raise RuntimeError("No candidate policies were generated.")


def candidate_obj(system_index: int) -> CandidatePolicy:
    if system_index < 1 or system_index > len(POLICIES):
        raise IndexError(f"system_index must be between 1 and {len(POLICIES)}.")
    return POLICIES[system_index - 1]


def candidate_name(system_index: int) -> str:
    return candidate_obj(system_index).name


def _resolve_policy_indices(k, policy_indices=None):
    if policy_indices is None:
        indices = list(range(1, k + 1))
    else:
        indices = list(policy_indices)

    if len(indices) != k:
        raise ValueError("k must match the number of candidate policy indices.")
    return indices


def policy_table(policy_indices):
    rows = []
    for i in policy_indices:
        cand = candidate_obj(i)
        rows.append(
            {
                "system": i,
                "policy": cand.name,
                "daily_qik": cand.daily_qik.tolist(),
            }
        )
    return pd.DataFrame(rows)


# =========================================================
# SIMULATION WRAPPER
# =========================================================
def MySim(system_index, n=1, seed=None):
    """
    Run simulation replications for one candidate policy.

    Parameters
    ----------
    system_index : int
        1-based candidate index
    n : int
        Number of replications
    seed : int or None
        Base seed
    """
    _require_policies()

    if seed is None:
        seed = BASE_SEED

    candidate = candidate_obj(system_index)
    num_replications = int(n or 1)

    rep_df = run_replications(
        num_replications=num_replications,
        num_weeks=NUM_WEEKS,
        loaded_inputs=LOADED_INPUTS,
        policy=candidate.build_policy(),
        base_seed=int(seed),
        warmup_weeks=WARMUP_WEEKS,
    )

    values = rep_df["H"].to_numpy(dtype=float)

    if num_replications == 1:
        return float(values[0])
    return values


# =========================================================
# SUBSET SELECTION WITH CRN
# =========================================================
def subset_crn(k, alpha, n, seed, policy_indices=None):
    """
    Common-random-number subset selection screen.
    """
    indices = _resolve_policy_indices(k, policy_indices)

    sample_matrix = np.column_stack(
        [np.asarray(MySim(system_index, n=n, seed=seed), dtype=float) for system_index in indices]
    )
    means = sample_matrix.mean(axis=0)

    if len(indices) == 1:
        summary_df = pd.DataFrame(
            {
                "system": indices,
                "policy": [candidate_name(i) for i in indices],
                "mean_H_subset": means,
            }
        )
        return indices, summary_df

    t_value = stats.t.ppf(
        1.0 - alpha / max(len(indices) - 1, 1),
        df=max(n - 1, 1),
    )

    keep_indices = []

    for col_i, system_i in enumerate(indices):
        eliminated = False

        for col_j, system_j in enumerate(indices):
            if system_i == system_j:
                continue

            diff = sample_matrix[:, col_i] - sample_matrix[:, col_j]
            s2_ij = float(np.var(diff, ddof=1)) if n > 1 else 0.0
            margin = t_value * math.sqrt(s2_ij / n) if n > 0 else 0.0

            # Smaller H is better
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
            "policy": [candidate_name(i) for i in indices],
            "mean_H_subset": means,
        }
    ).sort_values("mean_H_subset", ascending=True, ignore_index=True)

    return keep_indices, summary_df


# =========================================================
# KN WITH CRN
# =========================================================
def kn_crn(k, alpha, n0, delta, seed, policy_indices=None):
    """
    Kim-Nelson procedure with common random numbers.
    """
    indices = _resolve_policy_indices(k, policy_indices)

    if k == 1:
        best_system = indices[0]
        summary_df = pd.DataFrame(
            [
                {
                    "system": best_system,
                    "policy": candidate_name(best_system),
                    "mean_H_final": float(MySim(best_system, seed=seed)),
                }
            ]
        )
        return {"Best": best_system, "n": 1, "Elim": np.zeros(1), "summary": summary_df}

    if n0 < 2:
        raise ValueError("n0 must be at least 2 for KN with CRN.")

    II = np.arange(1, k + 1)
    Active = np.full(k, True)
    Elim = np.zeros(k)

    Yn0 = np.zeros((k, n0))

    # Initial CRN samples
    for i in range(k):
        system_index = indices[i]
        Yn0[i, :] = np.asarray(MySim(system_index, n=n0, seed=seed), dtype=float)

    Sigma = np.cov(Yn0, bias=False)

    eta = 0.5 * ((2 * alpha / (k - 1)) ** (-2 / (n0 - 1)) - 1)
    h2 = 2 * eta * (n0 - 1)

    Ysum = Yn0.sum(axis=1)
    r = n0

    while np.sum(Active) > 1:
        r += 1
        ATemp = Active.copy()

        # One extra CRN observation for each active system
        for i in II[Active]:
            system_index = indices[i - 1]
            Ysum[i - 1] += MySim(system_index, n=1, seed=seed + r - 1)

        for i in II[Active]:
            for l in II[Active]:
                if i == l:
                    continue

                S2diff = Sigma[i - 1, i - 1] + Sigma[l - 1, l - 1] - 2 * Sigma[i - 1, l - 1]
                W = max(0.0, (delta / 2.0) * (h2 * S2diff / delta**2 - r))

                # Smaller H is better
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
            "policy": [candidate_name(i) for i in indices],
            "mean_H_final": Ysum / r,
        }
    ).sort_values("mean_H_final", ascending=True, ignore_index=True)

    return {"Best": best_system, "n": r, "Elim": Elim, "summary": summary_df}


# =========================================================
# MAIN
# =========================================================
def main():
    _require_policies()

    num_systems = len(POLICIES)

    print("Candidate policy set")
    print(f"Timetable = {SEARCH_TIMETABLE}")
    print(f"Number of candidate systems = {num_systems}")
    print(policy_table(range(1, num_systems + 1)).to_string(index=False))

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
    print(policy_table(subset).to_string(index=False))

    if len(subset) == 1:
        best_system = subset[0]
        best_mean = subset_df.loc[subset_df["system"] == best_system, "mean_H_subset"].iloc[0]

        result = {
            "Best": best_system,
            "n": SUBSET_INITIAL_REPS,
            "Elim": np.zeros(1),
            "summary": pd.DataFrame(
                [
                    {
                        "system": best_system,
                        "policy": candidate_name(best_system),
                        "mean_H_final": best_mean,
                    }
                ]
            ),
        }
    else:
        result = kn_crn(
            k=len(subset),
            alpha=KN_ALPHA,
            n0=KN_INITIAL_REPS,
            delta=KN_DELTA,
            seed=BASE_SEED + 10_000,
            policy_indices=subset,
        )
        best_system = result["Best"]

    best_candidate = candidate_obj(best_system)

    print("\nFinal KN with CRN result")
    print(result["summary"].to_string(index=False))
    print(f"Best system = {best_system}")
    print(f"Total samples used = {result['n']}")

    print("\nSelected best policy")
    print(policy_table([best_system]).to_string(index=False))

    print("\nSelected best policy weekly Qik")
    print(
        pd.DataFrame(
            best_candidate.weekly_qik,
            index=["Interventional", "Angiography"],
            columns=[f"block_{k}" for k in range(40)],
        ).to_string()
    )

    final_eval_df = run_replications(
        num_replications=FINAL_EVAL_REPS,
        num_weeks=NUM_WEEKS,
        loaded_inputs=LOADED_INPUTS,
        policy=best_candidate.build_policy(),
        base_seed=BASE_SEED + 20_000,
        warmup_weeks=WARMUP_WEEKS,
    )

    print("\nFinal check on selected policy")
    print(
        final_eval_df[
            ["replication", "Z1_wait_time", "Z2_overtime", "Z3_congestion", "H"]
        ].to_string(index=False)
    )
    print(f"\nAverage H over final check = {final_eval_df['H'].mean():.4f}")


if __name__ == "__main__":
    main()