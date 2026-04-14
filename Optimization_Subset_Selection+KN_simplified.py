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
Star: Both classes can share the same daily Qik value at each
within-day block to reduce the search space.
"""

import math
from dataclasses import dataclass
from functools import partial
from itertools import islice, product

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

NUM_WEEKS = 210
WARMUP_WEEKS = 20
BASE_SEED = 123

SUBSET_ALPHA = 0.05
SUBSET_INITIAL_REPS = 2

KN_ALPHA = 0.05
KN_INITIAL_REPS = 3
KN_DELTA = 0.01

FINAL_EVAL_REPS = 5

SEARCH_TIMETABLE = "both"   # "R1", "R2", or "both"
MAX_QIK_VALUE = 2  # Max value for each Qik entry (0, 1, or 2)``
SHARE_DAILY_QIK_ACROSS_CLASSES = True



# For debugging only. Set to None to use all candidates per timetable.
MAX_CANDIDATES = None
MAX_PREVIEW_ROWS = 20


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
def _resolve_search_timetables():
    search_value = str(SEARCH_TIMETABLE).strip().upper()

    if search_value == "BOTH":
        return ["R1", "R2"]
    if search_value in {"R1", "R2"}:
        return [search_value]

    raise ValueError("SEARCH_TIMETABLE must be 'R1', 'R2', or 'both'.")


def _resolve_search_components(search_timetable):
    if search_timetable == "R1":
        timetable = Policy_defined.example_timetable_R1()
        build_policy_func = Policy_defined.build_bruteforce_policy_R1
    elif search_timetable == "R2":
        timetable = Policy_defined.example_timetable_R2()
        build_policy_func = Policy_defined.build_bruteforce_policy_R2

    return timetable, build_policy_func


def _build_candidate_policies(search_timetable):
    timetable, build_policy_func = _resolve_search_components(search_timetable)

    if SHARE_DAILY_QIK_ACROSS_CLASSES:
        generator = _generate_shared_daily_qik_candidates(
            max_value=MAX_QIK_VALUE,
            timetable=timetable,
        )
    else:
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
                name=f"{search_timetable}_cand_{idx}",
                daily_qik=daily_qik,
                weekly_qik=weekly_qik,
                build_policy=partial(build_policy_func, daily_qik=daily_qik),
            )
        )

    return candidates


ACTIVE_TIMETABLE = None
POLICIES = []


# =========================================================
# SMALL HELPERS
# =========================================================

def _require_policies():
    if len(POLICIES) == 0:
        raise RuntimeError("No candidate policies were generated.")


def candidate_obj(system_index: int) -> CandidatePolicy:
    return POLICIES[system_index - 1]


def candidate_name(system_index: int) -> str:
    return candidate_obj(system_index).name


def _resolve_policy_indices(k, policy_indices=None):
    if policy_indices is None:
        indices = list(range(1, k + 1))
    else:
        indices = list(policy_indices)
    return indices


def _shared_daily_feasible_blocks(timetable):
    weekly_mask = np.asarray(timetable.feasible_qik, dtype=int)
    stacked = np.stack(
        [weekly_mask[:, 8 * day : 8 * (day + 1)] for day in range(5)],
        axis=0,
    )

    # Shared block k is free if at least one class can use it on at least one weekday.
    feasible_daily_by_class = stacked.max(axis=0)
    return feasible_daily_by_class.max(axis=0)


def _generate_shared_daily_qik_candidates(
    max_value: int = 3,
    timetable=None,
):
    """
    Generator where both classes share one Qik value at each daily block.

    Decision variable:
    - shared_daily_qik shape = (8,)
    - one value per within-day block
    - expanded into daily_qik shape = (2, 8) by copying to both classes
    """
    if max_value < 0:
        raise ValueError("max_value must be >= 0.")

    if timetable is None:
        shared_free_blocks = np.ones(8, dtype=int)
    else:
        shared_free_blocks = _shared_daily_feasible_blocks(timetable)

    free_block_indices = list(np.where(shared_free_blocks == 1)[0])

    for values in product(range(max_value + 1), repeat=len(free_block_indices)):
        shared_daily_qik = np.zeros(8, dtype=int)
        for block_idx, value in zip(free_block_indices, values):
            shared_daily_qik[block_idx] = value

        daily_qik = np.tile(shared_daily_qik, (2, 1))
        yield daily_qik


def policy_table(policy_indices):
    rows = []
    for i in policy_indices:
        cand = candidate_obj(i)
        rows.append(
            {
                "system": i,
                "timetable": ACTIVE_TIMETABLE,
                "policy": cand.name,
                "daily_qik": cand.daily_qik.tolist(),
            }
        )
    return pd.DataFrame(rows)


def subset_result_table(subset, subset_df):
    kept_df = policy_table(subset)
    mean_df = subset_df[["system", "mean_H_subset"]].copy()
    out = kept_df.merge(mean_df, on="system", how="left")
    return out.sort_values(["mean_H_subset", "system"], ascending=[True, True], ignore_index=True)


def _set_active_search(search_timetable):
    global ACTIVE_TIMETABLE, POLICIES
    ACTIVE_TIMETABLE = str(search_timetable).upper()
    POLICIES = _build_candidate_policies(ACTIVE_TIMETABLE)


def _print_df_preview(df, max_rows=MAX_PREVIEW_ROWS):

    print(df.head(max_rows).to_string(index=False))
    print(f"... showing first {max_rows} of {len(df)} rows")


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
                    "mean_H_final": float(MySim(best_system, n=1, seed=seed)),
                }
            ]
        )
        return {"Best": best_system, "n": 1, "Elim": np.zeros(1), "summary": summary_df}

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
def _run_active_search():


    num_systems = len(POLICIES)
    timetable, _ = _resolve_search_components(ACTIVE_TIMETABLE)

    print("Candidate policy set")
    print(f"Timetable = {ACTIVE_TIMETABLE}")
    print(f"shared_daily_qik_across_classes = {SHARE_DAILY_QIK_ACROSS_CLASSES}")
    if SHARE_DAILY_QIK_ACROSS_CLASSES:
        shared_free_blocks = _shared_daily_feasible_blocks(timetable)
        print(f"shared free daily blocks = {int(shared_free_blocks.sum())}")
    print(f"Number of candidate systems = {num_systems}")
    _print_df_preview(policy_table(range(1, num_systems + 1)))

    subset, subset_df = subset_crn(
        k=num_systems,
        alpha=SUBSET_ALPHA,
        n=SUBSET_INITIAL_REPS,
        seed=BASE_SEED
    )

    print("\nSubset selection with CRN")
    print(f"subset_reps = {SUBSET_INITIAL_REPS}")
    _print_df_preview(subset_df)

    print("\nPolicies kept after subset screening")
    kept_subset_df = subset_result_table(subset, subset_df)
    kept_subset_df = kept_subset_df.head(10).copy()
    subset = kept_subset_df["system"].astype(int).tolist()

    print(f"Number of systems kept after subset screening = {len(subset)}")
    print(kept_subset_df.to_string(index=False))

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
    best_mean_h = float(
        result["summary"].loc[result["summary"]["system"] == best_system, "mean_H_final"].iloc[0]
    )

    print("\nFinal KN with CRN result")
    _print_df_preview(result["summary"])
    print(f"Best system = {best_system}")
    print(f"Total samples used = {result['n']}")

    print("\nSelected best policy")
    _print_df_preview(policy_table([best_system]))

    print("\nSelected best policy weekly Qik")
    print(
        pd.DataFrame(
            best_candidate.weekly_qik,
            index=["Interventional", "Angiography"],
            columns=[f"block_{k}" for k in range(40)],
        ).to_string()
    )

    return {
        "timetable": ACTIVE_TIMETABLE,
        "best_system": best_system,
        "best_policy_name": best_candidate.name,
        "best_mean_H": best_mean_h,
    }



def main():
    all_results = []

    for search_timetable in _resolve_search_timetables():
        print("\n" + "=" * 80)
        _set_active_search(search_timetable)
        all_results.append(_run_active_search())

    if len(all_results) == 1:
        return all_results

    comparison_df = pd.DataFrame(
        [
            {
                "timetable": item["timetable"],
                "best_system": item["best_system"],
                "policy": item["best_policy_name"],
                "best_mean_H": item["best_mean_H"],
            }
            for item in all_results
        ]
    ).sort_values("best_mean_H", ascending=True, ignore_index=True)

    print("\n" + "=" * 80)
    print("Overall comparison across timetables")
    print(comparison_df.to_string(index=False))

    best_timetable = str(comparison_df.iloc[0]["timetable"])
    overall_best = next(item for item in all_results if item["timetable"] == best_timetable)
    print(f"\nOverall best timetable = {overall_best['timetable']}")
    print(f"Overall best system = {overall_best['best_system']}")
    print(f"Overall best policy = {overall_best['best_policy_name']}")
    print(f"Overall best mean H = {overall_best['best_mean_H']:.4f}")

    return all_results


if __name__ == "__main__":
    main()
