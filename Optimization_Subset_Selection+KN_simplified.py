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
import time
from pathlib import Path
from dataclasses import dataclass
from functools import partial
from itertools import islice, product
import json

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

import Policy_defined
from input_loader import load_all_ir_inputs
from simulation_model import run_replications
from optimization_common import (
    ARRIVAL_JSON_PATH,
    SERVICE_JSON_PATH,
    RAW_DATA_PATH,
    resolve_search_timetables as _resolve_search_timetables_common,
    resolve_timetable_and_builders,
    serialize_qik as _serialize_qik,
    shared_daily_feasible_blocks as _shared_daily_feasible_blocks,
    load_common_inputs,
    make_policy_name,
)


# =========================================================
# USER SETTINGS
# =========================================================
TOTAL_RUN_LENGTH = 200
NUM_WEEKS = 180 # run length minus warmup
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
TOP_SUBSET_POLICIES = 10

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "optimization_subset_selection_kn_simplified_outputs5"



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
LOADED_INPUTS = load_common_inputs()


# =========================================================
# BUILD CANDIDATE POLICIES
# =========================================================
def _resolve_search_timetables():
    return _resolve_search_timetables_common(SEARCH_TIMETABLE)


def _resolve_search_components(search_timetable):
    timetable, build_policy_func, _ = resolve_timetable_and_builders(search_timetable)
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
                name=make_policy_name(search_timetable, "SubsetKN", weekly_qik),
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


def _progress_desc(stage: str) -> str:
    if ACTIVE_TIMETABLE is None:
        return stage
    return f"{ACTIVE_TIMETABLE} {stage}"



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


def _ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _save_csv(df: pd.DataFrame, filename: str) -> Path:
    output_path = _ensure_output_dir() / filename
    df.to_csv(output_path, index=False)
    return output_path



def _policy_export_table(policy_indices):
    rows = []
    for system_index in policy_indices:
        candidate = candidate_obj(system_index)
        rows.append(
            {
                "system": system_index,
                "timetable": ACTIVE_TIMETABLE,
                "policy": candidate.name,
                "daily_qik_json": _serialize_qik(candidate.daily_qik),
                "weekly_qik_json": _serialize_qik(candidate.weekly_qik),
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

    sample_columns = []
    for system_index in tqdm(
        indices,
        desc=_progress_desc("subset samples"),
        unit="policy",
    ):
        sample_columns.append(np.asarray(MySim(system_index, n=n, seed=seed), dtype=float))

    sample_matrix = np.column_stack(sample_columns)
    means = sample_matrix.mean(axis=0)

    t_value = stats.t.ppf(
        1.0 - alpha / max(len(indices) - 1, 1),
        df=max(n - 1, 1),
    )

    keep_indices = []

    screening_progress = tqdm(
        enumerate(indices),
        total=len(indices),
        desc=_progress_desc("subset screen"),
        unit="policy",
    )
    for col_i, system_i in screening_progress:
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

        screening_progress.set_postfix(kept=len(keep_indices), refresh=False)

    if not keep_indices:
        keep_indices = [indices[int(np.argmin(means))]]

    summary_df = pd.DataFrame(
        {
            "system": indices,
            "policy": [candidate_name(i) for i in indices],
            "mean_H_subset": means,
        }
    ).sort_values("mean_H_subset", ascending=True, ignore_index=True)
    summary_df.insert(0, "subset_rank", np.arange(1, len(summary_df) + 1))


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
                    "kn_rank": 1,
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
    for i in tqdm(
        range(k),
        desc=_progress_desc("KN initial samples"),
        unit="policy",
    ):
        system_index = indices[i]
        Yn0[i, :] = np.asarray(MySim(system_index, n=n0, seed=seed), dtype=float)

    Sigma = np.cov(Yn0, bias=False)

    eta = 0.5 * ((2 * alpha / (k - 1)) ** (-2 / (n0 - 1)) - 1)
    h2 = 2 * eta * (n0 - 1)

    Ysum = Yn0.sum(axis=1)
    r = n0
    ## I add tqdm for tracing the progress, as it takes too long to run the code
    with tqdm(
        total=k - 1,
        desc=_progress_desc("KN eliminations"),
        unit="policy",
    ) as elimination_progress:
        elimination_progress.set_postfix(replication=r, active=int(np.sum(Active)), refresh=False)

        while np.sum(Active) > 1:
            r += 1
            ATemp = Active.copy()
            active_before = int(np.sum(Active))

            # One extra CRN observation for each active system
            for i in II[Active]:
                system_index = indices[i - 1]
                Ysum[i - 1] += MySim(system_index, n=1, seed=seed + r)

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
            active_after = int(np.sum(Active))
            elimination_progress.update(active_before - active_after)
            elimination_progress.set_postfix(replication=r, active=active_after, refresh=False)

    best_local = int(II[Active][0])
    best_system = indices[best_local - 1]

    summary_df = pd.DataFrame(
        {
            "system": indices,
            "policy": [candidate_name(i) for i in indices],
            "mean_H_final": Ysum / r,
        }
    ).sort_values("mean_H_final", ascending=True, ignore_index=True)
    summary_df.insert(0, "kn_rank", np.arange(1, len(summary_df) + 1))

    return {"Best": best_system, "n": r, "Elim": Elim, "summary": summary_df}


def final_eval_table(policy_indices, subset_df, kn_df, num_replications, seed):
    subset_lookup = subset_df[["system", "subset_rank", "mean_H_subset"]].copy()
    kn_lookup = kn_df[["system", "kn_rank", "mean_H_final"]].copy()

    rows = []
    for system_index in tqdm(
        policy_indices,
        desc=_progress_desc("final eval"),
        unit="policy",
    ):
        candidate = candidate_obj(system_index)
        rep_df = run_replications(
            num_replications=num_replications,
            num_weeks=NUM_WEEKS,
            loaded_inputs=LOADED_INPUTS,
            policy=candidate.build_policy(),
            base_seed=int(seed),
            warmup_weeks=WARMUP_WEEKS,
        )
        h_values = rep_df["H"].to_numpy(dtype=float)
        rows.append(
            {
                "system": system_index,
                "policy": candidate.name,
                "mean_H_eval": float(h_values.mean()),
                "std_H_eval": float(h_values.std(ddof=1)) if len(h_values) > 1 else 0.0,
                "min_H_eval": float(h_values.min()),
                "max_H_eval": float(h_values.max()),
                "mean_Z1_eval": float(rep_df["Z1_wait_time"].mean()),
                "mean_Z2_eval": float(rep_df["Z2_overtime"].mean()),
                "mean_Z3_eval": float(rep_df["Z3_congestion"].mean()),
                "final_eval_reps": int(num_replications),
            }
        )

    final_df = pd.DataFrame(rows)
    final_df = final_df.merge(subset_lookup, on=["system"], how="left")
    final_df = final_df.merge(kn_lookup, on=["system"], how="left")
    final_df = final_df.merge(_policy_export_table(policy_indices), on=["system", "policy"], how="left")

    column_order = [
        "system",
        "timetable",
        "policy",
        "subset_rank",
        "kn_rank",
        "mean_H_subset",
        "mean_H_final",
        "mean_H_eval",
        "std_H_eval",
        "min_H_eval",
        "max_H_eval",
        "mean_Z1_eval",
        "mean_Z2_eval",
        "mean_Z3_eval",
        "final_eval_reps",
        "daily_qik_json",
        "weekly_qik_json",
    ]
    return final_df[column_order].sort_values(
        ["mean_H_eval", "system"],
        ascending=[True, True],
        ignore_index=True,
    )



# =========================================================
# MAIN
# =========================================================
def _run_active_search():
    run_start = time.perf_counter()

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
    kept_subset_df = kept_subset_df.head(TOP_SUBSET_POLICIES).copy()
    subset = kept_subset_df["system"].astype(int).tolist()

    print(f"Number of systems kept after subset screening = {len(subset)}")
    print(kept_subset_df.to_string(index=False))

    result = kn_crn(
        k=len(subset),
        alpha=KN_ALPHA,
        n0=KN_INITIAL_REPS,
        delta=KN_DELTA,
        seed=BASE_SEED,
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

    subset_csv_path = _save_csv(
        subset_df,
        f"{ACTIVE_TIMETABLE.lower()}_subset_summary.csv",
    )
    kn_csv_path = _save_csv(
        result["summary"],
        f"{ACTIVE_TIMETABLE.lower()}_kn_summary.csv",
    )

    final_eval_df = final_eval_table(
        policy_indices=subset,
        subset_df=subset_df,
        kn_df=result["summary"],
        num_replications=FINAL_EVAL_REPS,
        seed=BASE_SEED + 20_000,  
        # I change another seed here to produce the eval table, this is consistent with all other three algorithm
    )
    final_eval_df.insert(
        len(final_eval_df.columns) - 2,
        "selected_by_kn",
        final_eval_df["system"].eq(best_system),
    )
    final_csv_path = _save_csv(
        final_eval_df,
        f"{ACTIVE_TIMETABLE.lower()}_subset_policy_results.csv",
    )
    best_row = final_eval_df.loc[final_eval_df["system"] == best_system].iloc[0]
    best_eval_mean_h = float(best_row["mean_H_eval"])
    best_eval_mean_z1 = float(best_row["mean_Z1_eval"])
    best_eval_mean_z2 = float(best_row["mean_Z2_eval"])
    best_eval_mean_z3 = float(best_row["mean_Z3_eval"])

    print("\nFinal evaluation statistics for subset-selected policies")
    _print_df_preview(
        final_eval_df[
            [
                "system",
                "policy",
                "subset_rank",
                "kn_rank",
                "mean_H_eval",
                "std_H_eval",
                "min_H_eval",
                "max_H_eval",
                "selected_by_kn",
            ]
        ]
    )
    print(f"Saved subset summary CSV: {subset_csv_path}")
    print(f"Saved KN summary CSV: {kn_csv_path}")
    print(f"Saved final policy results CSV: {final_csv_path}")
    elapsed_seconds = time.perf_counter() - run_start
    print(f"Elapsed runtime for {ACTIVE_TIMETABLE}: {elapsed_seconds:.2f} seconds")

    return {
        "timetable": ACTIVE_TIMETABLE,
        "best_system": best_system,
        "best_policy_name": best_candidate.name,
        "best_mean_H": best_mean_h,
        "best_mean_H_eval": best_eval_mean_h,
        "best_mean_Z1_eval": best_eval_mean_z1,
        "best_mean_Z2_eval": best_eval_mean_z2,
        "best_mean_Z3_eval": best_eval_mean_z3,
        "final_results_df": final_eval_df,
        "subset_csv_path": str(subset_csv_path),
        "kn_csv_path": str(kn_csv_path),
        "final_csv_path": str(final_csv_path),
        "elapsed_seconds": elapsed_seconds,
    }



def main():
    overall_start = time.perf_counter()
    all_results = []

    for search_timetable in _resolve_search_timetables():
        print("\n" + "=" * 80)
        _set_active_search(search_timetable)
        all_results.append(_run_active_search())

    if len(all_results) == 1:
        total_elapsed_seconds = time.perf_counter() - overall_start
        print(f"Total runtime: {total_elapsed_seconds:.2f} seconds")
        return all_results

    comparison_df = pd.DataFrame(
        [
            {
                "timetable": item["timetable"],
                "best_system": item["best_system"],
                "policy": item["best_policy_name"],
                "best_mean_H_kn": item["best_mean_H"],
                "best_mean_H_eval": item["best_mean_H_eval"],
                "best_mean_Z1_eval": item["best_mean_Z1_eval"],
                "best_mean_Z2_eval": item["best_mean_Z2_eval"],
                "best_mean_Z3_eval": item["best_mean_Z3_eval"],
                "elapsed_seconds": item["elapsed_seconds"],
                "subset_csv_path": item["subset_csv_path"],
                "kn_csv_path": item["kn_csv_path"],
                "final_csv_path": item["final_csv_path"],
            }
            for item in all_results
        ]
    ).sort_values("best_mean_H_eval", ascending=True, ignore_index=True)

    print("\n" + "=" * 80)
    print("Overall comparison across timetables")
    print(comparison_df.to_string(index=False))

    comparison_csv_path = _save_csv(
        comparison_df,
        "overall_timetable_final_comparison.csv",
    )
    combined_final_csv_path = _save_csv(
        pd.concat([item["final_results_df"] for item in all_results], ignore_index=True),
        "R1_R2_timetables_policy_results_subset.csv",
    )

    best_timetable = str(comparison_df.iloc[0]["timetable"])
    overall_best = next(item for item in all_results if item["timetable"] == best_timetable)
    print(f"\nOverall best timetable = {overall_best['timetable']}")
    print(f"Overall best system = {overall_best['best_system']}")
    print(f"Overall best policy = {overall_best['best_policy_name']}")
    print(f"Overall best final-eval mean H = {overall_best['best_mean_H_eval']:.4f}")
    print(f"Saved overall comparison CSV: {comparison_csv_path}")
    print(f"Saved combined final policy results CSV: {combined_final_csv_path}")
    total_elapsed_seconds = time.perf_counter() - overall_start
    print(f"Total runtime: {total_elapsed_seconds:.2f} seconds")

    return all_results


if __name__ == "__main__":
    main()
