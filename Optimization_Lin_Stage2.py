from __future__ import annotations
import time
"""
Stage-II block-schedule search adapted from Lin et al. (2017).

This repository already fixes the stage-I tactical decision as a feasible
weekly timetable mask (`R1` or `R2`). This file implements the stage-II
search around that fixed timetable:

1. Build an initial weekly `Qik` schedule for each patient class.
2. Evaluate the incumbent schedule with simulation replications.
3. Use recent simulation statistics to estimate which classes and time
   blocks have the largest undesirable impact.
4. Probabilistically move a small pool of appointment slots from worse
   blocks to better blocks.
5. Keep a memory of visited schedules and update the best schedule only
   when a one-sided statistical test shows improvement.

Notes on the adaptation
-----------------------
- The paper's stage II also considers daily service discipline. The current
  simulation model keeps the FCFS service rule, so this script optimizes the
  block appointment schedule only.
- The paper's "procedure selection" step collapses here because the current
  model has one downstream procedure room. The search therefore selects
  patient classes and time blocks directly.
- The schedule decision variable is the full weekly `Qik` matrix with shape
  `(2, 40)`, not the repeated-daily simplification used by the brute-force
  scripts.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import Policy_defined
from input_loader import DistributionSpec, LoadedIRInputs, load_all_ir_inputs
from optimization_common import (
    ARRIVAL_JSON_PATH,
    SERVICE_JSON_PATH,
    RAW_DATA_PATH,
    resolve_search_timetables as _resolve_search_list,
    resolve_timetable as _resolve_timetable,
    serialize_qik as _serialize_qik,
    distribution_mean as _distribution_mean,
    load_common_inputs,
    make_policy_name,
)
from simulation_model import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    IROutpatientSchedulingSim,
    apply_feasibility_to_qik,
    compute_nonworking_overlap,
    policy_from_qik,
    qik_to_dataframe,
    run_replications,
)


MINUTES_PER_HOUR = 60.0
MINUTES_PER_DAY = 24.0 * 60.0

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "optimization_lin_stage2_outputs"

# ---------------------------------------------------------------------------
# Default initial points from SubsetKN best results (best known H).
# ---------------------------------------------------------------------------
_SUBSETKN_R1_DAILY = [0, 2, 0, 1, 2, 1, 2, 1]   # best KN for R1, H≈0.95
_SUBSETKN_R2_DAILY = [0, 2, 0, 2, 2, 1, 2, 0]   # best KN for R2, H≈1.05

DEFAULT_INITIAL_QIKS: dict[str, np.ndarray] = {
    "R1": np.array([_SUBSETKN_R1_DAILY * 5, _SUBSETKN_R1_DAILY * 5], dtype=int),
    "R2": np.array([_SUBSETKN_R2_DAILY * 5, _SUBSETKN_R2_DAILY * 5], dtype=int),
}


@dataclass
class Stage2Config:
    total_run_length: int = 200
    num_weeks: int = 180 # run length minus warmup
    warmup_weeks: int = 20
    base_seed: int = 123

    min_eval_reps: int = 3
    max_eval_reps: int = 6
    rel_half_width_epsilon: float = 0.10
    improvement_alpha: float = 0.10

    demand_buffer: float = 1.25
    max_q_per_block: int = 3
    initialization_mode: str = "all_ones"

    initial_psize: int = 1
    initial_pmax_size: int = 3
    max_pool_size: int = 6
    max_iterations: int = 20
    restart_after: int = 3
    max_proposal_attempts: int = 200

    block_time_bias: float = 0.40
    remove_idle_bonus: float = 0.35
    inverse_impact_floor: float = 0.05
    unscheduled_penalty: float = 0.25

    final_eval_reps: int = 100


@dataclass
class ScheduleEvaluation:
    qik: np.ndarray
    rep_df: pd.DataFrame
    mean_h: float
    rel_half_width: float
    class_impact: np.ndarray
    block_badness: np.ndarray
    block_usage_ratio: np.ndarray
    total_replications: int


def _procedure_mean_minutes(loaded_inputs: LoadedIRInputs) -> float:
    means = [_distribution_mean(spec) for spec in loaded_inputs.procedure_distributions.values()]
    if not means:
        raise ValueError("No procedure distributions were loaded.")
    return float(np.mean(means))


def expected_weekly_arrivals(loaded_inputs: LoadedIRInputs) -> np.ndarray:
    arrival = loaded_inputs.arrival_inputs

    interventional_mean = float(arrival.interventional_lambda_hat.sum()) * 5.0

    fit = arrival.angiography_pln_fit
    theta_mean = (1.0 - float(fit["p_zero"])) * math.exp(
        float(fit["mu"]) + 0.5 * float(fit["sigma"]) ** 2
    )
    angiography_mean = float(arrival.angiography_lambda_hat.sum()) * 5.0 * theta_mean

    return np.array([interventional_mean, angiography_mean], dtype=float)


def target_weekly_class_totals(
    timetable,
    loaded_inputs: LoadedIRInputs,
    config: Stage2Config,
) -> np.ndarray:
    expected = expected_weekly_arrivals(loaded_inputs)
    totals = np.ceil(expected * float(config.demand_buffer)).astype(int)
    totals = np.maximum(totals, 1)

    feasible_counts = timetable.feasible_qik.sum(axis=1).astype(int)
    if np.any(totals > feasible_counts * int(config.max_q_per_block)):
        raise ValueError(
            "Target weekly appointments exceed max_q_per_block capacity for the selected timetable."
        )

    return totals


def build_initial_qik(
    timetable,
    class_totals: np.ndarray,
    max_q_per_block: int,
    initialization_mode: str = "all_ones",
) -> np.ndarray:
    mode = str(initialization_mode).strip().lower()

    if mode == "all_ones":
        qik = np.ones((len(CLASS_NAMES), 40), dtype=int)
        return apply_feasibility_to_qik(qik, timetable)

    if mode != "demand_balanced":
        raise ValueError("initialization_mode must be 'all_ones' or 'demand_balanced'.")

    qik = np.zeros((len(CLASS_NAMES), 40), dtype=int)

    for class_idx in range(len(CLASS_NAMES)):
        feasible_blocks = np.where(np.asarray(timetable.feasible_qik[class_idx], dtype=int) == 1)[0]
        if class_totals[class_idx] > 0 and feasible_blocks.size == 0:
            raise ValueError(f"No feasible blocks available for class index {class_idx}.")

        for _ in range(int(class_totals[class_idx])):
            load_candidates = [
                block
                for block in feasible_blocks
                if qik[class_idx, block] < int(max_q_per_block)
            ]
            if not load_candidates:
                raise ValueError(
                    f"Unable to assign all initial appointments for class index {class_idx}."
                )
            chosen_block = min(load_candidates, key=lambda block: (qik[class_idx, block], block))
            qik[class_idx, chosen_block] += 1

    return apply_feasibility_to_qik(qik, timetable)


def _relative_half_width(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return math.inf

    mean_value = float(values.mean())
    if math.isclose(mean_value, 0.0, abs_tol=1e-12):
        return math.inf

    std_value = float(values.std(ddof=1))
    if math.isclose(std_value, 0.0, abs_tol=1e-12):
        return 0.0

    t_value = float(stats.t.ppf(0.975, df=n - 1))
    half_width = t_value * std_value / math.sqrt(n)
    return float(abs(half_width / mean_value))


def _paired_patient_impact_table(
    patients_df: pd.DataFrame,
    num_weeks: int,
    procedure_mean_minutes: float,
    unscheduled_penalty: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    if patients_df.empty:
        empty_df = pd.DataFrame(columns=["category", "weekly_block_index", "impact"])
        return empty_df, np.zeros(len(CLASS_NAMES), dtype=float)

    class_impact = np.zeros(len(CLASS_NAMES), dtype=float)

    unscheduled_df = patients_df[patients_df["scheduled_time"].isna()].copy()
    if not unscheduled_df.empty:
        for class_name, group in unscheduled_df.groupby("category"):
            class_idx = CLASS_TO_INDEX[str(class_name)]
            class_impact[class_idx] += float(len(group)) * float(unscheduled_penalty)

    scheduled_df = patients_df[patients_df["scheduled_time"].notna()].copy()
    if scheduled_df.empty:
        empty_df = pd.DataFrame(columns=["category", "weekly_block_index", "impact"])
        return empty_df, class_impact

    overtime_values = []
    for row in scheduled_df.itertuples(index=False):
        _, _, total_overlap = compute_nonworking_overlap(
            start_time=float(row.actual_proc_start),
            end_time=float(row.actual_proc_end),
            measurement_start_time=0.0,
        )
        overtime_values.append(total_overlap)

    scheduled_df["patient_overtime_min"] = overtime_values
    scheduled_df["queue_proxy"] = (
        scheduled_df["waiting_room_wait"].fillna(0.0) / max(float(procedure_mean_minutes), 1.0)
    )
    scheduled_df["impact"] = (
        0.6 * ((scheduled_df["total_wait_to_proc_start"].fillna(0.0) / MINUTES_PER_DAY) / 28.0)
        + 0.2 * ((scheduled_df["patient_overtime_min"] / MINUTES_PER_HOUR) / (2.5 * num_weeks))
        + 0.2 * (scheduled_df["queue_proxy"] / 2)
    )

    for class_name, group in scheduled_df.groupby("category"):
        class_idx = CLASS_TO_INDEX[str(class_name)]
        class_impact[class_idx] += float(group["impact"].sum())

    return scheduled_df[["category", "weekly_block_index", "impact"]].copy(), class_impact


def evaluate_schedule(
    qik: np.ndarray,
    timetable,
    loaded_inputs: LoadedIRInputs,
    config: Stage2Config,
    base_seed: int,
) -> ScheduleEvaluation:
    qik = apply_feasibility_to_qik(np.asarray(qik, dtype=int), timetable)
    procedure_mean_minutes = _procedure_mean_minutes(loaded_inputs)

    summary_rows: list[pd.DataFrame] = []
    block_impact_sum = np.zeros((len(CLASS_NAMES), 40), dtype=float)
    block_booked_count = np.zeros((len(CLASS_NAMES), 40), dtype=float)
    class_impact = np.zeros(len(CLASS_NAMES), dtype=float)

    policy = policy_from_qik(qik, timetable)

    for rep in range(int(config.max_eval_reps)):
        model = IROutpatientSchedulingSim(
            num_weeks=config.num_weeks,
            loaded_inputs=loaded_inputs,
            policy=policy,
            seed=base_seed + rep,
            warmup_weeks=config.warmup_weeks,
        )
        summary_df, patients_df, bookings_df = model.run()
        out = summary_df.copy()
        out.insert(0, "replication", rep + 1)
        summary_rows.append(out)

        patient_impact_df, rep_class_impact = _paired_patient_impact_table(
            patients_df=patients_df,
            num_weeks=config.num_weeks,
            procedure_mean_minutes=procedure_mean_minutes,
            unscheduled_penalty=config.unscheduled_penalty,
        )
        class_impact += rep_class_impact

        if not patient_impact_df.empty:
            grouped_impact = (
                patient_impact_df.groupby(["category", "weekly_block_index"])["impact"]
                .sum()
                .reset_index()
            )
            grouped_counts = (
                patient_impact_df.groupby(["category", "weekly_block_index"])
                .size()
                .reset_index(name="booked_count")
            )

            for row in grouped_impact.itertuples(index=False):
                class_idx = CLASS_TO_INDEX[str(row.category)]
                block_idx = int(row.weekly_block_index)
                block_impact_sum[class_idx, block_idx] += float(row.impact)

            for row in grouped_counts.itertuples(index=False):
                class_idx = CLASS_TO_INDEX[str(row.category)]
                block_idx = int(row.weekly_block_index)
                block_booked_count[class_idx, block_idx] += float(row.booked_count)

        rep_df = pd.concat(summary_rows, ignore_index=True)
        if rep + 1 >= int(config.min_eval_reps):
            rel_half_width = _relative_half_width(rep_df["H"].to_numpy(dtype=float))
            if rel_half_width <= float(config.rel_half_width_epsilon):
                break

    rep_df = pd.concat(summary_rows, ignore_index=True)
    mean_h = float(rep_df["H"].mean())
    rel_half_width = _relative_half_width(rep_df["H"].to_numpy(dtype=float))

    block_badness = np.zeros((len(CLASS_NAMES), 40), dtype=float)
    block_usage_ratio = np.zeros((len(CLASS_NAMES), 40), dtype=float)

    for class_idx in range(len(CLASS_NAMES)):
        feasible_blocks = np.where(np.asarray(timetable.feasible_qik[class_idx], dtype=int) == 1)[0]
        if feasible_blocks.size == 0:
            continue

        if feasible_blocks.size == 1:
            rank_fraction = {int(feasible_blocks[0]): 0.0}
        else:
            rank_fraction = {
                int(block): position / float(feasible_blocks.size - 1)
                for position, block in enumerate(feasible_blocks)
            }

        for block_idx in feasible_blocks:
            booked = float(block_booked_count[class_idx, block_idx])
            mean_impact = float(block_impact_sum[class_idx, block_idx] / booked) if booked > 0 else 0.0

            if qik[class_idx, block_idx] > 0:
                denom = float(qik[class_idx, block_idx] * config.num_weeks * len(rep_df))
                usage_ratio = booked / denom if denom > 0 else 0.0
            else:
                usage_ratio = 0.0

            idle_bonus = (
                float(config.remove_idle_bonus) * max(0.0, 1.0 - usage_ratio)
                if qik[class_idx, block_idx] > 0
                else 0.0
            )

            block_usage_ratio[class_idx, block_idx] = usage_ratio
            block_badness[class_idx, block_idx] = (
                mean_impact
                + float(config.block_time_bias) * float(rank_fraction[int(block_idx)])
                + idle_bonus
            )

    return ScheduleEvaluation(
        qik=qik,
        rep_df=rep_df,
        mean_h=mean_h,
        rel_half_width=rel_half_width,
        class_impact=class_impact,
        block_badness=block_badness,
        block_usage_ratio=block_usage_ratio,
        total_replications=len(rep_df),
    )


def _weighted_choice(
    items: list[int],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> int:
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if len(items) == 0:
        raise ValueError("Cannot sample from an empty item list.")
    if float(weights.sum()) <= 0.0:
        return int(rng.choice(items))
    probabilities = weights / float(weights.sum())
    return int(rng.choice(items, p=probabilities))


def _movable_classes(
    qik: np.ndarray,
    timetable,
    config: Stage2Config,
) -> list[int]:
    out = []
    for class_idx in range(len(CLASS_NAMES)):
        feasible_blocks = np.where(np.asarray(timetable.feasible_qik[class_idx], dtype=int) == 1)[0]
        removable = [block for block in feasible_blocks if qik[class_idx, block] > 0]
        addable = [block for block in feasible_blocks if qik[class_idx, block] < int(config.max_q_per_block)]
        if removable and len(feasible_blocks) >= 2 and (len(addable) >= 2 or any(block not in removable for block in addable)):
            out.append(class_idx)
    return out


def propose_new_schedule(
    current_eval: ScheduleEvaluation,
    timetable,
    config: Stage2Config,
    psize: int,
    memory: set[str],
    rng: np.random.Generator,
) -> np.ndarray | None:
    for _ in range(int(config.max_proposal_attempts)):
        candidate = np.asarray(current_eval.qik, dtype=int).copy()
        changed = False

        for _ in range(int(psize)):
            movable_classes = _movable_classes(candidate, timetable, config)
            if not movable_classes:
                break

            class_weights = np.asarray(
                [
                    current_eval.class_impact[class_idx] + 1e-6
                    for class_idx in movable_classes
                ],
                dtype=float,
            )
            class_idx = _weighted_choice(movable_classes, class_weights, rng)

            feasible_blocks = np.where(np.asarray(timetable.feasible_qik[class_idx], dtype=int) == 1)[0]
            removable_blocks = [block for block in feasible_blocks if candidate[class_idx, block] > 0]
            remove_weights = np.asarray(
                [
                    candidate[class_idx, block] * (current_eval.block_badness[class_idx, block] + 1e-6)
                    for block in removable_blocks
                ],
                dtype=float,
            )
            from_block = _weighted_choice(removable_blocks, remove_weights, rng)

            addable_blocks = [
                block
                for block in feasible_blocks
                if block != from_block and candidate[class_idx, block] < int(config.max_q_per_block)
            ]
            if not addable_blocks:
                continue

            add_weights = np.asarray(
                [
                    1.0 / (float(config.inverse_impact_floor) + current_eval.block_badness[class_idx, block])
                    for block in addable_blocks
                ],
                dtype=float,
            )
            to_block = _weighted_choice(addable_blocks, add_weights, rng)

            candidate[class_idx, from_block] -= 1
            candidate[class_idx, to_block] += 1
            changed = True

        if not changed:
            continue

        signature = _serialize_qik(candidate)
        if signature not in memory:
            return apply_feasibility_to_qik(candidate, timetable)

    return None


def one_sided_welch_pvalue_less(sample_x: np.ndarray, sample_y: np.ndarray) -> float:
    sample_x = np.asarray(sample_x, dtype=float)
    sample_y = np.asarray(sample_y, dtype=float)

    if sample_x.size == 0 or sample_y.size == 0:
        return 1.0

    mean_x = float(sample_x.mean())
    mean_y = float(sample_y.mean())

    if sample_x.size < 2 or sample_y.size < 2:
        return 0.0 if mean_x < mean_y else 1.0

    var_x = float(sample_x.var(ddof=1))
    var_y = float(sample_y.var(ddof=1))
    se2 = var_x / sample_x.size + var_y / sample_y.size

    if math.isclose(se2, 0.0, abs_tol=1e-12):
        return 0.0 if mean_x < mean_y else 1.0

    t_stat = (mean_x - mean_y) / math.sqrt(se2)
    numerator = se2 ** 2
    denominator = 0.0
    if sample_x.size > 1:
        denominator += (var_x / sample_x.size) ** 2 / (sample_x.size - 1)
    if sample_y.size > 1:
        denominator += (var_y / sample_y.size) ** 2 / (sample_y.size - 1)

    if math.isclose(denominator, 0.0, abs_tol=1e-12):
        return 0.0 if mean_x < mean_y else 1.0

    df = numerator / denominator
    return float(stats.t.cdf(t_stat, df=df))


def is_statistically_better(
    candidate_eval: ScheduleEvaluation,
    best_eval: ScheduleEvaluation,
    alpha: float,
) -> tuple[bool, float]:
    candidate_values = candidate_eval.rep_df["H"].to_numpy(dtype=float)
    best_values = best_eval.rep_df["H"].to_numpy(dtype=float)
    p_value = one_sided_welch_pvalue_less(candidate_values, best_values)

    improved = (
        candidate_eval.mean_h < best_eval.mean_h
        and p_value < float(alpha)
    )
    return improved, p_value


def _ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def optimize_timetable(
    search_timetable: str,
    loaded_inputs: LoadedIRInputs,
    config: Stage2Config,
    custom_initial_qik: np.ndarray | None = None,
) -> dict:
    timetable = _resolve_timetable(search_timetable)
    timetable.name = str(search_timetable).upper()

    if custom_initial_qik is not None:
        initial_qik = apply_feasibility_to_qik(
            np.asarray(custom_initial_qik, dtype=int), timetable
        )
    elif timetable.name in DEFAULT_INITIAL_QIKS:
        initial_qik = apply_feasibility_to_qik(
            DEFAULT_INITIAL_QIKS[timetable.name].copy(), timetable
        )
    else:
        initial_qik = build_initial_qik(
            timetable=timetable,
            class_totals=target_weekly_class_totals(timetable, loaded_inputs, config),
            max_q_per_block=config.max_q_per_block,
            initialization_mode=config.initialization_mode,
        )
    class_totals = initial_qik.sum(axis=1).astype(int)

    search_seed = int(config.base_seed)
    rng = np.random.default_rng(search_seed)

    best_eval = evaluate_schedule(
        qik=initial_qik,
        timetable=timetable,
        loaded_inputs=loaded_inputs,
        config=config,
        base_seed=search_seed,
    )
    current_eval = best_eval

    memory = {_serialize_qik(initial_qik)}
    psize = int(config.initial_psize)
    pmax_size = int(config.initial_pmax_size)
    since_restart = 0

    history_rows = [
        {
            "iteration": 0,
            "timetable": timetable.name,
            "psize": psize,
            "pmax_size": pmax_size,
            "incumbent_mean_H": current_eval.mean_h,
            "best_mean_H": best_eval.mean_h,
            "rel_half_width": current_eval.rel_half_width,
            "eval_replications": current_eval.total_replications,
            "improved_best": True,
            "p_value_vs_best": 0.0,
            "schedule_json": _serialize_qik(current_eval.qik),
        }
    ]
    # Stopping Criteria when the nax iteration is achieved 
    # or when the neighborhood search can no longer generate and accept improving candidate schedules.
    iteration = 0
    while iteration < int(config.max_iterations) and psize <= pmax_size:
        iteration += 1

        candidate_qik = propose_new_schedule(
            current_eval=current_eval,
            timetable=timetable,
            config=config,
            psize=psize,
            memory=memory,
            rng=rng,
        )
        if candidate_qik is None:
            break

        candidate_signature = _serialize_qik(candidate_qik)
        memory.add(candidate_signature)

        candidate_eval = evaluate_schedule(
            qik=candidate_qik,
            timetable=timetable,
            loaded_inputs=loaded_inputs,
            config=config,
            base_seed=search_seed,
        )

        current_eval = candidate_eval
        improved, p_value = is_statistically_better(
            candidate_eval=candidate_eval,
            best_eval=best_eval,
            alpha=config.improvement_alpha,
        )

        if improved:
            best_eval = candidate_eval
            pmax_size = min(pmax_size + 1, int(config.max_pool_size))
            psize = int(config.initial_psize)
            since_restart = 0
        else:
            since_restart += 1
            if since_restart >= int(config.restart_after):
                current_eval = best_eval
                psize += 1
                since_restart = 0

        history_rows.append(
            {
                "iteration": iteration,
                "timetable": timetable.name,
                "psize": psize,
                "pmax_size": pmax_size,
                "incumbent_mean_H": candidate_eval.mean_h,
                "best_mean_H": best_eval.mean_h,
                "rel_half_width": candidate_eval.rel_half_width,
                "eval_replications": candidate_eval.total_replications,
                "improved_best": improved,
                "p_value_vs_best": p_value,
                "schedule_json": candidate_signature,
            }
        )

    final_rep_df = run_replications(
        num_replications=int(config.final_eval_reps),
        num_weeks=int(config.num_weeks),
        loaded_inputs=loaded_inputs,
        policy=policy_from_qik(best_eval.qik, timetable),
        base_seed=search_seed + 20_000,
        # The final validation seed is changed so the final performance is independent of the 
        # random sample that was used to choose the policy.
        warmup_weeks=int(config.warmup_weeks),
    )

    history_df = pd.DataFrame(history_rows)
    initial_qik_df = qik_to_dataframe(initial_qik)
    initial_qik_df.insert(0, "timetable", timetable.name)
    best_qik_df = qik_to_dataframe(best_eval.qik)
    best_qik_df.insert(0, "timetable", timetable.name)

    policy_name = make_policy_name(timetable.name, "Lin", best_eval.qik)

    return {
        "timetable": timetable.name,
        "policy_name": policy_name,
        "class_totals": class_totals,
        "initial_qik": initial_qik,
        "initial_qik_df": initial_qik_df,
        "best_eval": best_eval,
        "history_df": history_df,
        "best_qik_df": best_qik_df,
        "final_rep_df": final_rep_df,
        "search_seed": search_seed,
    }


def save_result(result: dict, config: Stage2Config) -> dict:
    output_dir = _ensure_output_dir()
    timetable = str(result["timetable"]).lower()

    history_path = output_dir / f"{timetable}_history.csv"
    initial_qik_path = output_dir / f"{timetable}_initial_qik.csv"
    qik_path = output_dir / f"{timetable}_best_qik.csv"
    final_eval_path = output_dir / f"{timetable}_final_eval.csv"
    summary_path = output_dir / f"{timetable}_summary.json"

    result["history_df"].to_csv(history_path, index=False)
    result["initial_qik_df"].to_csv(initial_qik_path, index=False)
    result["best_qik_df"].to_csv(qik_path, index=False)
    result["final_rep_df"].to_csv(final_eval_path, index=False)

    best_eval = result["best_eval"]
    summary_payload = {
        "timetable": result["timetable"],
        "policy_name": result["policy_name"],
        "search_seed": int(result["search_seed"]),
        "class_totals": np.asarray(result["class_totals"], dtype=int).tolist(),
        "best_mean_H_search": float(best_eval.mean_h),
        "best_rel_half_width": float(best_eval.rel_half_width),
        "best_eval_replications": int(best_eval.total_replications),
        "final_eval_reps": int(config.final_eval_reps),
        "final_eval_mean_H": float(result["final_rep_df"]["H"].mean()),
        "final_eval_std_H": float(result["final_rep_df"]["H"].std(ddof=1))
        if len(result["final_rep_df"]) > 1
        else 0.0,
        "final_eval_mean_unscheduled": float(result["final_rep_df"]["unscheduled_count"].mean()),
        "best_qik_json": np.asarray(best_eval.qik, dtype=int).tolist(),
        "total_run_time": result.get("total_run_time", 0.0),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    return {
        "history_path": history_path,
        "initial_qik_path": initial_qik_path,
        "qik_path": qik_path,
        "final_eval_path": final_eval_path,
        "summary_path": summary_path,
    }


def _summary_row(result: dict) -> dict:
    final_rep_df = result["final_rep_df"]
    best_eval = result["best_eval"]
    return {
        "timetable": result["timetable"],
        "policy_name": result["policy_name"],
        "class_totals_json": json.dumps(np.asarray(result["class_totals"], dtype=int).tolist()),
        "search_mean_H": float(best_eval.mean_h),
        "search_rel_half_width": float(best_eval.rel_half_width),
        "search_eval_reps": int(best_eval.total_replications),
        "final_mean_H": float(final_rep_df["H"].mean()),
        "final_std_H": float(final_rep_df["H"].std(ddof=1)) if len(final_rep_df) > 1 else 0.0,
        "final_mean_Z1_days": float(final_rep_df["Z1_wait_time"].mean()),
        "final_mean_Z2_hours": float(final_rep_df["Z2_overtime"].mean()),
        "final_mean_Z3": float(final_rep_df["Z3_congestion"].mean()),
        "final_mean_unscheduled": float(final_rep_df["unscheduled_count"].mean()),
        "best_qik_json": _serialize_qik(best_eval.qik),
        "total_run_time": result.get("total_run_time", 0.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Lin et al. stage-II schedule search.")
    parser.add_argument("--timetable", default="both", help="R1, R2, or both")
    parser.add_argument("--num-weeks", type=int, default=180)
    parser.add_argument("--warmup-weeks", type=int, default=20)
    parser.add_argument("--min-eval-reps", type=int, default=3)
    parser.add_argument("--max-eval-reps", type=int, default=6)
    parser.add_argument("--final-eval-reps", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--demand-buffer", type=float, default=1.25)
    parser.add_argument("--max-q-per-block", type=int, default=3)
    parser.add_argument("--initialization-mode", default="all_ones", help="all_ones or demand_balanced")
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--restart-after", type=int, default=3)
    parser.add_argument(
        "--initial-qik-json", default=None,
        help='JSON string for a custom 2x40 initial Qik, e.g. \'[[0,1,...],[0,2,...]]\'',
    )
    return parser.parse_args()


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    config = Stage2Config(
        num_weeks=args.num_weeks,
        warmup_weeks=args.warmup_weeks,
        min_eval_reps=args.min_eval_reps,
        max_eval_reps=args.max_eval_reps,
        final_eval_reps=args.final_eval_reps,
        max_iterations=args.iterations,
        demand_buffer=args.demand_buffer,
        max_q_per_block=args.max_q_per_block,
        initialization_mode=args.initialization_mode,
        base_seed=args.base_seed,
        restart_after=args.restart_after,
    )

    loaded_inputs = load_common_inputs()

    # Parse optional custom initial Qik from CLI
    cli_initial_qik = None
    if args.initial_qik_json is not None:
        cli_initial_qik = np.asarray(json.loads(args.initial_qik_json), dtype=int)

    all_results = []
    for search_timetable in _resolve_search_list(args.timetable):
        print("\n" + "=" * 80)
        print(f"Running stage-II search for {search_timetable}")

        tt_start = time.perf_counter()
        result = optimize_timetable(
            search_timetable=search_timetable,
            loaded_inputs=loaded_inputs,
            config=config,
            custom_initial_qik=cli_initial_qik,
        )
        result["total_run_time"] = time.perf_counter() - tt_start
        paths = save_result(result, config)

        summary_row = _summary_row(result)
        all_results.append(summary_row)

        print(f"Policy name = {result['policy_name']}")
        print(f"Weekly class totals = {np.asarray(result['class_totals'], dtype=int).tolist()}")
        print(
            f"Search best mean H = {summary_row['search_mean_H']:.4f} "
            f"(rel half-width {summary_row['search_rel_half_width']:.4f}, "
            f"reps {summary_row['search_eval_reps']})"
        )
        print(
            f"Final evaluation mean H = {summary_row['final_mean_H']:.4f} "
            f"+/- {summary_row['final_std_H']:.4f}"
        )
        print(f"  Z1={summary_row['final_mean_Z1_days']:.4f} days, "
              f"Z2={summary_row['final_mean_Z2_hours']:.4f} hrs, "
              f"Z3={summary_row['final_mean_Z3']:.2f}")
        print(f"Saved history to {paths['history_path']}")
        print(f"Saved initial Qik to {paths['initial_qik_path']}")
        print(f"Saved best Qik to {paths['qik_path']}")
        print(f"Saved final evaluation to {paths['final_eval_path']}")
        print(f"Saved summary to {paths['summary_path']}")
        print("\nInitial weekly Qik")
        print(result["initial_qik_df"].to_string(index=False))
        print("\nBest weekly Qik")
        print(result["best_qik_df"].to_string(index=False))

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"\nTotal running time: {elapsed:.6f} seconds")

    if all_results:
        summary_df = pd.DataFrame(all_results).sort_values(
            ["final_mean_H", "timetable"],
            ascending=[True, True],
            ignore_index=True,
        )
        summary_path = _ensure_output_dir() / "stage2_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print("\n" + "=" * 80)
        print("Overall stage-II summary")
        print(summary_df.to_string(index=False))
        print(f"Saved summary table to {summary_path}")


if __name__ == "__main__":
    main()
