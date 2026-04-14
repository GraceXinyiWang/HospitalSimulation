from __future__ import annotations
import time
"""
Improved SAA for the hospital Qik optimization problem.

Key improvements over Optimization_SAA.py:
1. Wait objective uses per-patient expected wait in days (matching Z1).
2. Overtime constraint accounts for the lunch break (12-13).
3. Multiple rounding strategies from the LP relaxation are evaluated
   via short simulation runs to pick the best integer solution.
4. More SAA scenarios for better demand representation.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import Policy_defined
from input_loader import LoadedIRInputs, load_all_ir_inputs
from simulation_model import (
    BookingTimetable,
    MINUTES_PER_DAY,
    MINUTES_PER_HOUR,
    run_replications,
    sample_from_spec,
    weekly_block_metadata,
)
from optimization_common import (
    ARRIVAL_JSON_PATH,
    SERVICE_JSON_PATH,
    RAW_DATA_PATH,
    resolve_search_timetables,
    resolve_policy_builder,
    distribution_mean,
    daily_feasible_positions,
    shared_daily_feasible_blocks as _shared_daily_feasible_blocks_raw,
    full_week_feasible_positions,
    load_common_inputs,
    make_policy_name,
)


NUM_WEEKS = 180
WARMUP_WEEKS = 20
BASE_SEED = 123

SEARCH_TIMETABLE = "BOTH"
POLICY_SPACE = "full_week"
SHARE_DAILY_QIK_ACROSS_CLASSES = False
MAX_QIK_VALUE = 2

N_SAA = 50
SCREENING_REPLICATIONS = 5
VALIDATION_REPLICATIONS = 100
NUM_ROUNDING_CANDIDATES = 20
OUTPUT_DIR = Path("SAA2_output_folder")

# The procedure room is staffed 8-12 and 13-17 = 7 hours effective per day.
EFFECTIVE_STAFFED_MINUTES_PER_DAY = 7.0 * MINUTES_PER_HOUR


LOADED_INPUTS = load_common_inputs()


@dataclass
class SAA2Result:
    timetable: str
    policy_name: str
    weekly_qik: np.ndarray
    screening_mean_H: float
    validation_mean_H: float | None
    validation_std_H: float | None
    validation_mean_Z1: float | None
    validation_mean_Z2: float | None
    validation_mean_Z3: float | None


# =========================================================
# Feasible position helpers (reused from SAA)
# =========================================================
def _shared_daily_feasible_blocks(timetable: BookingTimetable):
    return list(np.where(_shared_daily_feasible_blocks_raw(timetable) == 1)[0])


def _decision_positions(timetable, policy_space, share_daily_qik_across_classes):
    policy_space = str(policy_space).strip().lower()
    if policy_space == "full_week":
        return full_week_feasible_positions(timetable), "weekly_qik"
    if share_daily_qik_across_classes:
        return _shared_daily_feasible_blocks(timetable), "daily_qik_shared"
    return daily_feasible_positions(timetable), "daily_qik"


def _vector_to_qik_input(x, timetable, policy_space, share_daily_qik_across_classes):
    x = np.asarray(x, dtype=float)
    positions, space_label = _decision_positions(
        timetable, policy_space, share_daily_qik_across_classes,
    )
    if space_label == "weekly_qik":
        qik = np.zeros((2, 40), dtype=float)
        for value, (i, k) in zip(x, positions):
            qik[i, k] = float(value)
        qik[timetable.feasible_qik == 0] = 0.0
        return qik
    qik = np.zeros((2, 8), dtype=float)
    if space_label == "daily_qik_shared":
        for value, k in zip(x, positions):
            qik[:, k] = float(value)
        return qik
    for value, (i, k) in zip(x, positions):
        qik[i, k] = float(value)
    return qik


def _qik_input_to_weekly(qik_input, timetable, policy_space):
    qik_input = np.asarray(qik_input, dtype=float)
    if str(policy_space).strip().lower() == "full_week":
        weekly_qik = qik_input.copy()
    else:
        weekly_qik = np.tile(qik_input, (1, 5))
    weekly_qik[timetable.feasible_qik == 0] = 0.0
    return weekly_qik


def _vector_to_weekly_qik(x, timetable, policy_space, share_daily_qik_across_classes):
    qik_input = _vector_to_qik_input(x, timetable, policy_space, share_daily_qik_across_classes)
    return _qik_input_to_weekly(qik_input, timetable, policy_space)


def _decision_basis_weekly_qik(timetable, policy_space, share_daily_qik_across_classes):
    positions, _ = _decision_positions(timetable, policy_space, share_daily_qik_across_classes)
    basis = np.zeros((len(positions), 2, 40), dtype=float)
    for pos in range(len(positions)):
        x = np.zeros(len(positions), dtype=float)
        x[pos] = 1.0
        basis[pos] = _vector_to_weekly_qik(x, timetable, policy_space, share_daily_qik_across_classes)
    return basis


# =========================================================
# Demand sampling (same as SAA)
# =========================================================
def _parse_bin_start_hours(bin_index, fallback_start_hour):
    hours = []
    for idx, label in enumerate(bin_index):
        text = str(label)
        match = re.match(r"\s*(\d{1,2})\s*:\s*00", text)
        hours.append(int(match.group(1)) if match else int(fallback_start_hour) + idx)
    return hours


def _piecewise_rate_per_hour(time_in_day_min, start_hours, rates):
    hour = int(time_in_day_min // MINUTES_PER_HOUR)
    for idx, start_hour in enumerate(start_hours):
        if start_hour <= hour < start_hour + 1:
            return float(rates[idx])
    return 0.0


def _draw_prep_time(category, loaded_inputs, rng):
    probs = loaded_inputs.prep_probabilities[category]
    labels = list(probs.keys())
    p = np.array([probs[label] for label in labels], dtype=float)
    p = p / p.sum()
    prep_type = str(rng.choice(labels, p=p))
    return max(0.0, sample_from_spec(loaded_inputs.prep_distributions[category][prep_type], rng))


def _build_horizon_block_table(total_sim_weeks, warmup_weeks):
    meta = weekly_block_metadata()
    rows = []
    for week in range(total_sim_weeks):
        for row in meta.itertuples(index=False):
            day_index = 7 * week + int(row.weekday)
            start_time = day_index * MINUTES_PER_DAY + int(row.start_hour) * MINUTES_PER_HOUR
            rows.append({
                "week": week,
                "weekday": int(row.weekday),
                "weekly_block_index": int(row.block_index),
                "start_time": float(start_time),
                "is_measured": week >= warmup_weeks,
            })
    df = pd.DataFrame(rows).sort_values("start_time", kind="stable").reset_index(drop=True)
    df["gap_to_next_days"] = (df["start_time"].shift(-1) - df["start_time"]) / MINUTES_PER_DAY
    df["gap_to_next_days"] = df["gap_to_next_days"].fillna(0.0)
    return df


def _sample_ready_demand_scenarios(n, seed, loaded_inputs, total_sim_weeks, warmup_weeks, block_table):
    block_starts = block_table["start_time"].to_numpy(dtype=float)
    num_blocks = len(block_starts)
    measurement_start_time = warmup_weeks * 7 * MINUTES_PER_DAY

    demand = np.zeros((n, 2, num_blocks), dtype=float)
    measured_arrivals = np.zeros(n, dtype=float)

    working_days = [7 * week + weekday for week in range(total_sim_weeks) for weekday in range(5)]
    arrival_inputs = loaded_inputs.arrival_inputs

    int_start_hours = _parse_bin_start_hours(
        arrival_inputs.interventional_lambda_hat.index, arrival_inputs.bin_start_hour,
    )
    int_rates = np.asarray(arrival_inputs.interventional_lambda_hat.values, dtype=float)
    int_lambda_max = float(int_rates.max()) if len(int_rates) > 0 else 0.0
    int_work_start = min(int_start_hours) * MINUTES_PER_HOUR
    int_work_end = (max(int_start_hours) + 1) * MINUTES_PER_HOUR

    ang_start_hours = _parse_bin_start_hours(
        arrival_inputs.angiography_lambda_hat.index, arrival_inputs.bin_start_hour,
    )
    ang_rates = np.asarray(arrival_inputs.angiography_lambda_hat.values, dtype=float)
    ang_fit = arrival_inputs.angiography_pln_fit

    for scenario in range(n):
        rng = np.random.default_rng(seed + scenario)
        for day_index in working_days:
            day_offset = day_index * MINUTES_PER_DAY

            if int_lambda_max > 0.0:
                t = day_offset + int_work_start
                while True:
                    t += float(rng.exponential(MINUTES_PER_HOUR / int_lambda_max))
                    if t >= day_offset + int_work_end:
                        break
                    rate_t = _piecewise_rate_per_hour(t - day_offset, int_start_hours, int_rates)
                    if rate_t > 0.0 and rng.random() <= rate_t / int_lambda_max:
                        if t >= measurement_start_time:
                            measured_arrivals[scenario] += 1.0
                        ready_time = t + _draw_prep_time("Interventional", loaded_inputs, rng)
                        block_idx = int(np.searchsorted(block_starts, ready_time, side="left"))
                        if block_idx < num_blocks:
                            demand[scenario, 0, block_idx] += 1.0

            theta = 0.0 if rng.random() < float(ang_fit["p_zero"]) else float(
                rng.lognormal(mean=float(ang_fit["mu"]), sigma=float(ang_fit["sigma"]))
            )
            counts = rng.poisson(theta * ang_rates)
            for col_idx, count in enumerate(counts):
                hour_start = ang_start_hours[col_idx] * MINUTES_PER_HOUR
                for _ in range(int(count)):
                    arrival_time = day_offset + hour_start + float(rng.uniform(0.0, MINUTES_PER_HOUR))
                    if arrival_time >= measurement_start_time:
                        measured_arrivals[scenario] += 1.0
                    ready_time = arrival_time + _draw_prep_time("Angiography", loaded_inputs, rng)
                    block_idx = int(np.searchsorted(block_starts, ready_time, side="left"))
                    if block_idx < num_blocks:
                        demand[scenario, 1, block_idx] += 1.0

    measured_arrivals = np.where(measured_arrivals > 0.0, measured_arrivals, 1.0)
    return demand, measured_arrivals


# =========================================================
# LP formulation (improved)
# =========================================================
def _solve_lp(
    timetable_name,
    timetable,
    policy_space,
    share_daily_qik_across_classes,
    demand_scenarios,
    measured_arrivals,
    block_table,
    max_qik_value,
):
    basis = _decision_basis_weekly_qik(timetable, policy_space, share_daily_qik_across_classes)
    num_positions = basis.shape[0]
    if num_positions == 0:
        raise RuntimeError(f"No free Qik positions for timetable {timetable_name}.")

    num_scenarios = demand_scenarios.shape[0]
    num_blocks = demand_scenarios.shape[2]

    weekly_meta = weekly_block_metadata()
    weekly_weekday = weekly_meta["weekday"].to_numpy(dtype=int)

    weekly_block_total_coeff = np.zeros((40, num_positions), dtype=float)
    weekly_day_total_coeff = np.zeros((5, num_positions), dtype=float)
    horizon_capacity_coeff = np.zeros((2, num_blocks, num_positions), dtype=float)

    for pos in range(num_positions):
        weekly_block_total_coeff[:, pos] = basis[pos].sum(axis=0)
        for weekday in range(5):
            weekly_day_total_coeff[weekday, pos] = float(basis[pos][:, weekly_weekday == weekday].sum())

    weekly_block_index = block_table["weekly_block_index"].to_numpy(dtype=int)
    for block in range(num_blocks):
        k = weekly_block_index[block]
        horizon_capacity_coeff[0, block, :] = basis[:, 0, k]
        horizon_capacity_coeff[1, block, :] = basis[:, 1, k]

    # Decision variables layout:
    # [x (Qik positions)] [backlog per scenario/class/block] [overtime per weekday] [congestion]
    x_offset = 0
    backlog_offset = num_positions
    num_backlog = num_scenarios * 2 * num_blocks
    overtime_offset = backlog_offset + num_backlog
    congestion_offset = overtime_offset + 5
    num_vars = congestion_offset + 1

    def backlog_var(scenario, cls, block):
        return backlog_offset + (scenario * 2 + cls) * num_blocks + block

    def overtime_var(weekday):
        return overtime_offset + weekday

    congestion_var = congestion_offset

    # --- Objective ---
    c_obj = np.zeros(num_vars, dtype=float)
    gap_to_next_days = block_table["gap_to_next_days"].to_numpy(dtype=float)
    measured_mask = block_table["is_measured"].to_numpy(dtype=bool)

    # Z1 component: backlog * gap gives patient-days of waiting.
    # Divide by total measured arrivals and 28-day target to match H formula.
    for scenario in range(num_scenarios):
        wait_scale = 0.6 / (28.0 * float(measured_arrivals[scenario]) * num_scenarios)
        for block in range(num_blocks):
            if measured_mask[block]:
                coeff = wait_scale * gap_to_next_days[block]
                c_obj[backlog_var(scenario, 0, block)] = coeff
                c_obj[backlog_var(scenario, 1, block)] = coeff

    # Z2 component: overtime in minutes, convert to hours, scale by /2.5.
    overtime_coeff = 0.2 / (2.5 * MINUTES_PER_HOUR)
    for weekday in range(5):
        c_obj[overtime_var(weekday)] = overtime_coeff

    # Z3 component: congestion proxy, scale by /2.
    c_obj[congestion_var] = 0.2 / 2.0

    # --- Constraints ---
    rows = []
    cols = []
    data = []
    b_ub = []
    row_id = 0

    # Backlog flow constraints: backlog[t] >= backlog[t-1] + demand[t] - capacity[t]
    for scenario in range(num_scenarios):
        for cls in range(2):
            for block in range(num_blocks):
                cap_coeff = horizon_capacity_coeff[cls, block, :]
                for pos in np.flatnonzero(cap_coeff):
                    rows.append(row_id)
                    cols.append(x_offset + int(pos))
                    data.append(-float(cap_coeff[pos]))
                rows.append(row_id)
                cols.append(backlog_var(scenario, cls, block))
                data.append(-1.0)
                if block > 0:
                    rows.append(row_id)
                    cols.append(backlog_var(scenario, cls, block - 1))
                    data.append(1.0)
                b_ub.append(-float(demand_scenarios[scenario, cls, block]))
                row_id += 1

    # Overtime constraints: proc_time * daily_slots <= effective_staffed + overtime
    # Using 7 effective hours (excluding lunch break 12-13)
    mean_proc_minutes = distribution_mean(LOADED_INPUTS.procedure_distributions["Interventional"])
    for weekday in range(5):
        coeff = mean_proc_minutes * weekly_day_total_coeff[weekday, :]
        for pos in np.flatnonzero(coeff):
            rows.append(row_id)
            cols.append(x_offset + int(pos))
            data.append(float(coeff[pos]))
        rows.append(row_id)
        cols.append(overtime_var(weekday))
        data.append(-1.0)
        b_ub.append(float(EFFECTIVE_STAFFED_MINUTES_PER_DAY))
        row_id += 1

    # Congestion constraints: total slots in a weekly block <= safe_limit + congestion
    for weekly_block in range(40):
        coeff = weekly_block_total_coeff[weekly_block, :]
        for pos in np.flatnonzero(coeff):
            rows.append(row_id)
            cols.append(x_offset + int(pos))
            data.append(float(coeff[pos]))
        rows.append(row_id)
        cols.append(congestion_var)
        data.append(-1.0)
        b_ub.append(1.0)
        row_id += 1

    a_ub = coo_matrix((data, (rows, cols)), shape=(row_id, num_vars)).tocsr()
    bounds = [(0.0, float(max_qik_value))] * num_positions
    bounds += [(0.0, None)] * num_backlog
    bounds += [(0.0, None)] * 5
    bounds += [(0.0, None)]

    res = linprog(c=c_obj, A_ub=a_ub, b_ub=np.asarray(b_ub, dtype=float), bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP solve failed for {timetable_name}: {res.message}")

    return res.x[:num_positions]


# =========================================================
# Multiple rounding strategies
# =========================================================
def _generate_rounded_candidates(x_lp, timetable, policy_space, share_daily_qik_across_classes, max_qik_value, n_candidates, rng):
    """Generate multiple integer Qik candidates from the LP relaxation."""
    candidates = []
    seen = set()

    def _add_candidate(x_int):
        weekly = np.rint(
            _vector_to_weekly_qik(x_int, timetable, policy_space, share_daily_qik_across_classes)
        ).astype(int)
        key = weekly.tobytes()
        if key not in seen:
            seen.add(key)
            candidates.append(weekly)

    # Strategy 1: standard rounding
    x_round = np.clip(np.rint(x_lp), 0, max_qik_value).astype(int)
    _add_candidate(x_round)

    # Strategy 2: floor
    x_floor = np.clip(np.floor(x_lp), 0, max_qik_value).astype(int)
    _add_candidate(x_floor)

    # Strategy 3: ceil
    x_ceil = np.clip(np.ceil(x_lp), 0, max_qik_value).astype(int)
    _add_candidate(x_ceil)

    # Strategy 4: randomized rounding — round up with probability = fractional part
    for _ in range(n_candidates - 3):
        frac = x_lp - np.floor(x_lp)
        x_rand = np.where(rng.random(len(x_lp)) < frac, np.ceil(x_lp), np.floor(x_lp))
        x_rand = np.clip(x_rand, 0, max_qik_value).astype(int)
        _add_candidate(x_rand)

    return candidates


# =========================================================
# Screen candidates via short simulation
# =========================================================
def _screen_candidates(candidates, timetable_name, policy_space, num_reps, seed):
    """Run short simulation for each candidate and return (mean_H, candidate) pairs."""
    results = []
    for weekly_qik in candidates:
        _, builder = resolve_policy_builder(timetable_name, policy_space)
        if str(policy_space).strip().lower() == "full_week":
            policy = builder(weekly_qik)
        else:
            policy = builder(weekly_qik[:, :8])

        rep_df = run_replications(
            num_replications=num_reps,
            num_weeks=NUM_WEEKS,
            loaded_inputs=LOADED_INPUTS,
            policy=policy,
            base_seed=seed,
            warmup_weeks=WARMUP_WEEKS,
        )
        mean_h = float(rep_df["H"].mean())
        results.append((mean_h, weekly_qik, rep_df))
    return results


# =========================================================
# Full validation
# =========================================================
def _validate_policy(weekly_qik, timetable_name, policy_space, num_reps, seed):
    _, builder = resolve_policy_builder(timetable_name, policy_space)
    if str(policy_space).strip().lower() == "full_week":
        policy = builder(weekly_qik)
    else:
        policy = builder(weekly_qik[:, :8])

    rep_df = run_replications(
        num_replications=num_reps,
        num_weeks=NUM_WEEKS,
        loaded_inputs=LOADED_INPUTS,
        policy=policy,
        base_seed=seed,
        warmup_weeks=WARMUP_WEEKS,
    )
    return {
        "mean_H": float(rep_df["H"].mean()),
        "std_H": float(rep_df["H"].std(ddof=1)) if len(rep_df) > 1 else 0.0,
        "mean_Z1": float(rep_df["Z1_wait_time"].mean()),
        "mean_Z2": float(rep_df["Z2_overtime"].mean()),
        "mean_Z3": float(rep_df["Z3_congestion"].mean()),
    }


# =========================================================
# Main solver
# =========================================================
def solve_saa2(
    n=N_SAA,
    seed=BASE_SEED,
    search_timetable=SEARCH_TIMETABLE,
    policy_space=POLICY_SPACE,
    max_qik_value=MAX_QIK_VALUE,
    share_daily_qik_across_classes=SHARE_DAILY_QIK_ACROSS_CLASSES,
    screening_replications=SCREENING_REPLICATIONS,
    validation_replications=VALIDATION_REPLICATIONS,
    n_rounding_candidates=NUM_ROUNDING_CANDIDATES,
):
    total_sim_weeks = WARMUP_WEEKS + NUM_WEEKS
    block_table = _build_horizon_block_table(total_sim_weeks, WARMUP_WEEKS)
    demand_scenarios, measured_arrivals = _sample_ready_demand_scenarios(
        n=n, seed=seed, loaded_inputs=LOADED_INPUTS,
        total_sim_weeks=total_sim_weeks, warmup_weeks=WARMUP_WEEKS,
        block_table=block_table,
    )

    print("\n" + "=" * 80)
    print("SAA2: Improved SAA with simulation-based candidate selection")
    print(f"Scenarios: {n}, Screening reps: {screening_replications}, Rounding candidates: {n_rounding_candidates}")
    print(f"Avg measured arrivals/scenario: {measured_arrivals.mean():.1f}")

    rng = np.random.default_rng(seed + 99_999)
    all_results: list[SAA2Result] = []

    for timetable_name in resolve_search_timetables(search_timetable):
        timetable, _ = resolve_policy_builder(timetable_name, policy_space)

        print(f"\n--- Timetable {timetable_name} ---")

        # Step 1: Solve LP relaxation
        print("Solving LP relaxation...")
        x_lp = _solve_lp(
            timetable_name, timetable, policy_space,
            share_daily_qik_across_classes, demand_scenarios,
            measured_arrivals, block_table, max_qik_value,
        )
        print(f"LP solution: {np.round(x_lp, 3).tolist()}")

        # Step 2: Generate multiple rounded candidates
        candidates = _generate_rounded_candidates(
            x_lp, timetable, policy_space,
            share_daily_qik_across_classes, max_qik_value,
            n_rounding_candidates, rng,
        )
        print(f"Generated {len(candidates)} unique rounded candidates")

        # Step 3: Screen candidates with short simulation
        print(f"Screening candidates with {screening_replications} replications each...")
        screen_results = _screen_candidates(
            candidates, timetable_name, policy_space,
            screening_replications, seed + 10_000,
        )

        # Pick best by screening mean H
        screen_results.sort(key=lambda x: x[0])
        best_mean_h, best_weekly_qik, _ = screen_results[0]
        print(f"Best screening mean H = {best_mean_h:.4f} (total slots = {int(best_weekly_qik.sum())})")

        # Show top 3
        for rank, (mh, wq, _) in enumerate(screen_results[:3], 1):
            print(f"  #{rank}: mean H = {mh:.4f}, slots = {int(wq.sum())}")

        # Step 4: Full validation of the best candidate
        print(f"Validating best candidate with {validation_replications} replications...")
        val = _validate_policy(
            best_weekly_qik, timetable_name, policy_space,
            validation_replications, seed + 20_000,
        )

        policy_name = make_policy_name(timetable_name, "SAA2", best_weekly_qik)
        result = SAA2Result(
            timetable=timetable_name,
            policy_name=policy_name,
            weekly_qik=best_weekly_qik,
            screening_mean_H=best_mean_h,
            validation_mean_H=val["mean_H"],
            validation_std_H=val["std_H"],
            validation_mean_Z1=val["mean_Z1"],
            validation_mean_Z2=val["mean_Z2"],
            validation_mean_Z3=val["mean_Z3"],
        )
        all_results.append(result)

        print(f"Policy name: {policy_name}")
        print(f"Validation mean H  = {val['mean_H']:.4f}")
        print(f"Validation std H   = {val['std_H']:.4f}")
        print(f"Validation mean Z1 = {val['mean_Z1']:.4f} days")
        print(f"Validation mean Z2 = {val['mean_Z2']:.4f} hours/week")
        print(f"Validation mean Z3 = {val['mean_Z3']:.4f}")

    best = min(all_results, key=lambda r: r.validation_mean_H if r.validation_mean_H is not None else float("inf"))
    return all_results, best


def main():
    start_time = time.perf_counter()
    all_results, best = solve_saa2()
    elapsed = time.perf_counter() - start_time

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save per-timetable results
    summary_rows = []
    for result in all_results:
        weekly_qik_df = pd.DataFrame(
            result.weekly_qik,
            index=["Interventional", "Angiography"],
            columns=[f"block_{k}" for k in range(40)],
        )
        qik_path = OUTPUT_DIR / f"{result.timetable.lower()}_best_weekly_qik.csv"
        weekly_qik_df.to_csv(qik_path)

        summary_rows.append({
            "timetable": result.timetable,
            "policy_name": result.policy_name,
            "total_slots": int(result.weekly_qik.sum()),
            "screening_mean_H": result.screening_mean_H,
            "validation_mean_H": result.validation_mean_H,
            "validation_std_H": result.validation_std_H,
            "validation_mean_Z1": result.validation_mean_Z1,
            "validation_mean_Z2": result.validation_mean_Z2,
            "validation_mean_Z3": result.validation_mean_Z3,
            "total_run_time": elapsed,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("validation_mean_H", ignore_index=True)
    summary_path = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("SAA2 Overall Summary")
    print(summary_df.to_string(index=False))
    print(f"\nBest policy: {best.policy_name}")
    print(f"Best timetable: {best.timetable}")
    print(f"Best validation mean H  = {best.validation_mean_H:.4f}")
    print(f"Best validation mean Z1 = {best.validation_mean_Z1:.4f} days")
    print(f"Best validation mean Z2 = {best.validation_mean_Z2:.4f} hours/week")
    print(f"Best validation mean Z3 = {best.validation_mean_Z3:.4f}")
    print(f"\nTotal run time: {elapsed:.1f} seconds")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
