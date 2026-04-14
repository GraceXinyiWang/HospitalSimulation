from __future__ import annotations

"""
LP-style SAA approximation for the hospital Qik optimization problem.

This is not the exact simulation objective. It builds a deterministic-equivalent
linear approximation:

1. Sample demand scenarios from the fitted arrival/prep models.
2. Approximate booking delay by linear backlog variables.
3. Approximate overtime and congestion by linear surrogate variables.
4. Solve the LP, round Qik, then validate the rounded policy in simulation.
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


ARRIVAL_JSON_PATH = "arrival_model_params.json"
SERVICE_JSON_PATH = "services rate.json"
RAW_DATA_PATH = "df_selected.xlsx"

NUM_WEEKS = 210
WARMUP_WEEKS = 20
BASE_SEED = 123

SEARCH_TIMETABLE = "BOTH"
POLICY_SPACE = "full_week"  # "daily_repeated" or "full_week"
SHARE_DAILY_QIK_ACROSS_CLASSES = False
MAX_QIK_VALUE = 2

N_SAA = 20
VALIDATION_REPLICATIONS = 10
OUTPUT_DIR = Path("SAA_output_folder")

STAFFED_MINUTES_PER_DAY = 8.0 * MINUTES_PER_HOUR
SAFE_STARTS_PER_BLOCK = 1.0


LOADED_INPUTS = load_all_ir_inputs(
    arrival_json_path=ARRIVAL_JSON_PATH,
    service_json_path=SERVICE_JSON_PATH,
    raw_data_path=RAW_DATA_PATH,
)


@dataclass
class ApproximationResult:
    timetable: str
    solver_message: str
    surrogate_objective: float
    surrogate_wait_component: float
    surrogate_overtime_component: float
    surrogate_congestion_component: float
    x_lp: np.ndarray
    x_rounded: np.ndarray
    qik_input_rounded: np.ndarray
    weekly_qik_rounded: np.ndarray
    validation_mean_H: float | None
    validation_min_H: float | None
    validation_max_H: float | None
    validation_std_H: float | None
    validation_mean_Z1: float | None
    validation_mean_Z2: float | None
    validation_mean_Z3: float | None
    validation_mean_unscheduled: float | None


def _resolve_search_timetables(search_timetable: str):
    value = str(search_timetable).strip().upper()
    if value == "BOTH":
        return ["R1", "R2"]
    if value in {"R1", "R2"}:
        return [value]
    raise ValueError("search_timetable must be 'R1', 'R2', or 'both'.")


def _resolve_search_components(search_timetable: str, policy_space: str):
    policy_space = str(policy_space).strip().lower()
    value = str(search_timetable).strip().upper()

    if value == "R1":
        timetable = Policy_defined.example_timetable_R1()
        daily_builder = Policy_defined.build_bruteforce_policy_R1
        weekly_builder = Policy_defined.build_general_policy_R1
    elif value == "R2":
        timetable = Policy_defined.example_timetable_R2()
        daily_builder = Policy_defined.build_bruteforce_policy_R2
        weekly_builder = Policy_defined.build_general_policy_R2
    else:
        raise ValueError("search_timetable must be 'R1' or 'R2'.")

    if policy_space == "daily_repeated":
        return timetable, daily_builder
    if policy_space == "full_week":
        return timetable, weekly_builder
    raise ValueError("policy_space must be 'daily_repeated' or 'full_week'.")


def _daily_feasible_positions(timetable: BookingTimetable):
    weekly_mask = np.asarray(timetable.feasible_qik, dtype=int)
    stacked = np.stack([weekly_mask[:, 8 * d : 8 * (d + 1)] for d in range(5)], axis=0)
    feasible_daily = stacked.max(axis=0)
    return list(zip(*np.where(feasible_daily == 1)))


def _shared_daily_feasible_blocks(timetable: BookingTimetable):
    weekly_mask = np.asarray(timetable.feasible_qik, dtype=int)
    stacked = np.stack([weekly_mask[:, 8 * d : 8 * (d + 1)] for d in range(5)], axis=0)
    feasible_daily_by_class = stacked.max(axis=0)
    return list(np.where(feasible_daily_by_class.max(axis=0) == 1)[0])


def _full_week_feasible_positions(timetable: BookingTimetable):
    return list(zip(*np.where(np.asarray(timetable.feasible_qik, dtype=int) == 1)))


def _decision_positions(timetable: BookingTimetable, policy_space: str, share_daily_qik_across_classes: bool):
    policy_space = str(policy_space).strip().lower()
    if policy_space == "full_week":
        return _full_week_feasible_positions(timetable), "weekly_qik"
    if share_daily_qik_across_classes:
        return _shared_daily_feasible_blocks(timetable), "daily_qik_shared"
    return _daily_feasible_positions(timetable), "daily_qik"


def _vector_to_qik_input_linear(
    x: np.ndarray,
    timetable: BookingTimetable,
    policy_space: str,
    share_daily_qik_across_classes: bool,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    positions, space_label = _decision_positions(
        timetable=timetable,
        policy_space=policy_space,
        share_daily_qik_across_classes=share_daily_qik_across_classes,
    )
    if len(x) != len(positions):
        raise ValueError("Length of x does not match the number of free decision positions.")

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


def _qik_input_to_weekly_linear(qik_input: np.ndarray, timetable: BookingTimetable, policy_space: str):
    qik_input = np.asarray(qik_input, dtype=float)
    if str(policy_space).strip().lower() == "full_week":
        weekly_qik = qik_input.copy()
    else:
        weekly_qik = np.tile(qik_input, (1, 5))
    weekly_qik[timetable.feasible_qik == 0] = 0.0
    return weekly_qik


def _vector_to_weekly_qik_linear(
    x: np.ndarray,
    timetable: BookingTimetable,
    policy_space: str,
    share_daily_qik_across_classes: bool,
):
    qik_input = _vector_to_qik_input_linear(
        x=x,
        timetable=timetable,
        policy_space=policy_space,
        share_daily_qik_across_classes=share_daily_qik_across_classes,
    )
    return _qik_input_to_weekly_linear(qik_input=qik_input, timetable=timetable, policy_space=policy_space)


def _decision_basis_weekly_qik(
    timetable: BookingTimetable,
    policy_space: str,
    share_daily_qik_across_classes: bool,
):
    positions, _ = _decision_positions(
        timetable=timetable,
        policy_space=policy_space,
        share_daily_qik_across_classes=share_daily_qik_across_classes,
    )
    basis = np.zeros((len(positions), 2, 40), dtype=float)
    for pos in range(len(positions)):
        x = np.zeros(len(positions), dtype=float)
        x[pos] = 1.0
        basis[pos] = _vector_to_weekly_qik_linear(
            x=x,
            timetable=timetable,
            policy_space=policy_space,
            share_daily_qik_across_classes=share_daily_qik_across_classes,
        )
    return basis


def _round_x(x_lp: np.ndarray, max_qik_value: int):
    x_int = np.rint(np.asarray(x_lp, dtype=float)).astype(int)
    return np.clip(x_int, 0, int(max_qik_value))


def _distribution_mean_minutes(spec) -> float:
    dist = str(spec.dist).strip().lower()
    p = spec.params
    if dist in {"deterministic", "constant"}:
        return float(p["value"])
    if dist == "uniform":
        return 0.5 * (float(p["low"]) + float(p["high"]))
    if dist in {"exponential", "expon"}:
        return float(p["mean"])
    if dist == "gamma":
        return float(p["shape"]) * float(p["scale"])
    if dist == "lognormal":
        return math.exp(float(p["mu"]) + 0.5 * float(p["sigma"]) ** 2)
    if dist == "weibull":
        return float(p.get("loc", 0.0)) + float(p["scale"]) * math.gamma(1.0 + 1.0 / float(p["shape"]))
    if dist == "empirical":
        return float(np.mean(np.asarray(p["samples"], dtype=float)))
    raise ValueError(f"Unsupported distribution for mean calculation: {spec.dist}")


def _parse_bin_start_hours(bin_index: Sequence[object], fallback_start_hour: int):
    hours = []
    for idx, label in enumerate(bin_index):
        text = str(label)
        match = re.match(r"\s*(\d{1,2})\s*:\s*00", text)
        hours.append(int(match.group(1)) if match else int(fallback_start_hour) + idx)
    return hours


def _piecewise_rate_per_hour(time_in_day_min: float, start_hours: Sequence[int], rates: Sequence[float]) -> float:
    hour = int(time_in_day_min // MINUTES_PER_HOUR)
    for idx, start_hour in enumerate(start_hours):
        if start_hour <= hour < start_hour + 1:
            return float(rates[idx])
    return 0.0


def _draw_prep_time(category: str, loaded_inputs: LoadedIRInputs, rng: np.random.Generator) -> float:
    probs = loaded_inputs.prep_probabilities[category]
    labels = list(probs.keys())
    p = np.array([probs[label] for label in labels], dtype=float)
    p = p / p.sum()
    prep_type = str(rng.choice(labels, p=p))
    return max(0.0, sample_from_spec(loaded_inputs.prep_distributions[category][prep_type], rng))


def _build_horizon_block_table(total_sim_weeks: int, warmup_weeks: int):
    meta = weekly_block_metadata()
    rows = []
    for week in range(total_sim_weeks):
        for row in meta.itertuples(index=False):
            day_index = 7 * week + int(row.weekday)
            start_time = day_index * MINUTES_PER_DAY + int(row.start_hour) * MINUTES_PER_HOUR
            rows.append(
                {
                    "week": week,
                    "weekday": int(row.weekday),
                    "weekly_block_index": int(row.block_index),
                    "start_time": float(start_time),
                    "is_measured": week >= warmup_weeks,
                }
            )
    df = pd.DataFrame(rows).sort_values("start_time", kind="stable").reset_index(drop=True)
    df["gap_to_next_min"] = df["start_time"].shift(-1) - df["start_time"]
    df["gap_to_next_min"] = df["gap_to_next_min"].fillna(0.0)
    return df


def _sample_ready_demand_scenarios(
    n: int,
    seed: int,
    loaded_inputs: LoadedIRInputs,
    total_sim_weeks: int,
    warmup_weeks: int,
    block_table: pd.DataFrame,
):
    block_starts = block_table["start_time"].to_numpy(dtype=float)
    num_blocks = len(block_starts)
    measurement_start_time = warmup_weeks * 7 * MINUTES_PER_DAY

    demand = np.zeros((n, 2, num_blocks), dtype=float)
    measured_arrivals = np.zeros(n, dtype=float)

    working_days = [7 * week + weekday for week in range(total_sim_weeks) for weekday in range(5)]
    arrival_inputs = loaded_inputs.arrival_inputs

    int_start_hours = _parse_bin_start_hours(
        arrival_inputs.interventional_lambda_hat.index,
        arrival_inputs.bin_start_hour,
    )
    int_rates = np.asarray(arrival_inputs.interventional_lambda_hat.values, dtype=float)
    int_lambda_max = float(int_rates.max()) if len(int_rates) > 0 else 0.0
    int_work_start = min(int_start_hours) * MINUTES_PER_HOUR
    int_work_end = (max(int_start_hours) + 1) * MINUTES_PER_HOUR

    ang_start_hours = _parse_bin_start_hours(
        arrival_inputs.angiography_lambda_hat.index,
        arrival_inputs.bin_start_hour,
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


def _solve_surrogate_lp_for_timetable(
    timetable_name: str,
    timetable: BookingTimetable,
    policy_space: str,
    share_daily_qik_across_classes: bool,
    demand_scenarios: np.ndarray,
    measured_arrivals: np.ndarray,
    block_table: pd.DataFrame,
    max_qik_value: int,
):
    basis = _decision_basis_weekly_qik(
        timetable=timetable,
        policy_space=policy_space,
        share_daily_qik_across_classes=share_daily_qik_across_classes,
    )
    num_positions = basis.shape[0]
    if num_positions == 0:
        raise RuntimeError(f"No free Qik decision variables found for timetable {timetable_name}.")

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

    x_offset = 0
    backlog_offset = num_positions
    num_backlog = num_scenarios * 2 * num_blocks
    overtime_offset = backlog_offset + num_backlog
    congestion_offset = overtime_offset + 5
    num_vars = congestion_offset + 1

    def backlog_var(scenario: int, cls: int, block: int):
        return backlog_offset + (scenario * 2 + cls) * num_blocks + block

    def overtime_var(weekday: int):
        return overtime_offset + weekday

    congestion_var = congestion_offset

    c_obj = np.zeros(num_vars, dtype=float)
    gap_to_next = block_table["gap_to_next_min"].to_numpy(dtype=float)
    measured_mask = block_table["is_measured"].to_numpy(dtype=bool)

    for scenario in range(num_scenarios):
        wait_scale = 0.6 / (28.0 * MINUTES_PER_DAY * float(measured_arrivals[scenario]) * num_scenarios)
        for block in range(num_blocks):
            if measured_mask[block]:
                coeff = wait_scale * gap_to_next[block]
                c_obj[backlog_var(scenario, 0, block)] = coeff
                c_obj[backlog_var(scenario, 1, block)] = coeff

    overtime_coeff = 0.2 / (2.5 * MINUTES_PER_HOUR)
    for weekday in range(5):
        c_obj[overtime_var(weekday)] = overtime_coeff
    c_obj[congestion_var] = 0.2 / 2.0

    rows = []
    cols = []
    data = []
    b_ub = []
    row_id = 0

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

    mean_proc_minutes = _distribution_mean_minutes(LOADED_INPUTS.procedure_distributions["Interventional"])
    for weekday in range(5):
        coeff = mean_proc_minutes * weekly_day_total_coeff[weekday, :]
        for pos in np.flatnonzero(coeff):
            rows.append(row_id)
            cols.append(x_offset + int(pos))
            data.append(float(coeff[pos]))
        rows.append(row_id)
        cols.append(overtime_var(weekday))
        data.append(-1.0)
        b_ub.append(float(STAFFED_MINUTES_PER_DAY))
        row_id += 1

    for weekly_block in range(40):
        coeff = weekly_block_total_coeff[weekly_block, :]
        for pos in np.flatnonzero(coeff):
            rows.append(row_id)
            cols.append(x_offset + int(pos))
            data.append(float(coeff[pos]))
        rows.append(row_id)
        cols.append(congestion_var)
        data.append(-1.0)
        b_ub.append(float(SAFE_STARTS_PER_BLOCK))
        row_id += 1

    a_ub = coo_matrix((data, (rows, cols)), shape=(row_id, num_vars)).tocsr()
    bounds = [(0.0, float(max_qik_value))] * num_positions
    bounds += [(0.0, None)] * num_backlog
    bounds += [(0.0, None)] * 5
    bounds += [(0.0, None)]

    res = linprog(
        c=c_obj,
        A_ub=a_ub,
        b_ub=np.asarray(b_ub, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP solve failed for {timetable_name}: {res.message}")

    x_lp = res.x[:num_positions]
    x_rounded = _round_x(x_lp=x_lp, max_qik_value=max_qik_value)
    qik_input_rounded = np.rint(
        _vector_to_qik_input_linear(
            x=x_rounded,
            timetable=timetable,
            policy_space=policy_space,
            share_daily_qik_across_classes=share_daily_qik_across_classes,
        )
    ).astype(int)
    weekly_qik_rounded = np.rint(
        _vector_to_weekly_qik_linear(
            x=x_rounded,
            timetable=timetable,
            policy_space=policy_space,
            share_daily_qik_across_classes=share_daily_qik_across_classes,
        )
    ).astype(int)

    wait_component = 0.0
    for scenario in range(num_scenarios):
        wait_scale = 0.6 / (28.0 * MINUTES_PER_DAY * float(measured_arrivals[scenario]) * num_scenarios)
        for block in range(num_blocks):
            if measured_mask[block]:
                coeff = wait_scale * gap_to_next[block]
                wait_component += coeff * res.x[backlog_var(scenario, 0, block)]
                wait_component += coeff * res.x[backlog_var(scenario, 1, block)]

    overtime_component = sum(overtime_coeff * res.x[overtime_var(d)] for d in range(5))
    congestion_component = c_obj[congestion_var] * res.x[congestion_var]

    return {
        "solver_message": res.message,
        "objective": float(res.fun),
        "wait_component": float(wait_component),
        "overtime_component": float(overtime_component),
        "congestion_component": float(congestion_component),
        "x_lp": x_lp,
        "x_rounded": x_rounded,
        "qik_input_rounded": qik_input_rounded,
        "weekly_qik_rounded": weekly_qik_rounded,
    }


def _validate_rounded_policy(
    timetable_name: str,
    policy_space: str,
    qik_input_rounded: np.ndarray,
    num_replications: int,
    seed: int,
):
    if num_replications <= 0:
        return {
            "mean_H": None,
            "min_H": None,
            "max_H": None,
            "std_H": None,
            "mean_Z1": None,
            "mean_Z2": None,
            "mean_Z3": None,
            "mean_unscheduled": None,
        }

    _, builder = _resolve_search_components(search_timetable=timetable_name, policy_space=policy_space)
    rep_df = run_replications(
        num_replications=num_replications,
        num_weeks=NUM_WEEKS,
        loaded_inputs=LOADED_INPUTS,
        policy=builder(qik_input_rounded),
        base_seed=seed,
        warmup_weeks=WARMUP_WEEKS,
    )
    return {
        "mean_H": float(rep_df["H"].mean()),
        "min_H": float(rep_df["H"].min()),
        "max_H": float(rep_df["H"].max()),
        "std_H": float(rep_df["H"].std(ddof=1)) if len(rep_df) > 1 else 0.0,
        "mean_Z1": float(rep_df["Z1_wait_time"].mean()),
        "mean_Z2": float(rep_df["Z2_overtime"].mean()),
        "mean_Z3": float(rep_df["Z3_congestion"].mean()),
        "mean_unscheduled": float(rep_df["unscheduled_count"].mean()),
    }


def solve_saa_hospital_qik(
    n: int = N_SAA,
    seed: int = BASE_SEED,
    search_timetable: str = SEARCH_TIMETABLE,
    policy_space: str = POLICY_SPACE,
    max_qik_value: int = MAX_QIK_VALUE,
    share_daily_qik_across_classes: bool = SHARE_DAILY_QIK_ACROSS_CLASSES,
    validation_replications: int = VALIDATION_REPLICATIONS,
):
    total_sim_weeks = WARMUP_WEEKS + NUM_WEEKS
    block_table = _build_horizon_block_table(total_sim_weeks=total_sim_weeks, warmup_weeks=WARMUP_WEEKS)
    demand_scenarios, measured_arrivals = _sample_ready_demand_scenarios(
        n=n,
        seed=seed,
        loaded_inputs=LOADED_INPUTS,
        total_sim_weeks=total_sim_weeks,
        warmup_weeks=WARMUP_WEEKS,
        block_table=block_table,
    )

    print("\n" + "=" * 80)
    print("Sampled hospital-demand scenarios for SAA approximation")
    print(f"Scenarios (n): {n}")
    print(f"Total horizon weeks: {total_sim_weeks}")
    print(f"Measured weeks: {NUM_WEEKS}")
    print(f"Average measured arrivals per scenario: {measured_arrivals.mean():.2f}")
    print(
        "Average ready-demand counts by class over the full horizon: "
        f"Interventional={demand_scenarios[:, 0, :].sum(axis=1).mean():.2f}, "
        f"Angiography={demand_scenarios[:, 1, :].sum(axis=1).mean():.2f}"
    )

    results: list[ApproximationResult] = []
    for timetable_name in _resolve_search_timetables(search_timetable):
        timetable, _ = _resolve_search_components(search_timetable=timetable_name, policy_space=policy_space)
        lp_out = _solve_surrogate_lp_for_timetable(
            timetable_name=timetable_name,
            timetable=timetable,
            policy_space=policy_space,
            share_daily_qik_across_classes=share_daily_qik_across_classes,
            demand_scenarios=demand_scenarios,
            measured_arrivals=measured_arrivals,
            block_table=block_table,
            max_qik_value=max_qik_value,
        )
        validation_out = _validate_rounded_policy(
            timetable_name=timetable_name,
            policy_space=policy_space,
            qik_input_rounded=lp_out["qik_input_rounded"],
            num_replications=validation_replications,
            seed=seed + 10_000,
        )
        result = ApproximationResult(
            timetable=timetable_name,
            solver_message=lp_out["solver_message"],
            surrogate_objective=lp_out["objective"],
            surrogate_wait_component=lp_out["wait_component"],
            surrogate_overtime_component=lp_out["overtime_component"],
            surrogate_congestion_component=lp_out["congestion_component"],
            x_lp=lp_out["x_lp"],
            x_rounded=lp_out["x_rounded"],
            qik_input_rounded=lp_out["qik_input_rounded"],
            weekly_qik_rounded=lp_out["weekly_qik_rounded"],
            validation_mean_H=validation_out["mean_H"],
            validation_min_H=validation_out["min_H"],
            validation_max_H=validation_out["max_H"],
            validation_std_H=validation_out["std_H"],
            validation_mean_Z1=validation_out["mean_Z1"],
            validation_mean_Z2=validation_out["mean_Z2"],
            validation_mean_Z3=validation_out["mean_Z3"],
            validation_mean_unscheduled=validation_out["mean_unscheduled"],
        )
        results.append(result)

        print("\n" + "=" * 80)
        print(f"Timetable = {timetable_name}")
        print(f"Policy space = {policy_space}")
        print(f"Solver message = {result.solver_message}")
        print(f"Surrogate SAA objective = {result.surrogate_objective:.6f}")
        print(f"  wait component       = {result.surrogate_wait_component:.6f}")
        print(f"  overtime component   = {result.surrogate_overtime_component:.6f}")
        print(f"  congestion component = {result.surrogate_congestion_component:.6f}")
        print(f"LP x = {np.round(result.x_lp, 4).tolist()}")
        print(f"Rounded x = {result.x_rounded.tolist()}")
        print(f"Rounded weekly Qik total slots = {int(result.weekly_qik_rounded.sum())}")
        if result.validation_mean_H is not None:
            print(f"Validation mean H = {result.validation_mean_H:.6f}")
            print(f"Validation min H  = {result.validation_min_H:.6f}")
            print(f"Validation max H  = {result.validation_max_H:.6f}")
            print(f"Validation std H  = {result.validation_std_H:.6f}")
            print(f"Validation mean Z1 = {result.validation_mean_Z1:.6f}")
            print(f"Validation mean Z2 = {result.validation_mean_Z2:.6f}")
            print(f"Validation mean Z3 = {result.validation_mean_Z3:.6f}")
            print(f"Validation mean unscheduled = {result.validation_mean_unscheduled:.6f}")

    best_result = min(results, key=lambda item: item.surrogate_objective)
    summary_table = pd.DataFrame(
        [
            {
                "timetable": item.timetable,
                "surrogate_objective": item.surrogate_objective,
                "surrogate_wait_component": item.surrogate_wait_component,
                "surrogate_overtime_component": item.surrogate_overtime_component,
                "surrogate_congestion_component": item.surrogate_congestion_component,
                "validation_mean_H": item.validation_mean_H,
                "validation_min_H": item.validation_min_H,
                "validation_max_H": item.validation_max_H,
                "validation_std_H": item.validation_std_H,
                "weekly_slots": int(item.weekly_qik_rounded.sum()),
                "x_rounded": item.x_rounded.tolist(),
            }
            for item in results
        ]
    ).sort_values(["surrogate_objective", "timetable"], ascending=[True, True], ignore_index=True)

    return {
        "surrogate_objective": best_result.surrogate_objective,
        "x_lp_opt": best_result.x_lp.copy(),
        "x_opt": best_result.x_rounded.copy(),
        "qik_opt": best_result.qik_input_rounded.copy(),
        "weekly_qik_opt": best_result.weekly_qik_rounded.copy(),
        "best_timetable": best_result.timetable,
        "best_validation_mean_H": best_result.validation_mean_H,
        "best_validation_min_H": best_result.validation_min_H,
        "best_validation_max_H": best_result.validation_max_H,
        "best_validation_std_H": best_result.validation_std_H,
        "summary_table": summary_table,
        "all_results": results,
    }


def main():
    out = solve_saa_hospital_qik()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_result_summary = pd.DataFrame(
        [
            {
                "best_timetable": out["best_timetable"],
                "surrogate_objective": out["surrogate_objective"],
                "validation_mean_H": out["best_validation_mean_H"],
                "validation_min_H": out["best_validation_min_H"],
                "validation_max_H": out["best_validation_max_H"],
                "validation_std_H": out["best_validation_std_H"],
                "x_opt": out["x_opt"].tolist(),
                "qik_opt": out["qik_opt"].tolist(),
            }
        ]
    )
    weekly_qik_df = pd.DataFrame(
        out["weekly_qik_opt"],
        index=["Interventional", "Angiography"],
        columns=[f"block_{k}" for k in range(40)],
    )

    best_result_summary_path = OUTPUT_DIR / "best_result_summary.csv"
    weekly_qik_path = OUTPUT_DIR / "best_result_weekly_qik.csv"
    summary_table_path = OUTPUT_DIR / "summary_table.csv"

    best_result_summary.to_csv(best_result_summary_path, index=False)
    weekly_qik_df.to_csv(weekly_qik_path)
    out["summary_table"].to_csv(summary_table_path, index=False)

    print("\n" + "=" * 80)
    print("Overall best LP-SAA approximation result")
    print(f"Best timetable: {out['best_timetable']}")
    print(f"Optimal rounded x: {out['x_opt'].tolist()}")
    print(f"Rounded Qik input: {out['qik_opt'].tolist()}")
    print(f"Rounded weekly Qik: {out['weekly_qik_opt'].tolist()}")
    print(f"Best surrogate SAA objective: {out['surrogate_objective']:.6f}")
    if out["best_validation_mean_H"] is not None:
        print(f"Validation mean H: {out['best_validation_mean_H']:.6f}")
        print(f"Validation min H:  {out['best_validation_min_H']:.6f}")
        print(f"Validation max H:  {out['best_validation_max_H']:.6f}")
        print(f"Validation std H:  {out['best_validation_std_H']:.6f}")
    print(f"Saved summary CSV: {best_result_summary_path}")
    print(f"Saved weekly Qik CSV: {weekly_qik_path}")
    print(f"Saved table CSV: {summary_table_path}")
    print("\nActual full weekly Qik")
    print(
        weekly_qik_df.to_string()
    )
    print("\nSummary table")
    print(out["summary_table"].to_string(index=False))


if __name__ == "__main__":
    main()
