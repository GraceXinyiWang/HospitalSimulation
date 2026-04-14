from __future__ import annotations

"""
Shared utilities for the three optimization scripts:
- Optimization_SAA.py
- Optimization_Lin_Stage2.py
- Optimization_Subset_Selection+KN_simplified.py

This module centralises timetable/policy resolution, Qik serialization,
distribution-mean calculation, and feasible-position helpers so that all
three scripts use the same naming and conventions.
"""

import json
import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import Policy_defined
from input_loader import DistributionSpec, LoadedIRInputs, load_all_ir_inputs
from simulation_model import BookingTimetable


# =========================================================
# COMMON INPUT PATHS
# =========================================================
ARRIVAL_JSON_PATH = "arrival_model_params.json"
SERVICE_JSON_PATH = "services rate.json"
RAW_DATA_PATH = "df_selected.xlsx"


# =========================================================
# LOAD INPUTS
# =========================================================
def load_common_inputs() -> LoadedIRInputs:
    """Load the standard set of IR inputs used by all optimizers."""
    return load_all_ir_inputs(
        arrival_json_path=ARRIVAL_JSON_PATH,
        service_json_path=SERVICE_JSON_PATH,
        raw_data_path=RAW_DATA_PATH,
    )


# =========================================================
# TIMETABLE / POLICY RESOLUTION
# =========================================================
def resolve_search_timetables(search_timetable: str) -> List[str]:
    """Convert a search-timetable string to a list of timetable names.

    Accepts 'R1', 'R2', or 'both' (case-insensitive).
    """
    value = str(search_timetable).strip().upper()
    if value == "BOTH":
        return ["R1", "R2"]
    if value in {"R1", "R2"}:
        return [value]
    raise ValueError("search_timetable must be 'R1', 'R2', or 'both'.")


def resolve_timetable(search_timetable: str) -> BookingTimetable:
    """Return the BookingTimetable object for the given name."""
    value = str(search_timetable).strip().upper()
    if value == "R1":
        return Policy_defined.example_timetable_R1()
    if value == "R2":
        return Policy_defined.example_timetable_R2()
    raise ValueError("search_timetable must be 'R1' or 'R2'.")


def resolve_timetable_and_builders(
    search_timetable: str,
) -> Tuple[BookingTimetable, Any, Any]:
    """Return (timetable, daily_builder, weekly_builder) for the given name.

    daily_builder:  build_bruteforce_policy_R1/R2  (input shape 2×8)
    weekly_builder: build_general_policy_R1/R2     (input shape 2×40)
    """
    value = str(search_timetable).strip().upper()
    if value == "R1":
        return (
            Policy_defined.example_timetable_R1(),
            Policy_defined.build_bruteforce_policy_R1,
            Policy_defined.build_general_policy_R1,
        )
    if value == "R2":
        return (
            Policy_defined.example_timetable_R2(),
            Policy_defined.build_bruteforce_policy_R2,
            Policy_defined.build_general_policy_R2,
        )
    raise ValueError("search_timetable must be 'R1' or 'R2'.")


def resolve_policy_builder(search_timetable: str, policy_space: str):
    """Return (timetable, builder) appropriate for the given policy space.

    policy_space = 'daily_repeated' → daily_builder
    policy_space = 'full_week'      → weekly_builder
    """
    timetable, daily_builder, weekly_builder = resolve_timetable_and_builders(search_timetable)
    space = str(policy_space).strip().lower()
    if space == "daily_repeated":
        return timetable, daily_builder
    if space == "full_week":
        return timetable, weekly_builder
    raise ValueError("policy_space must be 'daily_repeated' or 'full_week'.")


# =========================================================
# QIK SERIALIZATION
# =========================================================
def serialize_qik(qik: np.ndarray) -> str:
    """Deterministic JSON string for a Qik array (for hashing / comparison)."""
    return json.dumps(np.asarray(qik, dtype=int).tolist(), separators=(",", ":"))


# =========================================================
# DISTRIBUTION MEAN
# =========================================================
def distribution_mean(spec: DistributionSpec) -> float:
    """Compute the analytical mean of a fitted DistributionSpec (in its stored unit)."""
    dist = str(spec.dist).strip().lower()
    p = spec.params

    if dist in {"deterministic", "constant"}:
        return float(p["value"])
    if dist == "empirical":
        samples = np.asarray(p["samples"], dtype=float)
        if samples.size == 0:
            return 0.0
        return float(samples.mean())
    if dist == "uniform":
        return 0.5 * (float(p["low"]) + float(p["high"]))
    if dist in {"exponential", "expon"}:
        return float(p["mean"])
    if dist == "gamma":
        return float(p["shape"]) * float(p["scale"])
    if dist == "lognormal":
        return float(math.exp(float(p["mu"]) + 0.5 * float(p["sigma"]) ** 2))
    if dist == "weibull":
        shape = float(p["shape"])
        scale = float(p["scale"])
        loc = float(p.get("loc", 0.0))
        return float(loc + scale * math.gamma(1.0 + 1.0 / shape))
    raise ValueError(f"Unsupported distribution for mean calculation: {spec.dist}")


# =========================================================
# FEASIBLE-POSITION HELPERS
# =========================================================
def daily_feasible_positions(timetable: BookingTimetable) -> List[Tuple[int, int]]:
    """Per-class daily feasible (class_idx, block) positions.

    A daily block is feasible if the class can use it on at least one weekday.
    Returns list of (class_idx, within_day_block) tuples.
    """
    weekly_mask = np.asarray(timetable.feasible_qik, dtype=int)
    stacked = np.stack([weekly_mask[:, 8 * d: 8 * (d + 1)] for d in range(5)], axis=0)
    feasible_daily = stacked.max(axis=0)
    return list(zip(*np.where(feasible_daily == 1)))


def shared_daily_feasible_blocks(timetable: BookingTimetable) -> np.ndarray:
    """Return a length-8 binary array: 1 if any class is feasible on any weekday.

    Used when both classes share one Qik value per within-day block.
    """
    weekly_mask = np.asarray(timetable.feasible_qik, dtype=int)
    stacked = np.stack([weekly_mask[:, 8 * d: 8 * (d + 1)] for d in range(5)], axis=0)
    feasible_daily_by_class = stacked.max(axis=0)
    return feasible_daily_by_class.max(axis=0)


def full_week_feasible_positions(timetable: BookingTimetable) -> List[Tuple[int, int]]:
    """Per-class weekly feasible (class_idx, weekly_block) positions.

    Returns list of (class_idx, weekly_block_index) tuples where the timetable
    mask is 1.
    """
    return list(zip(*np.where(np.asarray(timetable.feasible_qik, dtype=int) == 1)))


# =========================================================
# POLICY NAMING
# =========================================================
def _qik_hash(weekly_qik: np.ndarray, length: int = 8) -> str:
    """Deterministic short hash of a weekly Qik array."""
    import hashlib
    raw = serialize_qik(weekly_qik).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def make_policy_name(timetable_name: str, method: str, weekly_qik: np.ndarray) -> str:
    """Build a consistent, comparable policy name.

    Format: {timetable}_{qik_id}_{method}

    The qik_id is a short hash derived from the weekly Qik content.
    The same Qik always produces the same id regardless of which
    optimizer found it or when it was run, so you can directly compare
    policies across separate script runs:
    - R1_a3f7b2c1_SAA and R1_a3f7b2c1_Lin → same schedule
    - R1_a3f7b2c1_SAA and R1_d4e8c9f0_SAA → different schedules
    """
    weekly_qik = np.asarray(weekly_qik, dtype=int)
    qik_id = _qik_hash(weekly_qik)
    return f"{timetable_name}_{qik_id}_{method}"
