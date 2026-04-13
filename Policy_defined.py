from __future__ import annotations

"""
Example weekly policies for the IR outpatient scheduling simulation.

Design
------
We support two Qik modes:

1. Same-each-weekday mode
   - input shape = (2, 8)
   - one daily pattern
   - repeated Monday to Friday
   - useful for brute force

2. Full-week mode
   - input shape = (2, 40)
   - Monday to Friday can all be different
   - useful for other optimization algorithms
"""

import numpy as np
from itertools import product

from simulation_model import (
    BookingTimetable,
    WeeklySchedulePolicy,
    apply_feasibility_to_qik,
    policy_from_qik,
)


# =========================================================
# TIMETABLES
# =========================================================
def example_timetable_R1() -> BookingTimetable:
    """R1: every class is allowed in every weekly block."""
    feasible_qik = np.ones((2, 40), dtype=int)

    # Interventional infeasible weekly blocks
    feasible_qik[1, [8,16,24]] = 0

    # Angiography infeasible weekly blocks
    feasible_qik[0, [3,5,8]] = 0

    return BookingTimetable(name="R1_placeholder", feasible_qik=feasible_qik)


def example_timetable_R2() -> BookingTimetable:
    """R2: placeholder infeasibility example."""
    feasible_qik = np.ones((2, 40), dtype=int)

    # Interventional infeasible weekly blocks
    feasible_qik[0, [2, 3, 10, 11, 18, 19, 26, 27, 34, 35]] = 0

    # Angiography infeasible weekly blocks
    feasible_qik[1, [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]] = 0

    return BookingTimetable(name="R2_placeholder", feasible_qik=feasible_qik)


# =========================================================
# QIK HELPERS
# =========================================================
def build_weekly_qik_from_daily(daily_qik: list[list[int]] | np.ndarray) -> np.ndarray:
    """
    Convert a daily Qik of shape (2, 8) into a weekly Qik of shape (2, 40)
    by repeating the same pattern Monday to Friday.
    """
    daily_qik = np.asarray(daily_qik, dtype=int)

    if daily_qik.shape != (2, 8):
        raise ValueError("daily_qik must have shape (2, 8).")

    if np.any(daily_qik < 0):
        raise ValueError("daily_qik entries must be nonnegative.")

    return np.tile(daily_qik, (1, 5))


def validate_weekly_qik(weekly_qik: list[list[int]] | np.ndarray) -> np.ndarray:
    """
    Validate and return a weekly Qik of shape (2, 40).
    """
    weekly_qik = np.asarray(weekly_qik, dtype=int)

    if weekly_qik.shape != (2, 40):
        raise ValueError("weekly_qik must have shape (2, 40).")

    if np.any(weekly_qik < 0):
        raise ValueError("weekly_qik entries must be nonnegative.")

    return weekly_qik


def qik_from_input(
    qik_input: list[list[int]] | np.ndarray | None,
    same_each_weekday: bool,
) -> np.ndarray:
    """
    Build a weekly Qik from either:
    - a daily Qik (2, 8), if same_each_weekday=True
    - a weekly Qik (2, 40), if same_each_weekday=False
    """
    if qik_input is None:
        if same_each_weekday:
            qik_input = default_daily_qik()
        else:
            qik_input = default_weekly_qik()

    if same_each_weekday:
        return build_weekly_qik_from_daily(qik_input)
    return validate_weekly_qik(qik_input)


def default_daily_qik() -> list[list[int]]:
    """
    Default daily Qik for same-each-weekday mode.
    Shape = (2, 8)
    """
    return [
        [1, 1, 1, 1, 1, 1, 0, 0],  # Interventional
        [1, 1, 1, 0, 1, 1, 1, 0],  # Angiography
    ]


def default_weekly_qik() -> list[list[int]]:
    """
    Default weekly Qik for full-week mode.
    Shape = (2, 40)
    Here Monday-Friday are allowed to differ.
    """
    return [
        # Interventional: 5 days x 8 blocks = 40
        [
            1, 1, 1, 1, 1, 1, 0, 0,   # Mon
            1, 1, 1, 0, 1, 1, 0, 0,   # Tue
            1, 1, 0, 0, 1, 1, 0, 0,   # Wed
            1, 1, 1, 1, 0, 0, 0, 0,   # Thu
            1, 0, 1, 0, 1, 0, 0, 0,   # Fri
        ],
        # Angiography
        [
            1, 1, 1, 0, 1, 1, 1, 0,   # Mon
            1, 0, 1, 1, 1, 0, 1, 0,   # Tue
            1, 1, 1, 1, 0, 0, 1, 0,   # Wed
            1, 1, 0, 1, 1, 1, 0, 0,   # Thu
            0, 1, 1, 1, 0, 1, 1, 0,   # Fri
        ],
    ]


# =========================================================
# POLICY BUILDERS
# =========================================================
def example_policy_R1(
    qik_input: list[list[int]] | np.ndarray | None = None,
    same_each_weekday: bool = True,
) -> WeeklySchedulePolicy:
    """
    Build a complete policy under R1.

    same_each_weekday=True:
        qik_input must be shape (2, 8)

    same_each_weekday=False:
        qik_input must be shape (2, 40)
    """
    timetable = example_timetable_R1()
    qik = qik_from_input(qik_input, same_each_weekday=same_each_weekday)
    qik = apply_feasibility_to_qik(qik, timetable)
    return policy_from_qik(qik, timetable)


def example_policy_R2(
    qik_input: list[list[int]] | np.ndarray | None = None,
    same_each_weekday: bool = True,
) -> WeeklySchedulePolicy:
    """
    Build a complete policy under R2.

    same_each_weekday=True:
        qik_input must be shape (2, 8)

    same_each_weekday=False:
        qik_input must be shape (2, 40)
    """
    timetable = example_timetable_R2()
    qik = qik_from_input(qik_input, same_each_weekday=same_each_weekday)
    qik = apply_feasibility_to_qik(qik, timetable)
    return policy_from_qik(qik, timetable)


# =========================================================
# GENERATORS FOR OPTIMIZATION
# =========================================================
def generate_bruteforce_daily_qik_candidates(
    max_value: int = 3,
    timetable: BookingTimetable | None = None,
):
    """
    Generator for brute force:
    - decision variable is daily Qik, shape (2, 8)
    - Monday-Friday are forced to be the same

    Each free entry takes values in {0, 1, ..., max_value}.

    If timetable is provided, a daily position is fixed to 0 only if it is
    infeasible on all five weekdays at that same within-day block index.
    """
    if max_value < 0:
        raise ValueError("max_value must be >= 0.")

    feasible_daily = np.ones((2, 8), dtype=int)

    if timetable is not None:
        weekly_mask = np.asarray(timetable.feasible_qik, dtype=int)
        if weekly_mask.shape != (2, 40):
            raise ValueError("timetable.feasible_qik must have shape (2, 40).")

        # Stack weekday masks: shape -> (5, 2, 8)
        stacked = np.stack(
            [weekly_mask[:, 8 * day : 8 * (day + 1)] for day in range(5)],
            axis=0,
        )

        # A daily position is feasible if it is feasible on at least one weekday
        feasible_daily = stacked.max(axis=0)

    free_positions = list(zip(*np.where(feasible_daily == 1)))

    for values in product(range(max_value + 1), repeat=len(free_positions)):
        daily_qik = np.zeros((2, 8), dtype=int)
        for (i, k), v in zip(free_positions, values):
            daily_qik[i, k] = v
        yield daily_qik


def generate_full_week_qik_candidates(
    max_value: int = 3,
    timetable: BookingTimetable | None = None,
):
    """
    Generator for other algorithms:
    - decision variable is full weekly Qik, shape (2, 40)
    - Monday-Friday can all be different

    Each free entry takes values in {0, 1, ..., max_value}.
    """
    if max_value < 0:
        raise ValueError("max_value must be >= 0.")

    feasible_weekly = np.ones((2, 40), dtype=int)

    if timetable is not None:
        feasible_weekly = np.asarray(timetable.feasible_qik, dtype=int)
        if feasible_weekly.shape != (2, 40):
            raise ValueError("timetable.feasible_qik must have shape (2, 40).")

    free_positions = list(zip(*np.where(feasible_weekly == 1)))

    for values in product(range(max_value + 1), repeat=len(free_positions)):
        weekly_qik = np.zeros((2, 40), dtype=int)
        for (i, k), v in zip(free_positions, values):
            weekly_qik[i, k] = v
        yield weekly_qik


# =========================================================
# OPTIONAL CONVENIENCE WRAPPERS
# =========================================================
def build_bruteforce_policy_R1(daily_qik: list[list[int]] | np.ndarray) -> WeeklySchedulePolicy:
    """
    For brute force algorithm:
    - input shape (2, 8)
    - repeated Monday-Friday
    """
    return example_policy_R1(qik_input=daily_qik, same_each_weekday=True)


def build_bruteforce_policy_R2(daily_qik: list[list[int]] | np.ndarray) -> WeeklySchedulePolicy:
    """
    For brute force algorithm:
    - input shape (2, 8)
    - repeated Monday-Friday
    """
    return example_policy_R2(qik_input=daily_qik, same_each_weekday=True)


def build_general_policy_R1(weekly_qik: list[list[int]] | np.ndarray) -> WeeklySchedulePolicy:
    """
    For other algorithms:
    - input shape (2, 40)
    - Monday-Friday can differ
    """
    return example_policy_R1(qik_input=weekly_qik, same_each_weekday=False)


def build_general_policy_R2(weekly_qik: list[list[int]] | np.ndarray) -> WeeklySchedulePolicy:
    """
    For other algorithms:
    - input shape (2, 40)
    - Monday-Friday can differ
    """
    return example_policy_R2(qik_input=weekly_qik, same_each_weekday=False)