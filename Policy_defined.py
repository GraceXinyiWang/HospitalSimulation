from __future__ import annotations

"""
Example weekly policies for the IR outpatient scheduling simulation.

Key idea
--------
R1 / R2:
    A timetable mask that tells us whether a patient class is allowed
    to be booked in each of the 40 weekly blocks.

Qik:
    The number of patients of class i booked into weekly block k.

A full policy = timetable + Qik.
"""

import numpy as np

from simulation_model import BookingTimetable, WeeklySchedulePolicy, apply_feasibility_to_qik, policy_from_qik


def example_timetable_R1() -> BookingTimetable:
    """R1 example: every class is allowed in every weekly block.

    feasible_qik has shape (2, 40):
    - row 0 = Interventional
    - row 1 = Angiography
    - 40 columns = 40 one-hour weekday blocks in a week
    """
    feasible_qik = np.ones((2, 40), dtype=int)
    return BookingTimetable(name="R1_placeholder", feasible_qik=feasible_qik)



def example_timetable_R2() -> BookingTimetable:
    """R2 example: some blocks are marked infeasible for each class.

    This is only a placeholder example. Replace these zeros with the real
    UHN timetable rules when you define the actual R2.
    """
    feasible_qik = np.ones((2, 40), dtype=int)

    # Interventional is not allowed in these weekly blocks.
    feasible_qik[0, [2, 3, 10, 11, 18, 19, 26, 27, 34, 35]] = 0

    # Angiography is not allowed in these weekly blocks.
    feasible_qik[1, [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]] = 0

    return BookingTimetable(name="R2_placeholder", feasible_qik=feasible_qik)



def example_qik() -> np.ndarray:
    """Create one simple example weekly booking matrix.

    qik has shape (2, 40):
    - row 0 = Interventional bookings in each weekly block
    - row 1 = Angiography bookings in each weekly block

    This example uses the same pattern on each weekday:
    - Interventional: 2 blocks per day, capacity 1 in each selected block
    - Angiography: 3 blocks per day, capacity 1 in each selected block
    """
    qik = np.ones((2, 40), dtype=int)

    for day in range(5):
        # Each weekday uses 8 blocks, so:
        # Monday    -> columns 0 to 7
        # Tuesday   -> columns 8 to 15
        # Wednesday -> columns 16 to 23
        # Thursday  -> columns 24 to 31
        # Friday    -> columns 32 to 39
        base = 8 * day

        # Interventional row = row 0.
        # Example: allow 1 Interventional patient in the 1st and 5th block of that day.
        qik[0, base + 0] = 1
        qik[0, base + 4] = 1

        # Angiography row = row 1.
        # Example: allow 1 Angiography patient in the 2nd, 3rd, and 7th block of that day.
        qik[1, base + 1] = 1
        qik[1, base + 2] = 1
        qik[1, base + 6] = 1

    return qik



def example_policy_R1() -> WeeklySchedulePolicy:
    """Build a complete example policy under R1.

    Step 1: build R1 timetable
    Step 2: build example Qik
    Step 3: force infeasible entries to zero
    Step 4: package everything into one policy object
    """
    timetable = example_timetable_R1()
    qik = apply_feasibility_to_qik(example_qik(), timetable)
    return policy_from_qik(qik, timetable)



def example_policy_R2() -> WeeklySchedulePolicy:
    """Build a complete example policy under R2."""
    timetable = example_timetable_R2()
    qik = apply_feasibility_to_qik(example_qik(), timetable)
    return policy_from_qik(qik, timetable)
