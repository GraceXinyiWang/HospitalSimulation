from __future__ import annotations

"""
Core simulation model for IR outpatient scheduling.

Model logic
-----------
1. Generate outpatient order arrivals.
   - Interventional: NHPP thinning.
   - Angiography: Poisson-lognormal daily effect.
2. At order arrival, assign a preparation type (by their own possibility).
3. Sample preparation duration.
4. When preparation ends, the patient becomes ready to schedule.
5. At ready-to-schedule time, the admin books the patient immediately into the
   earliest future feasible block with remaining class-specific Qik capacity.
6. After the booked time plus lateness, the patient arrives at the waiting area.
7. The patient starts the procedure immediately if the one room is free;
   otherwise the patient waits in a FIFO waiting-room queue.
8. The procedure ends, the room is released, and overtime is updated.

"""

import heapq
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from input_loader import DistributionSpec, LoadedIRInputs


MINUTES_PER_HOUR = 60.0
HOUR_PER_DAY = 24.0
MINUTES_PER_DAY = 24.0 * 60.0

# Eight one-hour booking blocks per weekday.
# Lunch break 12:00-13:00 is skipped. This is used to track the staff overtime
WORKDAY_START_HOURS = (8, 9, 10, 11, 13, 14, 15, 16)

CLASS_NAMES = ("Interventional", "Angiography")
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass
class BookingTimetable:
    """Feasibility mask for weekly booking blocks.

    feasible_qik shape = (2, 40)
    - row 0 = Interventional
    - row 1 = Angiography
    - 40 columns = 40 weekday blocks in one week

    Entry 1 means booking is allowed.
    Entry 0 means booking is not allowed.
    """

    name: str
    feasible_qik: np.ndarray


@dataclass
class WeeklySchedulePolicy:
    """A full scheduling policy = timetable + Qik."""

    qik: np.ndarray
    timetable: BookingTimetable


@dataclass
class Patient:
    """Patient record stored throughout the simulation."""

    patient_id: int
    category: str
    order_arrival_time: float
    prep_type: str
    prep_start_time: float
    prep_end_time: float
    ready_to_schedule_time: float
    scheduled_time: Optional[float] = None
    slot_label: Optional[str] = None
    weekly_block_index: Optional[int] = None
    lateness: Optional[float] = None
    waiting_room_arrival: Optional[float] = None
    actual_proc_start: Optional[float] = None
    actual_proc_end: Optional[float] = None


@dataclass
class AppointmentSlot:
    """One booking slot generated from Qik and the timetable."""

    time: float
    label: str
    weekly_block_index: int



def sample_from_spec(spec: DistributionSpec, rng: np.random.Generator) -> float:
    """Sample one value from a fitted distribution specification."""
    dist = spec.dist.lower()
    p = spec.params

    if dist in {"deterministic", "constant"}:
        return float(p["value"])
    if dist == "empirical":
        samples = np.asarray(p["samples"], dtype=float)
        if samples.size == 0:
            raise ValueError("Empirical distribution received no samples.")
        return float(rng.choice(samples))
    if dist in {"uniform"}:
        return float(rng.uniform(float(p["low"]), float(p["high"])))
    if dist in {"exponential", "expon"}:
        return float(rng.exponential(float(p["mean"])))
    if dist == "gamma":
        return float(rng.gamma(shape=float(p["shape"]), scale=float(p["scale"])))
    if dist == "lognormal":
        return float(rng.lognormal(mean=float(p["mu"]), sigma=float(p["sigma"])))
    if dist == "weibull":
        return float(float(p.get("loc", 0.0)) + float(p["scale"]) * rng.weibull(float(p["shape"])))
    raise ValueError(f"Unsupported distribution: {spec.dist}")



def weekly_block_metadata() -> pd.DataFrame:
    """Create a table describing the 40 weekly booking blocks.

    Block order is:
    Monday    8 blocks
    Tuesday   8 blocks
    Wednesday 8 blocks
    Thursday  8 blocks
    Friday    8 blocks
    """
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    rows: List[Dict[str, Any]] = []
    block_index = 0
    for weekday in range(5):
        for start_hour in WORKDAY_START_HOURS:
            rows.append(
                {
                    "block_index": block_index,
                    "weekday": weekday,
                    "weekday_name": weekday_names[weekday],
                    "start_hour": start_hour,
                    "label": f"{weekday_names[weekday]} {start_hour:02d}:00",
                }
            )
            block_index += 1
    return pd.DataFrame(rows)



def apply_feasibility_to_qik(qik: np.ndarray, timetable: BookingTimetable) -> np.ndarray:
    """Force infeasible Qik entries to zero.

    If timetable says a class cannot be booked in a block, then that block's Qik
    entry must be zero regardless of what was originally proposed.
    """
    qik = np.asarray(qik, dtype=int).copy()
    qik[timetable.feasible_qik == 0] = 0
    return qik



def policy_from_qik(qik: np.ndarray, timetable: BookingTimetable) -> WeeklySchedulePolicy:
    """Build one policy object from qik + timetable."""
    qik = np.asarray(qik, dtype=int)
    return WeeklySchedulePolicy(qik=qik, timetable=timetable)



def qik_to_dataframe(qik: np.ndarray) -> pd.DataFrame:
    """Convert the 2 x 40 Qik array into a readable table."""
    qik = np.asarray(qik, dtype=int)
    df = weekly_block_metadata().copy()
    df["Q_interventional"] = qik[0, :]
    df["Q_angiography"] = qik[1, :]
    df["Q_total"] = df["Q_interventional"] + df["Q_angiography"]
    return df


class IROutpatientSchedulingSim:
    """Main simulation object."""

    def __init__(
        self,
        num_weeks: int,
        loaded_inputs: LoadedIRInputs,
        policy: WeeklySchedulePolicy,
        seed: int = 123,
        warmup_weeks: int = 0,
    ):
        self.num_weeks = int(num_weeks)
        self.warmup_weeks = int(warmup_weeks)
        if self.num_weeks <= 0:
            raise ValueError("num_weeks must be positive")
        if self.warmup_weeks < 0:
            raise ValueError("warmup_weeks must be non-negative")

        self.total_sim_weeks = self.warmup_weeks + self.num_weeks
        self.num_calendar_days = 7 * self.total_sim_weeks
        self.measurement_start_time = self.warmup_weeks * 7 * MINUTES_PER_DAY
        self.loaded_inputs = loaded_inputs
        self.policy = policy
        self.rng = np.random.default_rng(seed)

        # Unpack loaded inputs once so the rest of the code is shorter.
        self.arrival_inputs = loaded_inputs.arrival_inputs
        self.prep_probabilities = loaded_inputs.prep_probabilities
        self.prep_distributions = loaded_inputs.prep_distributions
        self.procedure_distributions = loaded_inputs.procedure_distributions
        self.late_delay_distributions = loaded_inputs.late_delay_distributions

        # Event calendar state.
        self.current_time = 0.0
        self.event_counter = 0
        self.event_calendar: List[Tuple[float, int, str, Any]] = []

        # One-room procedure area.
        self.room_busy = False

        self.waiting_room_queue: List[Patient] = []

        # waiting_room_len tracks the current queue size, while
        # max_waiting_room_len stores the peak queue size over the run.
        self.waiting_room_len = 0
        self.max_waiting_room_len = 0
        self.waiting_room_area = 0.0
        self.last_area_time = 0.0

        # Output storage.
        self.patient_counter = 0
        self.patients: List[Patient] = []
        self.booking_records: List[Dict[str, Any]] = []
        self.unscheduled_patients_total = 0
        self.unscheduled_patients_measured = 0

        # Overtime statistics.
        self.total_lunch_overtime = 0.0
        self.total_after_hours_overtime = 0.0
        self.total_overtime = 0.0

        # Monday-Friday day indices over the full horizon.
        self.working_day_indices = self._build_working_day_indices()

        # Expand the weekly Qik policy into a list of actual appointment slots.
        self.class_slots = self._build_class_slots()

        # Pointer telling us the next unused slot for each class.
        self.next_slot_index = {name: 0 for name in CLASS_NAMES}

    def _build_working_day_indices(self) -> List[int]:
        """Return the calendar-day indices corresponding to Monday-Friday only."""
        out: List[int] = []
        for week in range(self.total_sim_weeks):
            week_offset = 7 * week
            for weekday in range(5):
                out.append(week_offset + weekday)
        return out

    def _schedule_event(self, event_time: float, event_type: str, payload: Any) -> None:
        """Push one event into the min-heap event calendar."""
        heapq.heappush(self.event_calendar, (float(event_time), self.event_counter, event_type, payload))
        self.event_counter += 1

    def _update_waiting_room_area(self, new_time: float) -> None:
        """Update time-integrated waiting-room queue length for congestion Z3."""
        effective_start = max(self.last_area_time, self.measurement_start_time)
        if new_time > effective_start:
            self.waiting_room_area += self.waiting_room_len * (new_time - effective_start)
        self.last_area_time = new_time

    def _parse_bin_start_hours(self, bin_index: Sequence[Any], fallback_start_hour: int) -> List[int]:
        """Read hour labels such as '8:00-9:00' from the JSON keys."""
        hours: List[int] = []
        for idx, label in enumerate(bin_index):
            text = str(label)
            match = re.match(r"\s*(\d{1,2})\s*:\s*00", text)
            if match:
                hours.append(int(match.group(1)))
            else:
                hours.append(fallback_start_hour + idx)
        return hours

    def _piecewise_rate_per_hour(self, time_in_day_min: float, start_hours: Sequence[int], rates: Sequence[float]) -> float:
        """Return the correct hourly rate for a given minute within the workday."""
        hour = int(time_in_day_min // 60.0)
        for idx, start_hour in enumerate(start_hours):
            if start_hour <= hour < start_hour + 1:
                return float(rates[idx])
        return 0.0

    def _draw_prep_type(self, category: str) -> str:
        """Draw long/medium/short/no-prep using class-specific probabilities."""
        probs = self.prep_probabilities[category]
        labels = list(probs.keys())
        p = np.array([probs[label] for label in labels], dtype=float)
        p = p / p.sum()
        return str(self.rng.choice(labels, p=p))

    def _draw_prep_time(self, category: str, prep_type: str) -> float:
        """Draw preparation duration in minutes."""
        spec = self.prep_distributions[category][prep_type]
        return max(0.0, sample_from_spec(spec, self.rng))

    def _draw_procedure_time(self, category: str) -> float:
        """Draw actual procedure duration in minutes."""
        spec = self.procedure_distributions[category]
        return max(0.0, sample_from_spec(spec, self.rng))

    def _draw_lateness(self, category: str) -> float:
        """Draw patient lateness in minutes after the booked appointment time."""
        spec = self.late_delay_distributions[category]
        return max(0.0, sample_from_spec(spec, self.rng))

    def _generate_interventional_arrivals_for_day(self, day_index: int) -> List[float]:
        """Generate one day's Interventional arrivals using NHPP thinning."""
        lambda_hat = self.arrival_inputs.interventional_lambda_hat
        start_hours = self._parse_bin_start_hours(lambda_hat.index, self.arrival_inputs.bin_start_hour)
        rates = [float(x) for x in lambda_hat.values]

        # lambda_max is the dominating rate used in the thinning algorithm.
        lambda_max = max(rates)
        work_start = min(start_hours) * MINUTES_PER_HOUR
        work_end = (max(start_hours) + 1) * MINUTES_PER_HOUR
        day_offset = day_index * MINUTES_PER_DAY

        arrivals: List[float] = []
        t = day_offset + work_start
        while True:
            # Candidate interarrival gap from Exp(lambda_max).
            t += float(self.rng.exponential(MINUTES_PER_HOUR / lambda_max))
            if t >= day_offset + work_end:
                break

            # Accept candidate with probability lambda(t) / lambda_max.
            rate_t = self._piecewise_rate_per_hour(t - day_offset, start_hours, rates)
            if self.rng.random() <= rate_t / lambda_max:
                arrivals.append(t)
        return arrivals

    def _generate_angiography_arrivals(self) -> List[float]:
        """Generate Angiography arrivals using the Poisson-lognormal model."""
        fit = self.arrival_inputs.angiography_pln_fit
        lambda_hat = self.arrival_inputs.angiography_lambda_hat
        start_hours = self._parse_bin_start_hours(lambda_hat.index, self.arrival_inputs.bin_start_hour)

        arrivals: List[float] = []
        for day in self.working_day_indices:
            # Day-specific random effect theta.
            if self.rng.random() < float(fit["p_zero"]):
                theta = 0.0
            else:
                theta = float(self.rng.lognormal(mean=float(fit["mu"]), sigma=float(fit["sigma"])))

            # Hourly counts for this working day.
            counts = self.rng.poisson(theta * lambda_hat.values)
            day_offset = day * MINUTES_PER_DAY
            for col_idx, count in enumerate(counts):
                hour_start = start_hours[col_idx] * MINUTES_PER_HOUR
                for _ in range(int(count)):
                    # Place each arrival uniformly within its hour bin.
                    arrivals.append(day_offset + hour_start + float(self.rng.uniform(0.0, MINUTES_PER_HOUR)))
        return arrivals

    def _build_class_slots(self) -> Dict[str, List[AppointmentSlot]]:
        """Expand weekly Qik into a time-ordered list of actual appointment slots.

        If qik[0, 12] = 3, then this method creates 3 separate Interventional slots
        at the time corresponding to weekly block 12 for each simulated week.
        """
        class_slots: Dict[str, List[AppointmentSlot]] = {name: [] for name in CLASS_NAMES}
        meta = weekly_block_metadata()
        qik = self.policy.qik
        feasible = self.policy.timetable.feasible_qik

        for week in range(self.total_sim_weeks):
            for _, row in meta.iterrows():
                k = int(row["block_index"])
                weekday = int(row["weekday"])
                start_hour = int(row["start_hour"])
                day_index = 7 * week + weekday
                block_time = day_index * MINUTES_PER_DAY + start_hour * MINUTES_PER_HOUR

                for class_name in CLASS_NAMES:
                    class_idx = CLASS_TO_INDEX[class_name]
                    if feasible[class_idx, k] == 1:
                        capacity = int(qik[class_idx, k])
                        for _ in range(capacity):
                            class_slots[class_name].append(
                                AppointmentSlot(
                                    time=block_time,
                                    label=str(row["label"]),
                                    weekly_block_index=k,
                                )
                            )

        # Sort so that "book earliest feasible future slot" becomes easy.
        for class_name in CLASS_NAMES:
            class_slots[class_name].sort(key=lambda slot: slot.time)
        return class_slots

    def schedule_initial_events(self) -> None:
        """Place all order-arrival events into the event calendar."""
        for day in self.working_day_indices:
            for arrival_time in self._generate_interventional_arrivals_for_day(day):
                self._schedule_event(arrival_time, "PatientArrival", "Interventional")
        for arrival_time in self._generate_angiography_arrivals():
            self._schedule_event(arrival_time, "PatientArrival", "Angiography")

    def handle_patient_arrival(self, category: str) -> None:
        """At order arrival, assign prep type and sample prep completion time."""
        self.patient_counter += 1
        prep_type = self._draw_prep_type(category)
        prep_start_time = self.current_time
        prep_end_time = prep_start_time + self._draw_prep_time(category, prep_type)

        patient = Patient(
            patient_id=self.patient_counter,
            category=category,
            order_arrival_time=self.current_time,
            prep_type=prep_type,
            prep_start_time=prep_start_time,
            prep_end_time=prep_end_time,
            ready_to_schedule_time=prep_end_time,
        )
        self.patients.append(patient)

        # Admin attempts booking when preparation ends.
        self._schedule_event(prep_end_time, "ReadyToSchedule", patient)

    def _book_patient(self, patient: Patient) -> None:
        """Book the patient into the earliest future slot for that class.

        FCFS logic is implemented by event time:
        the patient becomes eligible at ready_to_schedule_time,
        and booking happens at that moment.
        """
        slots = self.class_slots[patient.category]
        idx = self.next_slot_index[patient.category]

        # Skip any slot that is already in the past relative to when this patient
        # becomes ready to schedule.
        while idx < len(slots) and slots[idx].time < patient.ready_to_schedule_time:
            idx += 1
        self.next_slot_index[patient.category] = idx

        # No feasible future slot remains.
        if idx >= len(slots):
            self.unscheduled_patients_total += 1
            if patient.order_arrival_time >= self.measurement_start_time:
                self.unscheduled_patients_measured += 1
            return

        slot = slots[idx]
        self.next_slot_index[patient.category] += 1
        patient.scheduled_time = slot.time
        patient.slot_label = slot.label
        patient.weekly_block_index = slot.weekly_block_index

        # After booking, draw lateness and determine physical arrival to the waiting area.
        patient.lateness = self._draw_lateness(patient.category)
        patient.waiting_room_arrival = patient.scheduled_time + patient.lateness

        self.booking_records.append(
            {
                "patient_id": patient.patient_id,
                "category": patient.category,
                "ready_to_schedule_time": patient.ready_to_schedule_time,
                "scheduled_time": patient.scheduled_time,
                "slot_label": patient.slot_label,
                "weekly_block_index": patient.weekly_block_index,
                "lateness": patient.lateness,
            }
        )

        self._schedule_event(patient.waiting_room_arrival, "WaitingRoomArrival", patient)

    def handle_ready_to_schedule(self, patient: Patient) -> None:
        """Preparation is finished; book the patient now."""
        self._book_patient(patient)

    def _start_procedure(self, patient: Patient) -> None:
        """Start service in the one available procedure room."""
        self.room_busy = True

        patient.actual_proc_start = self.current_time
        duration = self._draw_procedure_time(patient.category)
        self._schedule_event(self.current_time + duration, "EndProcedure", patient)

    def handle_waiting_room_arrival(self, patient: Patient) -> None:
        """Patient physically arrives for the booked appointment."""
        if not self.room_busy:
            self._start_procedure(patient)
        else:
            self.waiting_room_queue.append(patient)
            self.waiting_room_len += 1
            self.max_waiting_room_len = max(self.max_waiting_room_len, self.waiting_room_len)

    def handle_end_procedure(self, patient: Patient) -> None:
        """Finish current procedure, free the room, and pull next waiting patient if any."""
        patient.actual_proc_end = self.current_time

        lunch_ot, after_ot, total_ot = compute_nonworking_overlap(
            patient.actual_proc_start,
            patient.actual_proc_end,
            measurement_start_time=self.measurement_start_time,
        )
        self.total_lunch_overtime += lunch_ot
        self.total_after_hours_overtime += after_ot
        self.total_overtime += total_ot

        if self.waiting_room_queue:
            next_patient = self.waiting_room_queue.pop(0)
            self.waiting_room_len -= 1
            self._start_procedure(next_patient)
        else:
            self.room_busy = False

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run one full simulation replication."""
        self.schedule_initial_events()

        while self.event_calendar:
            event_time, _, event_type, payload = heapq.heappop(self.event_calendar)

            # Update congestion area before the state changes at the new event time.
            self._update_waiting_room_area(event_time)
            self.current_time = event_time

            if event_type == "PatientArrival":
                self.handle_patient_arrival(payload)
            elif event_type == "ReadyToSchedule":
                self.handle_ready_to_schedule(payload)
            elif event_type == "WaitingRoomArrival":
                self.handle_waiting_room_arrival(payload)
            elif event_type == "EndProcedure":
                self.handle_end_procedure(payload)
            else:
                raise ValueError(f"Unknown event type: {event_type}")

        patients_df = patient_records_to_dataframe(
            self.patients,
            measurement_start_time=self.measurement_start_time,
        )
        bookings_df = pd.DataFrame(self.booking_records)
        if not bookings_df.empty:
            if patients_df.empty:
                bookings_df = bookings_df.iloc[0:0].copy()
            else:
                bookings_df = bookings_df[
                    bookings_df["patient_id"].isin(patients_df["patient_id"])
                ].copy()
        summary_df = build_summary_dataframe(self, patients_df)
        return summary_df, patients_df, bookings_df



def compute_nonworking_overlap(
    start_time: float,
    end_time: float,
    measurement_start_time: float = 0.0,
) -> Tuple[float, float, float]:
    """Compute how much of a procedure overlaps lunch or after-hours time."""
    start_time = max(start_time, measurement_start_time)
    if end_time <= start_time:
        return 0.0, 0.0, 0.0

    lunch_overlap = 0.0
    after_hours_overlap = 0.0

    start_day = int(start_time // MINUTES_PER_DAY)
    end_day = int(end_time // MINUTES_PER_DAY)

    for day in range(start_day, end_day + 1):
        day_offset = day * MINUTES_PER_DAY
        lunch_start = day_offset + 12.0 * 60.0
        lunch_end = day_offset + 13.0 * 60.0
        after_start = day_offset + 17.0 * 60.0
        day_end = day_offset + 24.0 * 60.0

        lunch_overlap += max(0.0, min(end_time, lunch_end) - max(start_time, lunch_start))
        after_hours_overlap += max(0.0, min(end_time, day_end) - max(start_time, after_start))

    return lunch_overlap, after_hours_overlap, lunch_overlap + after_hours_overlap



def patient_records_to_dataframe(
    patients: Sequence[Patient],
    measurement_start_time: float = 0.0,
) -> pd.DataFrame:
    """Convert all patient objects into one analysis table."""
    rows: List[Dict[str, Any]] = []
    for p in patients:
        rows.append(
            {
                "patient_id": p.patient_id,
                "category": p.category,
                "order_arrival_time": p.order_arrival_time,
                "prep_type": p.prep_type,
                "prep_start_time": p.prep_start_time,
                "prep_end_time": p.prep_end_time,
                "ready_to_schedule_time": p.ready_to_schedule_time,
                "scheduled_time": p.scheduled_time,
                "slot_label": p.slot_label,
                "weekly_block_index": p.weekly_block_index,
                "lateness": p.lateness,
                "waiting_room_arrival": p.waiting_room_arrival,
                "actual_proc_start": p.actual_proc_start,
                "actual_proc_end": p.actual_proc_end,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Derived performance measures at the patient level.
    df["prep_duration"] = df["prep_end_time"] - df["prep_start_time"]
    df["booking_wait"] = df["scheduled_time"] - df["ready_to_schedule_time"]
    df["waiting_room_wait"] = df["actual_proc_start"] - df["waiting_room_arrival"]
    df["total_wait_to_proc_start"] = df["actual_proc_start"] - df["ready_to_schedule_time"]
    df["procedure_duration"] = df["actual_proc_end"] - df["actual_proc_start"]
    if measurement_start_time > 0.0:
        df = df[df["order_arrival_time"] >= measurement_start_time].copy()
    return df



def build_summary_dataframe(model: IROutpatientSchedulingSim, patients_df: pd.DataFrame) -> pd.DataFrame:
    """Build one-row summary output with the three objective components and H."""
    if patients_df.empty:
        z1 = 0.0
        mean_booking_wait = 0.0
        mean_prep_duration = 0.0
        mean_lateness = 0.0
        mean_procedure_duration = 0.0
        scheduled_count = 0
        completed_count = 0
    else:
        # Z1 = average wait contribution over measured patients.
        # Completed patients use the observed wait-to-procedure-start measure.
        # Unscheduled patients use the censored wait
        #     simulation_end_time - order_arrival_time
        # only when that censored wait exceeds 28 days; otherwise they are
        # excluded from the Z1 average.
        z1_wait = patients_df["total_wait_to_proc_start"].copy()
        unscheduled_mask = patients_df["actual_proc_start"].isna()
        censored_wait = model.current_time - patients_df.loc[unscheduled_mask, "order_arrival_time"]
        z1_wait.loc[unscheduled_mask] = np.nan
        above_threshold_mask = censored_wait > 28.0 * MINUTES_PER_DAY
        z1_wait.loc[censored_wait.index[above_threshold_mask]] = censored_wait.loc[above_threshold_mask]
        z1 = float(z1_wait.mean())
        mean_booking_wait = float(patients_df["booking_wait"].dropna().mean())
        mean_prep_duration = float(patients_df["prep_duration"].dropna().mean())
        mean_lateness = float(patients_df["lateness"].dropna().mean())
        mean_procedure_duration = float(patients_df["procedure_duration"].dropna().mean())
        scheduled_count = int(patients_df["scheduled_time"].notna().sum())
        completed_count = int(patients_df["actual_proc_end"].notna().sum())

    # Z2 = average overtime per week, reported in hours.
    z2_minutes_per_week = float(model.total_overtime / model.num_weeks)
    z2 = z2_minutes_per_week / MINUTES_PER_HOUR

    # Z3 = maximum number of patients waiting in the waiting-room queue.
    z3 = float(model.max_waiting_room_len)

    # Overall weighted objective.
    # Z1 is scaled directly in days relative to the 28-day target.
    z1_days = z1 / MINUTES_PER_DAY
    z1_term = z1_days / 28.0

    # Z2 is already reported in hours/week, so scale that directly.
    z2_term = z2 / 2.5

    # Z3 is scaled relative to a target max waiting-room queue length of 2.
    z3_term = z3 / 2.0

    H = 0.6 * z1_term + 0.2 * z2_term + 0.2 * z3_term

    return pd.DataFrame(
        [
            {
                "timetable": model.policy.timetable.name,
                "num_weeks": model.num_weeks,
                "warmup_weeks": model.warmup_weeks,
                "total_sim_weeks": model.total_sim_weeks,
                "scheduled_count": scheduled_count,
                "completed_count": completed_count,
                "unscheduled_count": model.unscheduled_patients_measured,
                "unscheduled_count_total": model.unscheduled_patients_total,
                "mean_prep_duration": mean_prep_duration,
                "mean_booking_wait": mean_booking_wait,
                "mean_lateness": mean_lateness,
                "mean_procedure_duration": mean_procedure_duration,
                "Z1_wait_time": z1,
                "Z2_overtime": z2,
                "Z3_congestion": z3,
                "H": H,
                "total_overtime_min": model.total_overtime,
                "lunch_overtime_min": model.total_lunch_overtime,
                "after_hours_overtime_min": model.total_after_hours_overtime,
            }
        ]
    )



def run_replications(
    num_replications: int,
    num_weeks: int,
    loaded_inputs: LoadedIRInputs,
    policy: WeeklySchedulePolicy,
    base_seed: int = 123,
    warmup_weeks: int = 0,
) -> pd.DataFrame:
    """Run several independent replications and stack the summary rows."""
    rows: List[pd.DataFrame] = []
    for rep in range(num_replications):
        model = IROutpatientSchedulingSim(
            num_weeks=num_weeks,
            loaded_inputs=loaded_inputs,
            policy=policy,
            seed=base_seed + rep,
            warmup_weeks=warmup_weeks,
        )
        summary_df, _, _ = model.run()
        out = summary_df.copy()
        out.insert(0, "replication", rep + 1)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)



def summarize_replications(rep_df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple across-replication summary statistics."""
    metrics = ["Z1_wait_time", "Z2_overtime", "Z3_congestion", "H"]
    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        rows.append(
            {
                "metric": metric,
                "mean": float(rep_df[metric].mean()),
                "std": float(rep_df[metric].std(ddof=1)) if len(rep_df) > 1 else 0.0,
                "min": float(rep_df[metric].min()),
                "max": float(rep_df[metric].max()),
            }
        )
    return pd.DataFrame(rows)
