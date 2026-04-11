from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import SimClasses
import SimFunctions
import SimRNG

MINUTES_PER_HOUR = 60.0
MINUTES_PER_DAY = 24.0 * 60.0


# =========================================================
# CONFIG OBJECTS
# =========================================================
@dataclass
class SlotRule:
    start_hour: int
    allowed_categories: Tuple[str, ...]
    capacity: int
    label: str = ""


@dataclass
class ArrivalInputs:
    interventional_lambda_hat: pd.Series
    angiography_lambda_hat: pd.Series
    angiography_pln_fit: Dict[str, Any]
    bin_start_hour: int = 8
    bin_end_hour: int = 17


@dataclass
class DistributionSpec:
    dist: str
    params: Dict[str, float]


# =========================================================
# NORMALIZATION HELPERS
# =========================================================
def normalize_category(text: str) -> str:
    text = str(text).strip().lower()
    if text == "interventional":
        return "Interventional"
    if text == "angiography":
        return "Angiography"
    return str(text).strip().title()


def normalize_prep_name(text: str) -> str:
    text = str(text).strip().lower().replace(" ", "_")
    alias_map = {
        "long_prepare": "long_prepare",
        "medium_prepare": "medium_prepare",
        "short_prepare": "short_prepare",
        "no_prepare_required": "no_prepare_required",
    }
    return alias_map[text]


# =========================================================
# JSON LOADERS
# =========================================================
def load_arrival_inputs_from_json(json_path: str = "arrival_model_params.json") -> ArrivalInputs:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    interventional = payload["interventional_nhpp"]
    angiography = payload["angiography_pln"]

    return ArrivalInputs(
        interventional_lambda_hat=pd.Series(interventional["lambda_hat"]),
        angiography_lambda_hat=pd.Series(angiography["lambda_hat"]),
        angiography_pln_fit={
            "p_zero": float(angiography["pln_fit"]["p_zero"]),
            "mu": float(angiography["pln_fit"]["mu"]),
            "sigma": float(angiography["pln_fit"]["sigma"]),
        },
        bin_start_hour=int(interventional["start_hour"]),
        bin_end_hour=int(interventional["end_hour"]),
    )


def _spec_from_service_payload(payload: Dict[str, Any], multiplier: float) -> DistributionSpec:
    dist_name = str(payload["distribution"]).strip().lower()
    params = payload["parameters"]

    if dist_name == "gamma":
        return DistributionSpec(
            "gamma",
            {
                "shape": float(params["shape"]),
                "scale": float(params["scale"]) * multiplier,
            },
        )

    if dist_name == "lognormal":
        return DistributionSpec(
            "lognormal",
            {
                "mu": math.log(float(params["scale"])),
                "sigma": float(params["sigma"]),
            },
        )

    if dist_name == "weibull":
        return DistributionSpec(
            "weibull",
            {
                "shape": float(params["shape"]),
                "scale": float(params["scale"]) * multiplier,
                "loc": float(params.get("loc", 0.0)) * multiplier,
            },
        )

    if dist_name == "exponential":
        return DistributionSpec(
            "exponential",
            {
                "mean": float(params["scale"]) * multiplier,
            },
        )

    raise ValueError(f"Unsupported service distribution in JSON: {payload['distribution']}")


def load_service_inputs_from_json(
    json_path: str = "services rate.json",
    short_prepare_fallback: str = "category_range_uniform",
    no_prepare_value_min: float = 0.0,
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, DistributionSpec]],
    Dict[str, DistributionSpec],
    Dict[str, DistributionSpec],
    List[Dict[str, str]],
]:
    """
    Returns
    -------
    prep_probabilities
    prep_distributions
    procedure_distributions
    late_delay_distributions
    prep_fallback_groups

    Notes
    -----
    The JSON provides accepted parametric fits for long/medium prep groups only.
    For groups flagged as empirical fallback, this loader keeps the model executable by:
    - no_prepare_required -> deterministic 0 minutes
    - short_prepare -> optional fallback, default Uniform(0.3, 7) days in minutes

    That short-prepare fallback is a modeling assumption, not a fitted parametric result.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    prep_probabilities_raw = payload["preparation_category_percentages"]
    prep_probabilities: Dict[str, Dict[str, float]] = {}
    for cls, probs in prep_probabilities_raw.items():
        cls_norm = normalize_category(cls)
        prep_probabilities[cls_norm] = {
            normalize_prep_name(k): float(v) / 100.0
            for k, v in probs.items()
        }

    # Accepted preparation fits from JSON
    prep_distributions: Dict[str, Dict[str, DistributionSpec]] = {
        "Angiography": {},
        "Interventional": {},
    }
    for cls, group_payload in payload["preparation_duration_days"].items():
        cls_norm = normalize_category(cls)
        for prep_name, fit_payload in group_payload.items():
            prep_distributions[cls_norm][normalize_prep_name(prep_name)] = _spec_from_service_payload(
                fit_payload,
                multiplier=MINUTES_PER_DAY,
            )

    # Fallback groups explicitly called out by the fitting script
    prep_fallback_groups: List[Dict[str, str]] = []
    for item in payload.get("preparation_empirical_fallback_groups", []):
        cls_norm = normalize_category(item["classification"])
        prep_name = normalize_prep_name(item["category_prepared"])
        prep_fallback_groups.append(
            {
                "classification": cls_norm,
                "category_prepared": prep_name,
                "reason": str(item["reason"]),
            }
        )

    for cls in ("Angiography", "Interventional"):
        prep_distributions.setdefault(cls, {})
        prep_distributions[cls]["no_prepare_required"] = DistributionSpec(
            "deterministic",
            {"value": float(no_prepare_value_min)},
        )

        if "short_prepare" not in prep_distributions[cls]:
            if short_prepare_fallback == "category_range_uniform":
                prep_distributions[cls]["short_prepare"] = DistributionSpec(
                    "uniform",
                    {
                        "low": 0.3 * MINUTES_PER_DAY,
                        "high": 7.0 * MINUTES_PER_DAY,
                    },
                )
            elif short_prepare_fallback == "raise":
                raise ValueError(
                    f"Missing fitted short_prepare distribution for {cls}. "
                    "The provided JSON marks this group as empirical fallback only."
                )
            else:
                raise ValueError(f"Unsupported short_prepare_fallback: {short_prepare_fallback}")

    # Procedure duration: only one accepted fit in JSON, use it for both categories.
    procedure_all_payload = payload["procedure_duration_hours"]["All"]
    procedure_spec = _spec_from_service_payload(procedure_all_payload, multiplier=MINUTES_PER_HOUR)
    procedure_distributions = {
        "Angiography": procedure_spec,
        "Interventional": procedure_spec,
    }

    # Late delay distributions: JSON contains fitted positive late-time distributions.
    late_delay_distributions: Dict[str, DistributionSpec] = {}
    for key, fit_payload in payload["late_time_minutes"].items():
        cls = key.split("=", 1)[1]
        cls_norm = normalize_category(cls)
        late_delay_distributions[cls_norm] = _spec_from_service_payload(
            fit_payload,
            multiplier=1.0,
        )

    return (
        prep_probabilities,
        prep_distributions,
        procedure_distributions,
        late_delay_distributions,
        prep_fallback_groups,
    )


# =========================================================
# DISTRIBUTION HELPERS
# =========================================================
def sample_from_spec(spec: DistributionSpec, np_rng: np.random.Generator, stream: int) -> float:
    dist = spec.dist.strip().lower()
    p = spec.params

    if dist in {"deterministic", "constant"}:
        return float(p["value"])

    if dist in {"exponential", "expon"}:
        return float(SimRNG.Expon(float(p["mean"]), stream))

    if dist == "erlang":
        return float(SimRNG.Erlang(int(p["phases"]), float(p["mean"]), stream))

    if dist == "uniform":
        return float(SimRNG.Uniform(float(p["low"]), float(p["high"]), stream))

    if dist == "triangular":
        return float(
            SimRNG.Triangular(
                float(p["low"]),
                float(p["mode"]),
                float(p["high"]),
                stream,
            )
        )

    if dist == "normal":
        return float(SimRNG.Normal(float(p["mean"]), float(p["variance"]), stream))

    if dist == "gamma":
        return float(np_rng.gamma(shape=float(p["shape"]), scale=float(p["scale"])))

    if dist == "lognormal":
        return float(np_rng.lognormal(mean=float(p["mu"]), sigma=float(p["sigma"])))

    if dist == "weibull":
        return float(float(p.get("loc", 0.0)) + float(p["scale"]) * np_rng.weibull(float(p["shape"])))

    raise ValueError(f"Unsupported distribution: {spec.dist}")


# =========================================================
# PATIENT OBJECT
# =========================================================
class Patient(SimClasses.Entity):
    def __init__(self, patient_id: int, category: str):
        super().__init__()
        self.patient_id = patient_id
        self.category = category
        self.prep_type: Optional[str] = None
        self.prep_queue_enter: Optional[float] = None
        self.prep_start: Optional[float] = None
        self.prep_end: Optional[float] = None
        self.scheduling_queue_enter: Optional[float] = None
        self.scheduled_time: Optional[float] = None
        self.slot_label: Optional[str] = None
        self.lateness: Optional[float] = None
        self.waiting_room_arrival: Optional[float] = None
        self.actual_proc_start: Optional[float] = None
        self.actual_proc_end: Optional[float] = None
        self.room_id: Optional[int] = None


# =========================================================
# MAIN MODEL
# =========================================================
class IRSchedulingPythonSim:
    def __init__(
        self,
        num_days: int,
        arrival_inputs: ArrivalInputs,
        prep_probabilities: Dict[str, Dict[str, float]],
        prep_distributions: Dict[str, Dict[str, DistributionSpec]],
        procedure_distributions: Dict[str, DistributionSpec],
        late_delay_distributions: Dict[str, DistributionSpec],
        slot_rules: Sequence[SlotRule],
        num_procedure_rooms: int,
        prep_capacities: Dict[str, int],
        seed: int = 123,
    ):
        self.num_days = int(num_days)
        self.arrival_inputs = arrival_inputs
        self.prep_probabilities = prep_probabilities
        self.prep_distributions = prep_distributions
        self.procedure_distributions = procedure_distributions
        self.late_delay_distributions = late_delay_distributions
        self.slot_rules = list(slot_rules)
        self.num_procedure_rooms = int(num_procedure_rooms)
        self.prep_capacities = prep_capacities
        self.seed = int(seed)

        self.np_rng = np.random.default_rng(seed)
        SimRNG.InitializeRNSeed()
        for stream in range(1, 11):
            SimRNG.lcgrandst(self.seed + 100000 * stream, stream)

        self._setup_objects()

    def _setup_objects(self) -> None:
        self.Calendar = SimClasses.EventCalendar()
        self.ProcedureRooms = SimClasses.Resource()
        self.ProcedureRooms.SetUnits(self.num_procedure_rooms)

        self.PrepResources: Dict[str, SimClasses.Resource] = {}
        self.PrepQueues: Dict[str, SimClasses.FIFOQueue] = {}
        for prep_type, cap in self.prep_capacities.items():
            res = SimClasses.Resource()
            res.SetUnits(int(cap))
            self.PrepResources[prep_type] = res
            self.PrepQueues[prep_type] = SimClasses.FIFOQueue()

        self.ScheduleQueues = {
            "Interventional": SimClasses.FIFOQueue(),
            "Angiography": SimClasses.FIFOQueue(),
        }
        self.WaitingRoomQueue = SimClasses.FIFOQueue()

        self.PatientWaitTime = SimClasses.DTStat()
        self.SchedulingQueueWait = SimClasses.DTStat()
        self.WaitingRoomWait = SimClasses.DTStat()
        self.LatenessStat = SimClasses.DTStat()
        self.ProcedureDurationStat = SimClasses.DTStat()

        self.available_room_ids: List[int] = list(range(1, self.num_procedure_rooms + 1))
        self.patients: List[Patient] = []
        self.slot_assignments: List[Dict[str, Any]] = []
        self.patient_counter = 0

        self.total_lunch_overtime = 0.0
        self.total_after_hours_overtime = 0.0
        self.total_overtime = 0.0

    def _stream_for(self, name: str) -> int:
        mapping = {
            "interventional_gap": 1,
            "interventional_accept": 2,
            "angiography_day_factor": 3,
            "prep_route": 4,
            "prep_time": 5,
            "procedure_time": 6,
            "late_delay": 7,
            "uniform_time": 8,
        }
        return mapping[name]

    def _choose_prep_type(self, category: str) -> str:
        probs = self.prep_probabilities[category]
        labels = list(probs.keys())
        p = np.array([float(probs[k]) for k in labels], dtype=float)
        p = p / p.sum()
        return str(self.np_rng.choice(labels, p=p))

    def _sample_prep_time(self, patient: Patient) -> float:
        spec = self.prep_distributions[patient.category][patient.prep_type]
        return max(0.0, sample_from_spec(spec, self.np_rng, self._stream_for("prep_time")))

    def _sample_procedure_time(self, patient: Patient) -> float:
        spec = self.procedure_distributions[patient.category]
        return max(0.0, sample_from_spec(spec, self.np_rng, self._stream_for("procedure_time")))

    def _sample_lateness(self, patient: Patient) -> float:
        spec = self.late_delay_distributions[patient.category]
        return max(0.0, sample_from_spec(spec, self.np_rng, self._stream_for("late_delay")))

    @staticmethod
    def _parse_bin_start_hours(bin_index: Sequence[Any], fallback_start_hour: int) -> List[int]:
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
        hour = int(time_in_day_min // 60.0)
        for idx, start_hour in enumerate(start_hours):
            if start_hour <= hour < start_hour + 1:
                return float(rates[idx])
        return 0.0

    def _generate_interventional_arrivals_one_day_thinning(self, day_index: int) -> List[float]:
        lambda_hat = self.arrival_inputs.interventional_lambda_hat
        start_hours = self._parse_bin_start_hours(lambda_hat.index, self.arrival_inputs.bin_start_hour)
        rates = [float(x) for x in lambda_hat.values]

        if len(rates) == 0 or max(rates) <= 0:
            return []

        lambda_max = max(rates)
        work_start = min(start_hours) * MINUTES_PER_HOUR
        work_end = (max(start_hours) + 1) * MINUTES_PER_HOUR
        day_offset = day_index * MINUTES_PER_DAY

        arrivals: List[float] = []
        t = day_offset + work_start

        while True:
            gap = SimRNG.Expon(MINUTES_PER_HOUR / lambda_max, self._stream_for("interventional_gap"))
            t += gap
            if t >= day_offset + work_end:
                break

            rate_t = self._piecewise_rate_per_hour(t - day_offset, start_hours, rates)
            if rate_t <= 0:
                continue

            accept_prob = rate_t / lambda_max
            if SimRNG.lcgrand(self._stream_for("interventional_accept")) <= accept_prob:
                arrivals.append(float(t))

        return arrivals

    def _generate_angiography_arrivals_all_days(self) -> pd.DataFrame:
        fit = self.arrival_inputs.angiography_pln_fit
        lambda_hat = self.arrival_inputs.angiography_lambda_hat

        out = np.zeros((self.num_days, len(lambda_hat)), dtype=int)
        for d in range(self.num_days):
            theta = 0.0 if self.np_rng.uniform() < float(fit["p_zero"]) else self.np_rng.lognormal(
                mean=float(fit["mu"]),
                sigma=float(fit["sigma"]),
            )
            out[d, :] = self.np_rng.poisson(theta * lambda_hat.values)

        return pd.DataFrame(out, columns=lambda_hat.index)

    def schedule_initial_arrivals(self) -> None:
        for day in range(self.num_days):
            arrivals = self._generate_interventional_arrivals_one_day_thinning(day)
            for arr_time in arrivals:
                SimFunctions.SchedulePlus(self.Calendar, "PatientArrival", arr_time, "Interventional")

        sim_angiography = self._generate_angiography_arrivals_all_days()
        start_hours = self._parse_bin_start_hours(
            sim_angiography.columns,
            self.arrival_inputs.bin_start_hour,
        )

        for day in range(self.num_days):
            day_offset = day * MINUTES_PER_DAY
            for col_idx, _ in enumerate(sim_angiography.columns):
                count = int(sim_angiography.iloc[day, col_idx])
                if count <= 0:
                    continue

                hour_start = start_hours[col_idx] * MINUTES_PER_HOUR
                for _ in range(count):
                    within_hour = SimRNG.Uniform(0.0, MINUTES_PER_HOUR, self._stream_for("uniform_time"))
                    arr_time = day_offset + hour_start + within_hour
                    SimFunctions.SchedulePlus(self.Calendar, "PatientArrival", arr_time, "Angiography")

    def schedule_slot_open_events(self) -> None:
        for day in range(self.num_days):
            day_offset = day * MINUTES_PER_DAY
            for rule in self.slot_rules:
                event_time = day_offset + rule.start_hour * MINUTES_PER_HOUR
                SimFunctions.SchedulePlus(self.Calendar, "SlotOpen", event_time, rule)

    def handle_patient_arrival(self, category: str) -> None:
        self.patient_counter += 1
        patient = Patient(patient_id=self.patient_counter, category=category)
        patient.prep_type = self._choose_prep_type(category)
        self.patients.append(patient)

        if patient.prep_type == "no_prepare_required":
            patient.prep_start = SimClasses.Clock
            patient.prep_end = SimClasses.Clock
            patient.scheduling_queue_enter = SimClasses.Clock
            self.ScheduleQueues[patient.category].Add(patient)
            return

        prep_resource = self.PrepResources[patient.prep_type]
        prep_queue = self.PrepQueues[patient.prep_type]

        if prep_resource.Seize(1):
            patient.prep_start = SimClasses.Clock
            prep_time = self._sample_prep_time(patient)
            SimFunctions.SchedulePlus(self.Calendar, "EndPrep", prep_time, patient)
        else:
            patient.prep_queue_enter = SimClasses.Clock
            prep_queue.Add(patient)

    def handle_end_prep(self, patient: Patient) -> None:
        patient.prep_end = SimClasses.Clock
        patient.scheduling_queue_enter = SimClasses.Clock
        self.ScheduleQueues[patient.category].Add(patient)

        prep_type = patient.prep_type
        prep_resource = self.PrepResources[prep_type]
        prep_queue = self.PrepQueues[prep_type]
        prep_resource.Free(1)

        if prep_queue.NumQueue() > 0:
            next_patient = prep_queue.Remove()
            prep_resource.Seize(1)
            next_patient.prep_start = SimClasses.Clock
            prep_time = self._sample_prep_time(next_patient)
            SimFunctions.SchedulePlus(self.Calendar, "EndPrep", prep_time, next_patient)

    def _peek_oldest_eligible_patient(self, allowed_categories: Sequence[str]) -> Optional[Patient]:
        candidates: List[Patient] = []
        for cat in allowed_categories:
            q = self.ScheduleQueues[cat]
            if q.NumQueue() > 0:
                candidates.append(q.ThisQueue[0])

        if not candidates:
            return None

        candidates.sort(key=lambda p: (p.scheduling_queue_enter, p.CreateTime, p.patient_id))
        return candidates[0]

    def _remove_specific_patient_from_schedule_queue(self, patient: Patient) -> None:
        q = self.ScheduleQueues[patient.category]
        q.ThisQueue.remove(patient)
        q.WIP.Record(float(q.NumQueue()))

    def handle_slot_open(self, rule: SlotRule) -> None:
        slot_label = rule.label if rule.label else f"{rule.start_hour:02d}:00"

        for _ in range(int(rule.capacity)):
            patient = self._peek_oldest_eligible_patient(rule.allowed_categories)
            if patient is None:
                break

            self._remove_specific_patient_from_schedule_queue(patient)
            patient.scheduled_time = SimClasses.Clock
            patient.slot_label = slot_label
            sched_wait = patient.scheduled_time - patient.scheduling_queue_enter
            self.SchedulingQueueWait.Record(sched_wait)

            patient.lateness = self._sample_lateness(patient)
            self.LatenessStat.Record(patient.lateness)

            self.slot_assignments.append(
                {
                    "slot_time": SimClasses.Clock,
                    "slot_label": slot_label,
                    "allowed_categories": "|".join(rule.allowed_categories),
                    "slot_capacity": int(rule.capacity),
                    "patient_id": patient.patient_id,
                    "patient_category": patient.category,
                    "lateness": patient.lateness,
                }
            )

            SimFunctions.SchedulePlus(self.Calendar, "WaitingRoomArrival", patient.lateness, patient)

    def handle_waiting_room_arrival(self, patient: Patient) -> None:
        patient.waiting_room_arrival = SimClasses.Clock

        if self.ProcedureRooms.Seize(1):
            room_id = self.available_room_ids.pop(0)
            self._start_procedure(patient, room_id, came_from_waiting_room=False)
        else:
            self.WaitingRoomQueue.Add(patient)

    def _start_procedure(self, patient: Patient, room_id: int, came_from_waiting_room: bool) -> None:
        patient.actual_proc_start = SimClasses.Clock
        patient.room_id = room_id

        if came_from_waiting_room:
            self.WaitingRoomWait.Record(patient.actual_proc_start - patient.waiting_room_arrival)
        else:
            self.WaitingRoomWait.Record(0.0)

        total_wait = patient.actual_proc_start - patient.CreateTime
        self.PatientWaitTime.Record(total_wait)

        proc_time = self._sample_procedure_time(patient)
        self.ProcedureDurationStat.Record(proc_time)
        SimFunctions.SchedulePlus(self.Calendar, "EndProcedure", proc_time, patient)

    def handle_end_procedure(self, patient: Patient) -> None:
        patient.actual_proc_end = SimClasses.Clock
        self.ProcedureRooms.Free(1)
        freed_room_id = patient.room_id
        if freed_room_id is not None:
            self.available_room_ids.append(freed_room_id)
            self.available_room_ids.sort()

        lunch_ot, after_ot, total_ot = compute_nonworking_overlap(
            patient.actual_proc_start,
            patient.actual_proc_end,
        )
        self.total_lunch_overtime += lunch_ot
        self.total_after_hours_overtime += after_ot
        self.total_overtime += total_ot

        if self.WaitingRoomQueue.NumQueue() > 0:
            next_patient = self.WaitingRoomQueue.Remove()
            self.ProcedureRooms.Seize(1)
            room_id = self.available_room_ids.pop(0)
            self._start_procedure(next_patient, room_id, came_from_waiting_room=True)

    def initialize_replication(self) -> None:
        SimFunctions.SimFunctionsInit(self.Calendar)
        self.available_room_ids = list(range(1, self.num_procedure_rooms + 1))
        self.patients = []
        self.slot_assignments = []
        self.patient_counter = 0
        self.total_lunch_overtime = 0.0
        self.total_after_hours_overtime = 0.0
        self.total_overtime = 0.0

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.initialize_replication()
        self.schedule_initial_arrivals()
        self.schedule_slot_open_events()

        while self.Calendar.N() > 0:
            next_event = self.Calendar.Remove()
            SimClasses.Clock = next_event.EventTime

            if next_event.EventType == "PatientArrival":
                self.handle_patient_arrival(next_event.WhichObject)
            elif next_event.EventType == "EndPrep":
                self.handle_end_prep(next_event.WhichObject)
            elif next_event.EventType == "SlotOpen":
                self.handle_slot_open(next_event.WhichObject)
            elif next_event.EventType == "WaitingRoomArrival":
                self.handle_waiting_room_arrival(next_event.WhichObject)
            elif next_event.EventType == "EndProcedure":
                self.handle_end_procedure(next_event.WhichObject)
            else:
                raise ValueError(f"Unknown event type: {next_event.EventType}")

        patients_df = patient_records_to_dataframe(self.patients)
        summary_df = build_summary_dataframe(self, patients_df)
        slot_df = pd.DataFrame(self.slot_assignments)
        return summary_df, patients_df, slot_df


# =========================================================
# KPI HELPERS
# =========================================================
def compute_nonworking_overlap(start_time: float, end_time: float) -> Tuple[float, float, float]:
    lunch_overlap = 0.0
    after_hours_overlap = 0.0

    start_day = int(start_time // MINUTES_PER_DAY)
    end_day = int(end_time // MINUTES_PER_DAY)

    for day in range(start_day, end_day + 1):
        day_offset = day * MINUTES_PER_DAY
        lunch_start = day_offset + 12.0 * 60.0
        lunch_end = day_offset + 13.0 * 60.0
        after_start = day_offset + 17.0 * 60.0
        day_end_time = day_offset + 24.0 * 60.0

        lunch_overlap += max(0.0, min(end_time, lunch_end) - max(start_time, lunch_start))
        after_hours_overlap += max(0.0, min(end_time, day_end_time) - max(start_time, after_start))

    return lunch_overlap, after_hours_overlap, lunch_overlap + after_hours_overlap


def patient_records_to_dataframe(patients: Sequence[Patient]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for p in patients:
        rows.append(
            {
                "patient_id": p.patient_id,
                "category": p.category,
                "order_arrival_time": p.CreateTime,
                "prep_type": p.prep_type,
                "prep_queue_enter": p.prep_queue_enter,
                "prep_start": p.prep_start,
                "prep_end": p.prep_end,
                "scheduling_queue_enter": p.scheduling_queue_enter,
                "scheduled_time": p.scheduled_time,
                "slot_label": p.slot_label,
                "lateness": p.lateness,
                "waiting_room_arrival": p.waiting_room_arrival,
                "actual_proc_start": p.actual_proc_start,
                "actual_proc_end": p.actual_proc_end,
                "room_id": p.room_id,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["wait_to_scan_start"] = df["actual_proc_start"] - df["order_arrival_time"]
    df["prep_wait"] = df["prep_start"] - df["prep_queue_enter"]
    df["prep_service_time"] = df["prep_end"] - df["prep_start"]
    df["scheduling_wait"] = df["scheduled_time"] - df["scheduling_queue_enter"]
    df["waiting_room_wait"] = df["actual_proc_start"] - df["waiting_room_arrival"]
    df["procedure_duration"] = df["actual_proc_end"] - df["actual_proc_start"]
    return df


def build_summary_dataframe(model: IRSchedulingPythonSim, patients_df: pd.DataFrame) -> pd.DataFrame:
    completed_mask = patients_df["actual_proc_end"].notna() if not patients_df.empty else pd.Series(dtype=bool)
    scheduled_mask = patients_df["scheduled_time"].notna() if not patients_df.empty else pd.Series(dtype=bool)

    def safe_mean(series_name: str) -> float:
        if patients_df.empty or series_name not in patients_df:
            return np.nan
        clean = patients_df[series_name].dropna()
        return float(clean.mean()) if clean.shape[0] > 0 else np.nan

    def safe_quantile(series_name: str, q: float) -> float:
        if patients_df.empty or series_name not in patients_df:
            return np.nan
        clean = patients_df[series_name].dropna()
        return float(clean.quantile(q)) if clean.shape[0] > 0 else np.nan

    return pd.DataFrame(
        [
            {
                "num_days": model.num_days,
                "num_patients": 0 if patients_df.empty else int(len(patients_df)),
                "num_scheduled": 0 if patients_df.empty else int(scheduled_mask.sum()),
                "num_completed": 0 if patients_df.empty else int(completed_mask.sum()),
                "num_unscheduled": 0 if patients_df.empty else int((~scheduled_mask).sum()),
                "mean_wait_to_scan_start": safe_mean("wait_to_scan_start"),
                "p90_wait_to_scan_start": safe_quantile("wait_to_scan_start", 0.90),
                "mean_scheduling_wait": safe_mean("scheduling_wait"),
                "mean_waiting_room_wait": safe_mean("waiting_room_wait"),
                "avg_waiting_room_count": model.WaitingRoomQueue.Mean(),
                "max_waiting_room_count": model.WaitingRoomQueue.WIP.Max,
                "avg_interventional_schedule_queue": model.ScheduleQueues["Interventional"].Mean(),
                "avg_angiography_schedule_queue": model.ScheduleQueues["Angiography"].Mean(),
                "avg_room_utilization": (
                    model.ProcedureRooms.Mean() / model.num_procedure_rooms
                    if model.num_procedure_rooms > 0
                    else np.nan
                ),
                "total_lunch_overtime_min": model.total_lunch_overtime,
                "total_after_hours_overtime_min": model.total_after_hours_overtime,
                "total_overtime_min": model.total_overtime,
            }
        ]
    )


# =========================================================
# EXPERIMENT HELPERS
# =========================================================
def run_replications(
    num_replications: int,
    num_days: int,
    arrival_inputs: ArrivalInputs,
    prep_probabilities: Dict[str, Dict[str, float]],
    prep_distributions: Dict[str, Dict[str, DistributionSpec]],
    procedure_distributions: Dict[str, DistributionSpec],
    late_delay_distributions: Dict[str, DistributionSpec],
    slot_rules: Sequence[SlotRule],
    num_procedure_rooms: int,
    prep_capacities: Dict[str, int],
    base_seed: int = 123,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for rep in range(int(num_replications)):
        model = IRSchedulingPythonSim(
            num_days=num_days,
            arrival_inputs=arrival_inputs,
            prep_probabilities=prep_probabilities,
            prep_distributions=prep_distributions,
            procedure_distributions=procedure_distributions,
            late_delay_distributions=late_delay_distributions,
            slot_rules=slot_rules,
            num_procedure_rooms=num_procedure_rooms,
            prep_capacities=prep_capacities,
            seed=base_seed + rep,
        )
        summary_df, _, _ = model.run()
        row = summary_df.iloc[0].to_dict()
        row["replication"] = rep + 1
        rows.append(row)

    return pd.DataFrame(rows)


def policy_objective(
    summary_row: pd.Series,
    w_wait: float = 1.0,
    w_congestion: float = 15.0,
    w_overtime: float = 0.10,
) -> float:
    return (
        float(w_wait) * float(summary_row["mean_wait_to_scan_start"])
        + float(w_congestion) * float(summary_row["avg_waiting_room_count"])
        + float(w_overtime) * float(summary_row["total_overtime_min"])
    )


# =========================================================
# EXAMPLE SLOT RULES
# =========================================================
def example_slot_rules() -> List[SlotRule]:
    return [
        SlotRule(8, ("Interventional",), 1, "08:00 INT"),
        SlotRule(9, ("Interventional", "Angiography"), 2, "09:00 FLEX"),
        SlotRule(10, ("Angiography",), 2, "10:00 ANG"),
        SlotRule(11, ("Angiography",), 1, "11:00 ANG"),
        SlotRule(13, ("Interventional",), 1, "13:00 INT"),
        SlotRule(14, ("Interventional", "Angiography"), 2, "14:00 FLEX"),
        SlotRule(15, ("Angiography",), 2, "15:00 ANG"),
        SlotRule(16, ("Interventional", "Angiography"), 2, "16:00 FLEX"),
    ]


# =========================================================
# PARAMETER SUMMARY
# =========================================================
def summarize_loaded_parameters(
    arrival_inputs: ArrivalInputs,
    prep_probabilities: Dict[str, Dict[str, float]],
    prep_distributions: Dict[str, Dict[str, DistributionSpec]],
    procedure_distributions: Dict[str, DistributionSpec],
    late_delay_distributions: Dict[str, DistributionSpec],
    prep_fallback_groups: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    return {
        "interventional_lambda_hat": arrival_inputs.interventional_lambda_hat.to_dict(),
        "angiography_lambda_hat": arrival_inputs.angiography_lambda_hat.to_dict(),
        "angiography_pln_fit": arrival_inputs.angiography_pln_fit,
        "prep_probabilities": prep_probabilities,
        "prep_distributions": {
            cls: {k: {"dist": v.dist, **v.params} for k, v in groups.items()}
            for cls, groups in prep_distributions.items()
        },
        "procedure_distributions": {
            cls: {"dist": v.dist, **v.params} for cls, v in procedure_distributions.items()
        },
        "late_delay_distributions": {
            cls: {"dist": v.dist, **v.params} for cls, v in late_delay_distributions.items()
        },
        "prep_fallback_groups": list(prep_fallback_groups),
    }


if __name__ == "__main__":
    
    arrival_inputs = load_arrival_inputs_from_json("arrival_model_params.json")
    (
        prep_probabilities,
        prep_distributions,
        procedure_distributions,
        late_delay_distributions,
        prep_fallback_groups,
    ) = load_service_inputs_from_json(
        "services rate.json",
        short_prepare_fallback="category_range_uniform",
    )

    print("Loaded fitted inputs:")
    print(json.dumps(
        summarize_loaded_parameters(
            arrival_inputs,
            prep_probabilities,
            prep_distributions,
            procedure_distributions,
            late_delay_distributions,
            prep_fallback_groups,
        ),
        indent=2,
    ))

    model = IRSchedulingPythonSim(
        num_days=365,
        arrival_inputs=arrival_inputs,
        prep_probabilities=prep_probabilities,
        prep_distributions=prep_distributions,
        procedure_distributions=procedure_distributions,
        late_delay_distributions=late_delay_distributions,
        slot_rules=example_slot_rules(),
        num_procedure_rooms=2,
        prep_capacities={"long_prepare": 1, "medium_prepare": 2, "short_prepare": 2},
        seed=123,
    )

    summary_df, patients_df, slot_df = model.run()
    print(summary_df)
    print(patients_df['category'].value_counts())
    print(slot_df.head())
