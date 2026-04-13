from __future__ import annotations

"""
Load all fitted inputs used by the IR outpatient simulation.

This file does only one job:
1. Read the arrival JSON.
2. Read the service/preparation/lateness JSON.
3. Convert those JSON values into Python objects that the simulation can use.

The simulation file does not reopen the raw JSON every time it needs a sample.
Instead, this loader is called first, and the simulation receives the already-loaded
objects in memory.
"""

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Unit conversions used when the JSON stores times in hours or days.
MINUTES_PER_HOUR = 60.0
MINUTES_PER_DAY = 24.0 * 60.0


@dataclass
class ArrivalInputs:
    """Arrival-model inputs loaded from the arrival JSON.

    interventional_lambda_hat:
        Hourly NHPP rates for Interventional arrivals.
    angiography_lambda_hat:
        Hourly baseline rates for the Angiography Poisson-lognormal model.
    angiography_pln_fit:
        Poisson-lognormal day-factor parameters for Angiography.
    """

    interventional_lambda_hat: pd.Series
    angiography_lambda_hat: pd.Series
    angiography_pln_fit: Dict[str, float]
    bin_start_hour: int = 8
    bin_end_hour: int = 17


@dataclass
class DistributionSpec:
    """A generic probability distribution specification.

    dist stores the distribution name such as gamma/lognormal/uniform.
    params stores the corresponding fitted parameters.
    """

    dist: str
    params: Dict[str, Any]


@dataclass
class LoadedIRInputs:
    """Everything the simulation needs after loading both JSON files."""

    arrival_inputs: ArrivalInputs
    prep_probabilities: Dict[str, Dict[str, float]]
    prep_distributions: Dict[str, Dict[str, DistributionSpec]]
    procedure_distributions: Dict[str, DistributionSpec]
    late_delay_distributions: Dict[str, DistributionSpec]
    prep_fallback_groups: List[Dict[str, str]]


CLASS_NAMES = ("Interventional", "Angiography")
PREP_NAMES = (
    "long_prepare",
    "medium_prepare",
    "short_prepare",
    "no_prepare_required",
)


def normalize_category(text: str) -> str:
    """Map category names in the JSON into the two standard class labels."""
    text = str(text).strip().lower()
    if text == "interventional":
        return "Interventional"
    if text == "angiography":
        return "Angiography"
    return str(text).strip().title()



def normalize_prep_name(text: str) -> str:
    """Convert preparation labels into one standard snake_case format."""
    text = str(text).strip().lower().replace(" ", "_")
    alias_map = {
        "long_prepare": "long_prepare",
        "medium_prepare": "medium_prepare",
        "short_prepare": "short_prepare",
        "no_prepare_required": "no_prepare_required",
    }
    if text not in alias_map:
        raise ValueError(f"Unsupported preparation name: {text}")
    return alias_map[text]



def _load_short_prepare_empirical_from_raw(raw_data_path: str) -> Dict[str, Any]:
    """Load empirical short-prepare samples once from the raw outpatient file.

    The raw file is opened only once during input loading, not every time a patient
    needs a preparation-time sample. The samples are then kept in memory and the
    simulation draws from that in-memory empirical distribution.
    """
   
    df = pd.read_excel(raw_data_path)

    df = df[["classification", "category_prepared", "Preparation_duration_days"]].dropna().copy()
    df["classification"] = df["classification"].map(normalize_category)
    df["category_prepared"] = df["category_prepared"].map(normalize_prep_name)

    out: Dict[str, Any] = {}
    for cls in CLASS_NAMES:
        samples = (
            df.loc[
                (df["classification"] == cls)
                & (df["category_prepared"] == "short_prepare"),
                "Preparation_duration_days",
            ]
            .astype(float)
            .to_numpy()
            * MINUTES_PER_DAY
        )
        out[cls] = samples
    return out


def load_arrival_inputs_from_json(json_path: str = "arrival_model_params.json") -> ArrivalInputs:
    """Read the arrival JSON and return the fitted arrival inputs.

    Interventional is modeled as NHPP.
    Angiography is modeled as Poisson-lognormal.
    """
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
    """Convert one fitted distribution from the service JSON into DistributionSpec.

    multiplier is used to convert the original unit into minutes when needed.
    For example:
    - hours -> minutes
    - days  -> minutes
    - minutes -> multiplier 1.0
    """
    dist_name = str(payload["distribution"]).strip().lower()
    params = payload["parameters"]

    if dist_name == "gamma":
        loc = float(params.get("loc", 0.0))
        if not math.isclose(loc, 0.0, abs_tol=1e-12):
            raise ValueError("Gamma distributions with nonzero loc are not supported by the simulation.")
        return DistributionSpec(
            "gamma",
            {
                "shape": float(params["shape"]),
                "scale": float(params["scale"]) * multiplier,
            },
        )

    if dist_name == "lognormal":
        loc = float(params.get("loc", 0.0))
        if not math.isclose(loc, 0.0, abs_tol=1e-12):
            raise ValueError("Lognormal distributions with nonzero loc are not supported by the simulation.")
        return DistributionSpec(
            "lognormal",
            {
                # scipy lognormal fit is stored with "scale" = exp(mu).
                # If the original unit is hours or days and we convert to minutes,
                # then the lognormal mean parameter must shift by log(multiplier).
                "mu": math.log(float(params["scale"]) * multiplier),
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
        loc = float(params.get("loc", 0.0))
        if not math.isclose(loc, 0.0, abs_tol=1e-12):
            raise ValueError("Exponential distributions with nonzero loc are not supported by the simulation.")
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
    raw_data_path: Optional[str] = None,
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, DistributionSpec]],
    Dict[str, DistributionSpec],
    Dict[str, DistributionSpec],
    List[Dict[str, str]],
]:
    """Read the service JSON and return the fitted preparation/procedure/lateness inputs.

    Returns
    -------
    prep_probabilities:
        Class-specific probabilities of long/medium/short/no-prep.
    prep_distributions:
        Distribution of preparation duration for each class and prep type.
    procedure_distributions:
        Distribution of actual procedure duration.
    late_delay_distributions:
        Distribution of patient lateness after being booked.
    prep_fallback_groups:
        JSON metadata telling us which groups did not have an accepted parametric fit.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Preparation-type probabilities are stored as percentages in the JSON.
    # Divide by 100 so the simulation can use them as probabilities.
    prep_probabilities_raw = payload["preparation_category_percentages"]
    prep_probabilities: Dict[str, Dict[str, float]] = {}
    for cls, probs in prep_probabilities_raw.items():
        cls_norm = normalize_category(cls)
        prep_probabilities[cls_norm] = {
            normalize_prep_name(k): float(v) / 100.0
            for k, v in probs.items()
        }

    # Preparation-time distributions are stored in days in the JSON,
    # so they are converted into minutes here.
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

    # Keep the fallback metadata for reference.
    prep_fallback_groups: List[Dict[str, str]] = []
    for item in payload.get("preparation_empirical_fallback_groups", []):
        prep_fallback_groups.append(
            {
                "classification": normalize_category(item["classification"]),
                "category_prepared": normalize_prep_name(item["category_prepared"]),
                "reason": str(item["reason"]),
            }
        )

    # If short-prepare should be modeled empirically, read the raw data once here.
    empirical_short_prepare = None
    if short_prepare_fallback == "empirical_from_raw":
        if raw_data_path is None:
            raise ValueError("raw_data_path is required when short_prepare_fallback='empirical_from_raw'.")
        empirical_short_prepare = _load_short_prepare_empirical_from_raw(raw_data_path)

    # The fitting JSON did not provide a parametric distribution for every prep group.
    # This loader fills in the two needed cases so the simulation can still run.
    for cls in CLASS_NAMES:
        prep_distributions.setdefault(cls, {})

        # No-prep means ready immediately.
        prep_distributions[cls]["no_prepare_required"] = DistributionSpec(
            "deterministic",
            {"value": float(no_prepare_value_min)},
        )

        # If short-prepare did not have an accepted parametric fit,
        # either use an empirical resampling distribution from the raw data
        # or a simple uniform fallback.
        if "short_prepare" not in prep_distributions[cls]:
            if short_prepare_fallback == "empirical_from_raw":
                prep_distributions[cls]["short_prepare"] = DistributionSpec(
                    "empirical",
                    {"samples": empirical_short_prepare[cls]},
                )
            elif short_prepare_fallback == "category_range_uniform":
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

    # Procedure duration is currently one common distribution for both classes.
    procedure_all_payload = payload["procedure_duration_hours"]["All"]
    procedure_spec = _spec_from_service_payload(procedure_all_payload, multiplier=MINUTES_PER_HOUR)
    procedure_distributions = {
        "Angiography": procedure_spec,
        "Interventional": procedure_spec,
    }

    # Lateness is already in minutes, so no conversion is needed.
    late_delay_distributions: Dict[str, DistributionSpec] = {}
    for key, fit_payload in payload["late_time_minutes"].items():
        cls = key.split("=", 1)[1]
        late_delay_distributions[normalize_category(cls)] = _spec_from_service_payload(
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



def load_all_ir_inputs(
    arrival_json_path: str = "arrival_model_params.json",
    service_json_path: str = "services rate.json",
    short_prepare_fallback: str = "category_range_uniform",
    no_prepare_value_min: float = 0.0,
    raw_data_path = "df_selected.xlsx"
) -> LoadedIRInputs:
    """Convenience wrapper that loads both JSON files at once."""
    arrival_inputs = load_arrival_inputs_from_json(arrival_json_path)
    (
        prep_probabilities,
        prep_distributions,
        procedure_distributions,
        late_delay_distributions,
        prep_fallback_groups,
    ) = load_service_inputs_from_json(
        service_json_path,
        short_prepare_fallback=short_prepare_fallback,
        no_prepare_value_min=no_prepare_value_min,
        raw_data_path=raw_data_path,
    )

    return LoadedIRInputs(
        arrival_inputs=arrival_inputs,
        prep_probabilities=prep_probabilities,
        prep_distributions=prep_distributions,
        procedure_distributions=procedure_distributions,
        late_delay_distributions=late_delay_distributions,
        prep_fallback_groups=prep_fallback_groups,
    )
