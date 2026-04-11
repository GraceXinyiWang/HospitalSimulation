import json
from pathlib import Path

import pandas as pd
from arrival_rate import simulate_poisson_lognormal_days

BASE_DIR = Path(__file__).resolve().parent
ARRIVAL_JSON_PATH = BASE_DIR / "arrival_model_params.json"

with open(ARRIVAL_JSON_PATH, "r", encoding="utf-8") as f:
    params = json.load(f)

pln_params = params["angiography_pln"]
lambda_hat = pd.Series(pln_params["lambda_hat"])
pln_fit = pln_params["pln_fit"]

sim_days = simulate_poisson_lognormal_days(
    n_days=100,
    lambda_hat=lambda_hat,
    fit=pln_fit,
    random_state=123,
)
