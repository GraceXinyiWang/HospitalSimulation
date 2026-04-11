import json
import pandas as pd
from arrival_rate import simulate_poisson_lognormal_days

with open("angiography_pln_params.json", "r") as f:
    params = json.load(f)

lambda_hat = pd.Series(params["lambda_hat"])
pln_fit = params["pln_fit"]

sim_days = simulate_poisson_lognormal_days(
    n_days=100,
    lambda_hat=lambda_hat,
    fit=pln_fit,
    random_state=123,
)