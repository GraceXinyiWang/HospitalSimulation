"""Small demo script for the IR outpatient scheduling model.

This script shows the typical workflow:
1. Load JSON inputs.
2. Build one weekly policy.
3. Run one simulation.
4. Run several replications.
"""

from input_loader import load_all_ir_inputs
from Policy_defined import example_policy_R1
from simulation_model import IROutpatientSchedulingSim, qik_to_dataframe, run_replications, summarize_replications

# Example experiment settings.
WARMUP_WEEKS = 20
ONE_RUN_WEEKS = 210
REPLICATION_WEEKS = 156
NUM_REPLICATIONS = 100
MINUTES_PER_DAY = 24.0 * 60.0

PRINT_LABEL_MAP = {
    "mean_prep_duration": "mean_prep_duration_min",
    "mean_booking_wait": "mean_booking_wait_min",
    "mean_lateness": "mean_lateness_min",
    "mean_procedure_duration": "mean_procedure_duration_min",
    "Z1_wait_time": "Z1_avg_patient_wait_time_day",
    "Z2_overtime": "Z2_overtime_hour/week",
}

# Load the two fitted JSON files once at the beginning.
loaded_inputs = load_all_ir_inputs(
    arrival_json_path="arrival_model_params.json",
    service_json_path="services rate.json",
)

# Choose one example weekly policy.
policy = example_policy_R1()

print("Weekly Qik template")
print(qik_to_dataframe(policy.qik).head(12))
print(f"\nMeasured weeks per one-run example: {ONE_RUN_WEEKS}")
print(f"Warmup weeks before measurement: {WARMUP_WEEKS}")

# Run one simulation instance.
model = IROutpatientSchedulingSim(
    num_weeks=ONE_RUN_WEEKS,
    warmup_weeks=WARMUP_WEEKS,
    loaded_inputs=loaded_inputs,
    policy=policy,
    seed=123,
)
summary_df, patients_df, bookings_df = model.run()
summary_print_df = summary_df.rename(columns=PRINT_LABEL_MAP)
summary_print_df["Z1_avg_patient_wait_time_day"] = (
    summary_df["Z1_wait_time"] / MINUTES_PER_DAY
)

print("\nOne-run summary")
print(summary_print_df.T)
print("\nBookings")
print(bookings_df.head())
print("\nPatients")
print(patients_df['category'].value_counts())

# Run multiple independent replications using different seeds.
rep_df = run_replications(
    num_replications=NUM_REPLICATIONS,
    num_weeks=REPLICATION_WEEKS,
    loaded_inputs=loaded_inputs,
    policy=policy,
    base_seed=123,
    warmup_weeks=WARMUP_WEEKS,
)
rep_print_df = rep_df.rename(columns=PRINT_LABEL_MAP)
rep_print_df["Z1_avg_patient_wait_time_day"] = rep_df["Z1_wait_time"] / MINUTES_PER_DAY
stats_print_df = summarize_replications(rep_df).copy()
z1_mask = stats_print_df["metric"] == "Z1_wait_time"
stats_print_df.loc[z1_mask, ["mean", "std", "min", "max"]] = (
    stats_print_df.loc[z1_mask, ["mean", "std", "min", "max"]] / MINUTES_PER_DAY
)
stats_print_df["metric"] = stats_print_df["metric"].replace(PRINT_LABEL_MAP)

print("\nReplication summary")
print(rep_print_df[["replication", "Z1_avg_patient_wait_time_day", "Z2_overtime_hour/week", "Z3_congestion", "H"]])
print("\nAcross-replication statistics")
print(stats_print_df)
