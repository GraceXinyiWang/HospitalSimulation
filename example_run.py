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
WARMUP_WEEKS = 52
ONE_RUN_WEEKS = 520
REPLICATION_WEEKS = 156
NUM_REPLICATIONS = 100

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

print("\nOne-run summary")
print(summary_df.T)
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

print("\nReplication summary")
print(rep_df[["replication", "Z1_wait_time", "Z2_overtime", "Z3_congestion", "H"]])
print("\nAcross-replication statistics")
print(summarize_replications(rep_df))
