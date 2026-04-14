# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Discrete-event simulation model for **Interventional Radiology (IR) outpatient scheduling** with optimization. The model simulates patient arrivals, preparation, booking into weekly appointment blocks, waiting-room queuing, and single-room procedures. The goal is to find optimal weekly appointment capacity allocations (Qik matrices) that minimize a weighted objective of patient wait time, staff overtime, and waiting-room congestion.

## Running the Simulation

```bash
# Basic demo: load inputs, run one simulation, run 100 replications
python example_run.py

# Optimization scripts (long-running)
python Optimization_SAA.py
python Optimization_Lin_Stage2.py
python "Optimization_Subset_Selection+KN_simplified.py"
python subset_selection.py          # subset-selection stage only

# Warm-up / DOE analysis
python Design_of_Experiment_Analysis.py
```

No build step, test framework, or package manager config exists. Dependencies: numpy, pandas, scipy, matplotlib, tqdm. Some scripts also use openpyxl (for xlsx reads).

## Architecture

### Core Simulation Pipeline

1. **input_loader.py** — Loads two JSON files (`arrival_model_params.json`, `services rate.json`) into `LoadedIRInputs`. Converts fitted distribution parameters (gamma, lognormal, weibull, etc.) into `DistributionSpec` objects. Handles unit conversions (days/hours to minutes). Also reads `df_selected.xlsx` for empirical short-prepare fallback.

2. **simulation_model.py** — The main simulation engine (`IROutpatientSchedulingSim`). Uses a heap-based event calendar with four event types: `PatientArrival` → `ReadyToSchedule` → `WaitingRoomArrival` → `EndProcedure`. Arrival generation uses NHPP thinning (Interventional) and Poisson-lognormal (Angiography). Outputs three objectives:
   - **Z1**: average patient wait time (order arrival to procedure start)
   - **Z2**: average weekly overtime (hours)
   - **Z3**: max waiting-room queue length
   - **H**: weighted combination (0.6·Z1/28days + 0.2·Z2/2.5hrs + 0.2·Z3/2)

3. **Policy_defined.py** — Defines scheduling policies. A policy = `BookingTimetable` (feasibility mask, shape 2×40) + `Qik` (capacity allocation, shape 2×40). Two timetables defined: R1 and R2. Supports two Qik modes: daily-repeated (2×8 tiled to 2×40) for brute-force, and full-week (2×40) for other optimizers. Includes brute-force candidate generators.

### Optimization Scripts

- **Optimization_Subset_Selection+KN_simplified.py** — Brute-force enumeration with subset selection + Kim-Nelson (KN) ranking-and-selection using common random numbers (CRN).
- **Optimization_SAA.py** — Sample Average Approximation via LP relaxation (scipy linprog), then rounds and validates in simulation.
- **Optimization_Lin_Stage2.py** — Neighborhood search adapted from Lin et al. (2017). Probabilistically moves slots between blocks based on simulation feedback.

### Legacy / Utility Modules

- **SimClasses.py**, **SimFunctions.py**, **SimRNG.py** — Generic discrete-event simulation framework (Clock, CTStat, DTStat, Entity, EventCalendar, FIFOQueue, Resource). These are **not used** by the main IR simulation (which has its own heap-based event system in simulation_model.py). They appear to be educational building blocks.

## Key Domain Concepts

- **Qik matrix** (2×40): rows = [Interventional, Angiography], columns = 40 weekly blocks (5 weekdays × 8 hourly blocks per day, with lunch break 12-13 skipped). Entry = number of appointment slots allocated.
- **Timetable feasibility mask** (2×40): binary mask of which blocks each class is allowed to book into. Applied on top of Qik.
- **Warmup weeks**: simulation runs warmup period before measurement to reach steady state. Typical: 20 weeks warmup, 156-210 weeks measured.
- Time unit throughout the simulation is **minutes**. JSON inputs store durations in days or hours; `input_loader.py` converts to minutes.
