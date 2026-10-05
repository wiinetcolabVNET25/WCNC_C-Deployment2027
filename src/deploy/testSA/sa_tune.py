#!/usr/bin/env python3

import argparse
import itertools
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS = {
    "cell": {
        "trace": "c_and_g_trace",
        "clusters": "vc_from_c",

        "N_required": [2.0, 1.0, 3.0],
        "Tn_required": [90.0, 120.0, 60.0],
    },

    "graph": {
        "trace": "c_and_g_trace",
        "clusters": "vc_from_g",

        "N_required": [1.0, 2.0, 3.0],
        "Tn_required": [120.0, 90.0, 60.0],
    },

    "vehicle": {
        "trace": "v_trace_reduced",
        "clusters": "vc_from_v",

        "N_required": [3.0, 1.0, 2.0],
        "Tn_required": [60.0, 120.0, 90.0],
    },
}

# ---------------------------------------------------------------------------
# Filename
# ---------------------------------------------------------------------------

def history_filename(
    trace_name, cluster_name,
    number_of_rsus,
    N_required, Tn_required,
    iterations, initial_temperature, cooling_rate, alpha,
    seed,
):
    return (
        f"{trace_name}_{cluster_name}_his_"
        f"{number_of_rsus}_"
        f"{N_required[0]:g}_{N_required[1]:g}_{N_required[2]:g}_"
        f"{Tn_required[0]:g}_{Tn_required[1]:g}_{Tn_required[2]:g}_"
        f"{iterations}_"
        f"{initial_temperature:.3f}_"
        f"{cooling_rate:.3f}_"
        f"{alpha:.2f}_"
        f"{seed}.csv"
    )

# ---------------------------------------------------------------------------
# Run one SA experiment
# ---------------------------------------------------------------------------

def run_sa(
    executable,
    data_dir, results_dir,
    trace_name, cluster_name,
    number_of_rsus,
    N_required, Tn_required,
    iterations, initial_temperature, cooling_rate, alpha,
    seed,
):
    cmd = [
        str(executable),

        str(data_dir), str(results_dir),

        trace_name, cluster_name,

        str(number_of_rsus),

        str(N_required[0]), str(N_required[1]), str(N_required[2]),

        str(Tn_required[0]), str(Tn_required[1]), str(Tn_required[2]),

        str(iterations), str(initial_temperature), str(cooling_rate), str(alpha),

        str(seed),
    ]

    return subprocess.run(cmd, check=True, text=True)

# ---------------------------------------------------------------------------
# Read final result
# ---------------------------------------------------------------------------

def read_final_result(history_file):
    df = pd.read_csv(history_file, sep=";")

    if df.empty:
        raise RuntimeError(f"Empty history file: {history_file}")

    final = df.iloc[-1]

    return {
        "best_objective": float(final["best_objective"]),
        "best_c0_s": float(final["best_c0_s"]),
        "best_c1_s": float(final["best_c1_s"]),
        "best_c2_s": float(final["best_c2_s"]),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("sa_tune_res"))
    parser.add_argument("--rsus", type=int, required=True)
    parser.add_argument("--scenarios", nargs="+",
        choices=SCENARIOS.keys(), default=list(SCENARIOS.keys()))

    parser.add_argument("--initial-temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seeds", nargs="+", type=int, default=range(1, 31))

    parser.add_argument("--cleanup", action="store_true",
        help="Delete per-scenario generated files after tuning, keeping only tuning results.",)

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    executable = Path(args.executable)

    if not executable.is_absolute():
        executable = script_dir / executable

    executable = executable.resolve()

    if not executable.is_file():
        raise FileNotFoundError(
            f"SA executable not found: {executable}"
        )

    if not executable.stat().st_mode & 0o111:
        raise PermissionError(
            f"SA executable is not executable: {executable}"
        )

    # -----------------------------------------------------------------------
    # Parameter grid
    # -----------------------------------------------------------------------

    # iterations_values = [ 1000, 2000 ]
    # cooling_values = [ 0.950, 0.980 ]
    # iterations_values = [ 1000, 2000, 3000, 4000, 5000, 10000, 15000 ]
    # cooling_values = [ 0.950, 0.980, 0.990, 0.995, 0.999, ]

    iterations_values = [ 10000 ]
    cooling_values = [ 0.99 ]

    args.results_dir.mkdir(parents=True, exist_ok=True)
    rsu_results_dir = (args.results_dir / f"rsu_{args.rsus:04d}")
    rsu_results_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # -----------------------------------------------------------------------
    # Experiments
    # -----------------------------------------------------------------------

    for scenario_name in args.scenarios:

        scenario = SCENARIOS[scenario_name]
        
        trace_name = scenario["trace"]
        cluster_name = scenario["clusters"]

        N_required = scenario["N_required"]
        Tn_required = scenario["Tn_required"]

        scenario_results_dir = (rsu_results_dir / scenario_name)
        scenario_results_dir.mkdir(parents=True, exist_ok=True)

        for iterations, cooling_rate in itertools.product(iterations_values, cooling_values):
            for seed in args.seeds:

                try:

                    run_sa(
                        executable=executable,
                        data_dir=args.data_dir,
                        results_dir=scenario_results_dir,
                        trace_name=trace_name,
                        cluster_name=cluster_name,
                        number_of_rsus=args.rsus,
                        N_required=N_required,
                        Tn_required=Tn_required,
                        iterations=iterations,
                        initial_temperature=args.initial_temperature,
                        cooling_rate=cooling_rate,
                        alpha=args.alpha,
                        seed=seed,
                    )

                    history_file = (
                        scenario_results_dir
                        / history_filename(
                            trace_name=trace_name,
                            cluster_name=cluster_name,
                            number_of_rsus=args.rsus,
                            N_required=N_required,
                            Tn_required=Tn_required,
                            iterations=iterations,
                            initial_temperature=args.initial_temperature,
                            cooling_rate=cooling_rate,
                            alpha=args.alpha,
                            seed=seed,
                        )
                    )

                    if not history_file.exists():
                        raise RuntimeError(f"Missing history file: {history_file}")

                    metrics = read_final_result(history_file)

                    results.append({
                        "scenario": scenario_name,
                        "N": tuple(N_required),
                        "Tn": tuple(Tn_required),
                        "rsus": args.rsus,
                        "seed": seed,
                        "best_objective": metrics["best_objective"],
                        "best_c0_s": metrics["best_c0_s"],
                        "best_c1_s": metrics["best_c1_s"],
                        "best_c2_s": metrics["best_c2_s"],
                    })

                except (
                    subprocess.CalledProcessError,
                    RuntimeError,
                    ValueError,
                    OSError,
                ) as e:

                    # Keep errors in the generated dataset without stopping
                    # the entire tuning campaign.
                    results.append({
                        "scenario": scenario_name,
                        "N": tuple(N_required),
                        "Tn": tuple(Tn_required),
                        "rsus": args.rsus,
                        "seed": seed,
                        "best_objective": np.nan,
                        "best_c0_s": np.nan,
                        "best_c1_s": np.nan,
                        "best_c2_s": np.nan,
                        "error": str(e),
                    })

    # -----------------------------------------------------------------------
    # Save raw results
    # -----------------------------------------------------------------------

    raw_df = pd.DataFrame(results)
    raw_df.to_csv(rsu_results_dir / "tuning_raw.csv", sep=";", index=False,)

    # -----------------------------------------------------------------------
    # Aggregate over seeds
    # -----------------------------------------------------------------------

    summary_df = (
        raw_df
        .groupby(
            [
                "scenario",
            ],
            dropna=False,
        )
        .agg(
            best_objective_mean=("best_objective", "mean"),
            best_objective_std=("best_objective", "std"),
            successful_runs=("best_objective", "count"),
        )
        .reset_index()
    )

    summary_df.to_csv(rsu_results_dir / "tuning_summary.csv", sep=";", index=False,)

    if args.cleanup:
        for scenario_name, _ in SCENARIOS.items():
            
            scenario_results_dir = (rsu_results_dir / scenario_name)
            if scenario_results_dir.is_dir():
                shutil.rmtree(scenario_results_dir)

if __name__ == "__main__":
    main()