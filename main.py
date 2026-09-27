"""
Runs the microgrid dataset through both the Rule-Based Hysteresis baseline
and the PyTorch Linear Regression controller, computes DC-level and
Inverter/AC-level RMSE for each, fills in Format.md, and plots the
per-sample RMSE comparison (Project MD File.md, step 3).
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.generate_dataset import generate
from model.linear_regression_controller import decide as ai_decide
from model.linear_regression_controller import train as train_ai_model
from simulation.hysteresis_controller import decide as hysteresis_decide
from simulation.physics import (
    R_BATTERY,
    R_SOLAR,
    SwitchState,
    buck_converter,
    dc_ac_inverter,
    source_bus_voltage,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
FORMAT_MD_PATH = os.path.join(
    os.path.dirname(ROOT), "ProjectBullshitMKCx2", "Format.md"
)


def run_controller(df: pd.DataFrame, state: np.ndarray, rng: np.random.Generator) -> dict:
    v_bus = source_bus_voltage(df["b_voltage"].values, df["sc_voltage"].values, state)

    industry_target = df[["industry_1", "industry_2", "industry_3"]].mean(axis=1).values
    consumer_target = df[["consumer_1", "consumer_2", "consumer_3"]].mean(axis=1).values

    dc_error = v_bus - industry_target
    dc_rmse = float(np.sqrt(np.mean(dc_error**2)))

    industry_delivered = dc_ac_inverter(v_bus, rng)
    consumer_dc = buck_converter(v_bus, consumer_target)
    consumer_delivered = dc_ac_inverter(consumer_dc, rng)

    industry_cols = df[["industry_1", "industry_2", "industry_3"]].values
    consumer_cols = df[["consumer_1", "consumer_2", "consumer_3"]].values

    industry_ac_error = industry_delivered[:, None] - industry_cols
    consumer_ac_error = consumer_delivered[:, None] - consumer_cols
    ac_errors = np.concatenate([industry_ac_error.ravel(), consumer_ac_error.ravel()])
    ac_rmse = float(np.sqrt(np.mean(ac_errors**2)))

    # Energy contribution: assume 1A nominal draw per block, integrated over
    # a 1-second-per-sample timeline, weighted by each source's Millman share
    # when in TANDEM.
    battery_share = np.where(
        state == SwitchState.BATTERY_ONLY,
        1.0,
        np.where(state == SwitchState.TANDEM, (1.0 / R_BATTERY) / (1.0 / R_BATTERY + 1.0 / R_SOLAR), 0.0),
    )
    solar_share = np.where(
        state == SwitchState.SOLAR_ONLY,
        1.0,
        np.where(state == SwitchState.TANDEM, (1.0 / R_SOLAR) / (1.0 / R_BATTERY + 1.0 / R_SOLAR), 0.0),
    )
    battery_energy_wh = float(np.sum(df["b_voltage"].values * battery_share * 1.0 / 3600.0))
    solar_energy_wh = float(np.sum(df["sc_voltage"].values * solar_share * 1.0 / 3600.0))

    return {
        "dc_rmse": dc_rmse,
        "ac_rmse": ac_rmse,
        "battery_energy_wh": battery_energy_wh,
        "solar_energy_wh": solar_energy_wh,
        "dc_error": dc_error,
    }


def update_format_md(results: dict) -> None:
    ai = results["ai"]
    hyst = results["hysteresis"]

    table = (
        "| Battery Supplied Energy | Solar Grid Energy | DC level RMSE with Rule-Based Hysteresis Switching | "
        "DC level RMSE with AI Switching | Inverter/AC Level RMSE with Rule-Based Hysteresis Switching | "
        "Inverter/AC Level RMSE with AI Switching |\n"
        "| ----------------------- | ----------------- | ---------------------------------------------------- | "
        "-------------------------------- | ----------------------------------------------------------- | "
        "----------------------------------------- |\n"
        f"| {ai['battery_energy_wh']:.4f} Wh | {ai['solar_energy_wh']:.4f} Wh | "
        f"{hyst['dc_rmse']:.4f} V | {ai['dc_rmse']:.4f} V | "
        f"{hyst['ac_rmse']:.4f} V | {ai['ac_rmse']:.4f} V |\n"
    )
    with open(FORMAT_MD_PATH, "w") as f:
        f.write(table)
    print(f"Updated {FORMAT_MD_PATH}")


def plot_comparison(hyst_error: np.ndarray, ai_error: np.ndarray) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    window = 20
    hyst_running_rmse = np.sqrt(
        pd.Series(hyst_error**2).rolling(window, min_periods=1).mean().values
    )
    ai_running_rmse = np.sqrt(
        pd.Series(ai_error**2).rolling(window, min_periods=1).mean().values
    )

    plt.figure(figsize=(10, 5))
    plt.plot(hyst_running_rmse, label="Rule-Based Hysteresis Switching", color="tab:orange")
    plt.plot(ai_running_rmse, label="AI (Linear Regression) Switching", color="tab:blue")
    plt.xlabel("Sample index")
    plt.ylabel(f"Rolling DC-level RMSE (V, window={window})")
    plt.title("DC Bus RMSE: Rule-Based Hysteresis vs. AI Switching")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "rmse_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    df = generate()
    rng = np.random.default_rng(123)

    industry_target = df[["industry_1", "industry_2", "industry_3"]].mean(axis=1).values

    hyst_state = hysteresis_decide(df["b_voltage"].values, df["sc_voltage"].values, industry_target)
    hyst_results = run_controller(df, hyst_state, rng)

    ai_model = train_ai_model()
    ai_state = ai_decide(ai_model, df["b_voltage"].values, df["sc_voltage"].values, industry_target)
    ai_results = run_controller(df, ai_state, rng)

    results = {"hysteresis": hyst_results, "ai": ai_results}

    print("Rule-Based Hysteresis -> DC RMSE: {:.4f} V, AC RMSE: {:.4f} V".format(
        hyst_results["dc_rmse"], hyst_results["ac_rmse"]))
    print("AI (Linear Regression) -> DC RMSE: {:.4f} V, AC RMSE: {:.4f} V".format(
        ai_results["dc_rmse"], ai_results["ac_rmse"]))

    update_format_md(results)
    plot_comparison(hyst_results["dc_error"], ai_results["dc_error"])


if __name__ == "__main__":
    main()
