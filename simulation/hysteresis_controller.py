"""
Deterministic hysteresis/state-machine baseline (OptionalUpgradesForFinalVersion.md,
section 4: Rule-Based Threshold Switching), used in place of a human
"manual switching" baseline since electronic relay actuation on both sides
of the comparison keeps the benchmark fair.

Decision rule, evaluated once per timestep from the industry-side target
(the load that sits directly on the DC bus, so it is the tightest
constraint on switching state):
    - If Battery alone covers the target within `margin`, use BATTERY_ONLY.
    - Else if Solar alone covers the target within `margin`, use SOLAR_ONLY.
    - Else use TANDEM.
A small hysteresis band avoids chattering between states when voltages are
near a threshold.
"""
import numpy as np

from simulation.physics import SwitchState

MARGIN = 0.75  # volts of tolerated undershoot before a source is "insufficient"
HYSTERESIS_BAND = 0.2


def decide(v_battery: np.ndarray, v_solar: np.ndarray, target: np.ndarray) -> np.ndarray:
    v_battery = np.asarray(v_battery, dtype=float)
    v_solar = np.asarray(v_solar, dtype=float)
    target = np.asarray(target, dtype=float)

    battery_sufficient = v_battery >= (target - MARGIN - HYSTERESIS_BAND)
    solar_sufficient = v_solar >= (target - MARGIN - HYSTERESIS_BAND)

    state = np.full(target.shape, SwitchState.TANDEM, dtype=int)
    state = np.where(battery_sufficient, SwitchState.BATTERY_ONLY, state)
    state = np.where(~battery_sufficient & solar_sufficient, SwitchState.SOLAR_ONLY, state)
    return state
