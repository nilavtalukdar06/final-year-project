"""
Physical models for the microgrid: source combination, DC-DC buck
regulation, and DC-AC inversion. Voltage sources are modeled with a small
internal resistance so parallel ("tandem") combination follows Millman's
theorem rather than a naive average.
"""
from enum import IntEnum

import numpy as np

# Internal resistance of each 6-unit block (ohms). Battery banks are stiffer
# (lower Zint) than a solar array, which is why tandem combination pulls the
# bus voltage closer to V_B than to V_SC.
R_BATTERY = 0.5
R_SOLAR = 1.2

BUCK_TARGET_MIN = 3.0
BUCK_TARGET_MAX = 7.0
BUCK_EFFICIENCY = 0.94  # voltage droop under load, not power efficiency

INVERTER_EFFICIENCY = 0.96
INVERTER_RIPPLE_STD = 0.05  # volts, AC-side ripple/noise on top of the mean


class SwitchState(IntEnum):
    BATTERY_ONLY = 0
    SOLAR_ONLY = 1
    TANDEM = 2


def source_bus_voltage(v_battery: np.ndarray, v_solar: np.ndarray, state: np.ndarray) -> np.ndarray:
    """DC bus voltage produced by the switching apparatus for each state."""
    v_battery = np.asarray(v_battery, dtype=float)
    v_solar = np.asarray(v_solar, dtype=float)
    state = np.asarray(state)

    v_tandem = (v_battery / R_BATTERY + v_solar / R_SOLAR) / (1.0 / R_BATTERY + 1.0 / R_SOLAR)

    out = np.where(
        state == SwitchState.BATTERY_ONLY,
        v_battery,
        np.where(state == SwitchState.SOLAR_ONLY, v_solar, v_tandem),
    )
    return out


def buck_converter(v_bus: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Regulate v_bus down toward `target` (clamped to the 3-7V band). A buck
    converter cannot step voltage UP, so if v_bus < target the best it can
    do is pass v_bus through (minus a small droop).
    """
    v_bus = np.asarray(v_bus, dtype=float)
    target = np.clip(np.asarray(target, dtype=float), BUCK_TARGET_MIN, BUCK_TARGET_MAX)
    regulated = np.minimum(v_bus, target) * BUCK_EFFICIENCY
    return regulated


def dc_ac_inverter(v_dc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Convert a DC level to its AC-equivalent RMS delivered voltage."""
    v_dc = np.asarray(v_dc, dtype=float)
    ripple = rng.normal(0.0, INVERTER_RIPPLE_STD, size=v_dc.shape)
    return v_dc * INVERTER_EFFICIENCY + ripple
