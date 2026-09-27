"""
Generates the CSV of source voltages and changing consumer/industry load
sets described in Project MD File.md, step G.

Columns:
    b_voltage, sc_voltage,
    consumer_1, consumer_2, consumer_3   (1-5 V demand each)
    industry_1, industry_2, industry_3   (5-9 V demand each)
"""
import numpy as np
import pandas as pd

N_SAMPLES = 500
SEED = 42

B_VOLTAGE_RANGE = (6.0, 9.0)   # battery bank sags as it discharges
SC_VOLTAGE_RANGE = (2.0, 9.0)  # solar bank varies with irradiance
CONSUMER_RANGE = (1.0, 5.0)
INDUSTRY_RANGE = (5.0, 9.0)


def generate(n_samples: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    b_voltage = rng.uniform(*B_VOLTAGE_RANGE, n_samples)
    sc_voltage = rng.uniform(*SC_VOLTAGE_RANGE, n_samples)

    consumers = rng.uniform(*CONSUMER_RANGE, size=(n_samples, 3))
    industries = rng.uniform(*INDUSTRY_RANGE, size=(n_samples, 3))

    df = pd.DataFrame(
        {
            "b_voltage": b_voltage,
            "sc_voltage": sc_voltage,
            "consumer_1": consumers[:, 0],
            "consumer_2": consumers[:, 1],
            "consumer_3": consumers[:, 2],
            "industry_1": industries[:, 0],
            "industry_2": industries[:, 1],
            "industry_3": industries[:, 2],
        }
    )
    return df


if __name__ == "__main__":
    out_path = __file__.replace("generate_dataset.py", "microgrid_data.csv")
    generate().to_csv(out_path, index=False)
    print(f"Wrote dataset to {out_path}")
