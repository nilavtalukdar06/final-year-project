"""
PyTorch Linear Regression controller (Project MD File.md, step 2 and 3;
kept as Linear Regression per project decision, with the unfair "vs. manual
switching" comparison replaced by the hysteresis baseline instead).

The model does not classify a switching state directly. It learns, per
candidate SwitchState, the DC bus voltage that state would produce given
(v_battery, v_solar). At inference we run all three candidates through the
model and pick whichever predicted bus voltage lands closest to the
industry-side target voltage -- i.e. a one-step lookahead the same
threshold controllers can't do, since the linear layer approximates the
Millman-combination nonlinearity per state.
"""
import numpy as np
import torch
from torch import nn

from simulation.physics import SwitchState, source_bus_voltage

N_STATES = 3


class BusVoltageRegressor(nn.Module):
    """One linear layer per candidate switch state: (v_battery, v_solar) -> predicted bus voltage."""

    def __init__(self):
        super().__init__()
        self.state_heads = nn.Linear(2, N_STATES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.state_heads(x)


def make_training_data(n_samples: int = 4000, seed: int = 7):
    rng = np.random.default_rng(seed)
    v_battery = rng.uniform(6.0, 9.0, n_samples).astype(np.float32)
    v_solar = rng.uniform(2.0, 9.0, n_samples).astype(np.float32)

    targets = np.stack(
        [
            source_bus_voltage(v_battery, v_solar, np.full(n_samples, s))
            for s in (SwitchState.BATTERY_ONLY, SwitchState.SOLAR_ONLY, SwitchState.TANDEM)
        ],
        axis=1,
    ).astype(np.float32)

    x = np.stack([v_battery, v_solar], axis=1)
    return torch.from_numpy(x), torch.from_numpy(targets)


def train(epochs: int = 300, lr: float = 0.05) -> BusVoltageRegressor:
    model = BusVoltageRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x, y = make_training_data()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def decide(model: BusVoltageRegressor, v_battery: np.ndarray, v_solar: np.ndarray, target: np.ndarray) -> np.ndarray:
    v_battery = np.asarray(v_battery, dtype=np.float32)
    v_solar = np.asarray(v_solar, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    x = torch.from_numpy(np.stack([v_battery, v_solar], axis=1))
    predicted = model(x).numpy()  # shape (n, 3)

    error = np.abs(predicted - target[:, None])
    return np.argmin(error, axis=1)
