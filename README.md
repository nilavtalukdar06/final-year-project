# Microgrid Infrastructure Project

Implements the design in `../ProjectBullshitMKCx2/Project MD File.md`, with
two fixes from `OptionalUpgradesForFinalVersion.md` applied:

1. **DC-DC Buck Converter** instead of a transformer on the DC side
   (transformers require AC; a DC transformer just saturates and heats up).
2. **Rule-Based Hysteresis Switching** instead of a "manual switching"
   baseline, since human reaction time (200ms-2s) vs. relay actuation
   (~1-10ms) isn't a fair comparison. Linear Regression is kept as the AI
   controller per the original spec.

## Layout

- `data/generate_dataset.py` — generates `microgrid_data.csv`: B/SC source
  voltages plus 3 consumer (1-5V) and 3 industry (5-9V) load sets per
  Project MD File.md step G.
- `simulation/physics.py` — source combination (Millman's theorem for
  tandem mode), buck regulation, DC-AC inversion.
- `simulation/hysteresis_controller.py` — the rule-based baseline.
- `model/linear_regression_controller.py` — PyTorch linear regression
  model; predicts DC bus voltage per candidate switch state and picks
  whichever state best matches the industry-side demand.
- `main.py` — runs both controllers over the dataset, computes DC-level and
  Inverter/AC-level RMSE for each, writes the results row into
  `../ProjectBullshitMKCx2/Format.md`, and saves a rolling-RMSE comparison
  plot to `results/rmse_comparison.png`.
- `matlab/build_microgrid_model.m` — MATLAB script that programmatically
  builds the corresponding Simulink model (battery/solar blocks, switching
  apparatus + CU, buck converter, dual DC-AC converters, load blocks).
  **Not run or verified here** — this environment has no MATLAB. Open it in
  MATLAB (Simulink + Simscape Electrical) and check port wiring
  interactively; block library paths and port indices may need adjusting
  for your MATLAB version.

## Running the Python pipeline

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

This was run once during setup: 500-sample dataset, AI controller reached
DC RMSE 0.756V vs. 1.279V for the hysteresis baseline (AC-level: 1.114V vs.
1.269V). Exact numbers will vary run-to-run since the dataset and model
init are randomized (see `SEED` in `data/generate_dataset.py` and the
`seed` arg in `model/linear_regression_controller.make_training_data`).

## Not yet done

- The Simulink model is generated as a script but not built/verified in an
  actual MATLAB session.
- No ONNX export or Simulink Deep Learning Toolbox integration wiring the
  trained PyTorch model into the .slx file (OptionalUpgradesForFinalVersion.md
  section 3) — the CU subsystem in the .m script is a placeholder.
