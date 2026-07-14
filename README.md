# Sensor Calibration Model

A PyTorch model that learns to calibrate a temperature sensor from raw resistance readings, using a Steinhart-Hart-style logarithmic equation:

```
temperature = a / (log(resistance) + b) + c
```

Instead of fixed physical constants, `a`, `b`, and `c` are treated as learnable parameters and fitted to noisy synthetic lab data via gradient descent.

## How it works

1. **Synthetic data generation** — raw resistance values (10–500 Ω) are mapped to a "true" temperature curve, then Gaussian noise is added to simulate real sensor jitter.
2. **Model** — `CalibrationModel` is a custom `nn.Module` with three learnable scalar parameters (`a`, `b`, `c`) plugged into the calibration equation.
3. **Training** — Adam optimizer minimizes MSE loss over 5000 epochs on a 70/30 train/test split.
4. **Evaluation** — the trained model is saved, reloaded, and scored on held-out test data (MSE and RMSE).
5. **Visualization** — a predicted-vs-actual scatter plot is saved to `plots/calibration_results.png`.

## Project structure

```
sensor-calibration/
├── sensor_calibration.py     # main training/evaluation script
├── models/
│   └── calibration_model.pth # pretrained model checkpoint
├── plots/                    # output plots (generated on run)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python sensor_calibration.py
```

This retrains the model from scratch, saves the checkpoint to `models/calibration_model.pth`, evaluates it on the test set, and writes a results plot to `plots/calibration_results.png`.

## Loading the pretrained model

```python
import torch
from sensor_calibration import CalibrationModel

torch.serialization.add_safe_globals([CalibrationModel])
model = torch.load('models/calibration_model.pth')
model.eval()
```

## Notes

- Ground-truth parameters used to generate the synthetic data: `a=500.0, b=2.5, c=50.0`.
- Random seed is fixed (`torch.manual_seed(42)`) for reproducibility.
