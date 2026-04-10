# Quick Start Guide

Get started with Battery SOH Framework in 5 minutes.

## Installation

```bash
pip install battery-soh
```

Or install from source:

```bash
git clone https://github.com/yourusername/battery-soh-framework.git
cd battery-soh-framework
pip install -e "."
```

## Prerequisites

Download the [XJTU Battery Dataset](https://doi.org/10.1016/j.jpowsour.2021.230639) and place it in `data/XJTU data/`.

Expected structure:
```
data/XJTU data/
├── 2C_battery-1.csv
├── 2C_battery-2.csv
├── 3C_battery-1.csv
└── ...
```

## Basic Example

```python
from battery_soh import Seed
from battery_soh.data import XJTULoader, BatteryWiseSplit
from battery_soh.models import create_model, count_parameters
from battery_soh.training import LightningTrainer, TrainingConfig
from battery_soh.evaluation import Evaluator
from battery_soh.utils import set_seed

# 1. Set random seed for reproducibility
set_seed(Seed(42))

# 2. Load data
loader = XJTULoader()
X, y, battery_ids = loader.load_batch("2C")

# 3. Split data (battery-wise to prevent leakage)
splitter = BatteryWiseSplit(seed=Seed(42))
train_split, val_split, test_split = splitter.split(X, y, battery_ids)

# 4. Create model
model = create_model("mlp", use_mim=False)
print(f"Model parameters: {count_parameters(model):,}")

# 5. Train
config = TrainingConfig(epochs=200, patience=30)
trainer = LightningTrainer(config)
result = trainer.fit(
    model,
    train_data=(train_split.X, train_split.y),
    val_data=(val_split.X, val_split.y)
)
print(f"Best validation loss: {result.best_val_loss:.4f}")

# 6. Evaluate under missing data
evaluator = Evaluator(
    missing_mode="MCAR",
    missing_rate=0.3,
    imputation_method="mean",
    use_mim=False
)
metrics = evaluator.evaluate(
    model=model,
    X=test_split.X,
    y=test_split.y,
    seed=Seed(42)
)
print(f"Test MAE: {metrics.mae:.4f}")
```

## Using MIM

To use Missing Indicator Method (MIM):

```python
# Create model with MIM (32D input instead of 16D)
model = create_model("mlp", use_mim=True)

# ... train model ...

# Evaluate with MIM
evaluator = Evaluator(
    missing_mode="MCAR",
    missing_rate=0.3,
    imputation_method="mean",
    use_mim=True  # Enable MIM
)
```

## Testing Different Missing Modes

```python
modes = ["MCAR", "MAR", "MNAR"]
results = {}

for mode in modes:
    evaluator = Evaluator(
        missing_mode=mode,
        missing_rate=0.3,
        imputation_method="mean",
        use_mim=False
    )
    metrics = evaluator.evaluate(model, X_test, y_test, seed=Seed(42))
    results[mode] = metrics.mae
    print(f"{mode}: MAE = {metrics.mae:.4f}")
```

## Next Steps

- Read the [Architecture Guide](architecture.md) to understand the design
- Check [Tutorials](tutorials/) for more examples
- See [API Reference](api/) for detailed API documentation
