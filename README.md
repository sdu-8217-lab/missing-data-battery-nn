# Battery SOH Prediction Framework

[![CI](https://github.com/yourusername/battery-soh-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/battery-soh-framework/actions)
[![Coverage](https://codecov.io/gh/yourusername/battery-soh-framework/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/battery-soh-framework)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A long-term open-source framework for battery State of Health (SOH) prediction under missing data scenarios.

## Features

- **9-Level Experimental Architecture**: Rigorous experimental design separating training and testing variables
- **Missing Data Support**: MCAR, MAR, MNAR missing patterns with 4 imputation methods
- **MIM (Missing Indicator Method)**: Compare models with/without missingness indicators
- **Battery-wise Splitting**: Prevents data leakage by splitting at battery level
- **Parameter Budget**: All models constrained to 16K-32K parameters for fair comparison
- **Type Safe**: Full type annotations with mypy strict checking
- **Well Tested**: >90% test coverage

## Quick Start

```bash
pip install battery-soh
```

```python
from battery_soh import Seed
from battery_soh.data import XJTULoader
from battery_soh.models import create_model
from battery_soh.training import LightningTrainer, TrainingConfig

# Load data
loader = XJTULoader()
X, y, battery_ids = loader.load_batch("2C")

# Create and train model
model = create_model("mlp", use_mim=False)
trainer = LightningTrainer(TrainingConfig(epochs=200))
result = trainer.fit(model, (X_train, y_train), (X_val, y_val))
```

See [Quick Start Guide](docs/quickstart.md) for complete example.

## Architecture

The framework follows a clean, layered architecture:

```
battery_soh/
├── core/        # Types, constants, interfaces
├── data/        # Data loading and preprocessing
├── models/      # Neural network architectures (MLP, LSTM, CNN)
├── missing/     # Missing generators (MCAR/MAR/MNAR) and imputers
├── training/    # PyTorch Lightning trainer
├── evaluation/  # Metrics and evaluation pipeline

```

### Key Design: 9-Level Architecture

**Above the Divide (Training)**:
- L1: Seed (reproducibility)
- L2: Dataset (XJTU)
- L3: Batch (2C, 3C, R2.5, R3, RW, Sim_satellite)
- L4: Model (MLP, LSTM, CNN)
- L5: use_mim (True/False)
- L6: Training missing rates

**Below the Divide (Testing)**:
- L7: Missing mode (MCAR, MAR, MNAR)
- L8: Test missing rate
- L9: Imputation method

This design allows training 14,400 models that can be tested across 432,000 combinations efficiently.

## Installation

### From PyPI

```bash
pip install battery-soh
```

### From Source

```bash
git clone https://github.com/yourusername/battery-soh-framework.git
cd battery-soh-framework
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- See [pyproject.toml](pyproject.toml) for full dependencies

## Usage

### Basic Training and Evaluation

```python
from battery_soh.data import XJTULoader, BatteryWiseSplit
from battery_soh.models import create_model
from battery_soh.training import LightningTrainer, TrainingConfig
from battery_soh.evaluation import Evaluator

# Load and split data
loader = XJTULoader()
X, y, battery_ids = loader.load_batch("2C")
splitter = BatteryWiseSplit(seed=Seed(42))
train, val, test = splitter.split(X, y, battery_ids)

# Train model
model = create_model("mlp", use_mim=False)
trainer = LightningTrainer(TrainingConfig(epochs=200))
trainer.fit(model, (train.X, train.y), (val.X, val.y))

# Evaluate with missing data
evaluator = Evaluator(
    missing_mode="MCAR",
    missing_rate=0.3,
    imputation_method="mean",
    use_mim=False
)
metrics = evaluator.evaluate(model, test.X, test.y, seed=Seed(42))
print(f"MAE: {metrics.mae:.4f}")
```

### Using MIM

```python
# Create model with MIM (32D input)
model = create_model("mlp", use_mim=True)

# Evaluate with MIM
evaluator = Evaluator(
    missing_mode="MCAR",
    missing_rate=0.3,
    imputation_method="mean",
    use_mim=True  # Enable MIM
)
```

## Documentation

- [Quick Start](docs/quickstart.md) - Get started in 5 minutes
- [Architecture](docs/architecture.md) - Detailed architecture documentation
- [Datasets](docs/DATASETS.md) - Dataset information
- [Model Specs](docs/architecture.md) - Model specifications
- [All Documentation](docs/index.md) - Complete documentation index

## Development

### Setup

```bash
pip install -e ".[dev]"
```

### Testing

```bash
pytest tests/ -v --cov=battery_soh
```

### Type Checking

```bash
mypy battery_soh/ --strict
```

### Linting

```bash
ruff check battery_soh/
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{battery_soh_framework,
  title={Battery SOH Prediction Framework},
  author={Battery SOH Research Team},
  year={2024},
  url={https://github.com/yourusername/battery-soh-framework}
}
```

## Acknowledgments

- XJTU Battery Dataset: https://doi.org/10.1016/j.jpowsour.2021.230639
- PyTorch Lightning: https://lightning.ai
