# Battery SOH Framework Documentation

Welcome to the Battery State of Health (SOH) Prediction Framework documentation.

## Overview

This is a long-term open-source framework for battery SOH prediction under missing data scenarios. It implements the 9-Level Experimental Architecture for systematic comparison of Missing Indicator Method (MIM) vs. traditional imputation approaches.

## Quick Start

```python
from battery_soh import Seed
from battery_soh.data import XJTULoader
from battery_soh.models import create_model

# Load data
loader = XJTULoader()
X, y, battery_ids = loader.load_batch("2C")

# Create model
model = create_model("mlp", use_mim=False)

# Train and evaluate...
```

See [Quick Start Guide](./quickstart.md) for complete example.

## Architecture

The framework follows a clean layered architecture:

```
battery_soh/
├── core/        # Types, constants, interfaces
├── data/        # Data loading and preprocessing
├── models/      # Neural network architectures
├── missing/     # Missing data generators and imputers
├── training/    # Model training
├── evaluation/  # Model evaluation
└── utils/       # Utilities
```

## Key Concepts

### 9-Level Experimental Architecture

The framework implements a rigorous experimental design:

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

### MIM vs Imputation

**Key Insight**: MIM and imputation are orthogonal, not opposing:

| | No MIM (16D) | With MIM (32D) |
|---|---|---|
| Mean | Mean only | Mean + mask |
| KNN | KNN only | KNN + mask |

The research question: Does adding MIM indicators help, given the same imputation?

## Installation

```bash
pip install battery-soh
```

For development:

```bash
git clone https://github.com/yourusername/battery-soh-framework.git
cd battery-soh-framework
pip install -e ".[dev]"
```

## Documentation Structure

- [Quick Start](./quickstart.md) - Get started in 5 minutes
- [Architecture](./architecture.md) - Detailed architecture documentation
- [Datasets](./DATASETS.md) - Dataset information
- [Architecture](./architecture.md) - Model specifications
- [Root README](../README.md) - Complete documentation index

## Contributing

We welcome contributions! See [Contributing Guide](../CONTRIBUTING.md) for details.

## License

MIT License - see LICENSE file for details.

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
