# Architecture Documentation

## Overview

The Battery SOH Framework follows a clean, layered architecture designed for long-term maintainability and extensibility.

## Design Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Explicit Over Implicit**: All behavior is explicit and documented
3. **Type Safety**: Full type annotations throughout
4. **Testability**: All components are unit testable
5. **Composability**: Small components that can be combined

## Layer Structure

### Core Layer (`battery_soh/core/`)

**Purpose**: Type definitions, constants, and interfaces

```python
# types.py - Type aliases for clarity
Seed = NewType("Seed", int)
Features = NDArray[np.float32]
```

**Key Files**:
- `types.py`: Type aliases (Seed, Features, Mask, etc.)
- `constants.py`: All constants in one place
- `interfaces.py`: Protocol definitions

### Data Layer (`battery_soh/data/`)

**Purpose**: Data loading and preprocessing

```
data/
├── loader.py       # XJTULoader
├── transforms.py   # Feature extraction
└── splits.py       # Battery-wise splitting
```

**Key Design**: Battery-wise splitting prevents data leakage

```python
splitter = BatteryWiseSplit(seed=Seed(42))
train, val, test = splitter.split(X, y, battery_ids)
# Same battery never appears in two splits
```

### Model Layer (`battery_soh/models/`)

**Purpose**: Neural network architectures

```
models/
├── architectures/  # MLP, LSTM, CNN implementations
└── factory.py      # create_model() function
```

**Parameter Budget**: All models respect 16,384-32,768 parameter constraint

### Missing Data Layer (`battery_soh/missing/`)

**Purpose**: Missing pattern generation and imputation

```
missing/
├── generators/     # MCAR, MAR, MNAR
└── imputers/       # mean, knn, iterative, zero
```

**Design Pattern**: Strategy pattern for easy extension

### Training Layer (`battery_soh/training/`)

**Purpose**: Model training with PyTorch Lightning

- Early stopping
- Learning rate scheduling
- Reproducible training

### Evaluation Layer (`battery_soh/evaluation/`)

**Purpose**: Model evaluation under missing data

```python
evaluator = Evaluator(
    missing_mode="MCAR",
    missing_rate=0.3,
    imputation_method="mean",
    use_mim=False
)
metrics = evaluator.evaluate(model, X, y, seed)
```

## 9-Level Architecture Implementation

The framework implements the 9-level experimental architecture from meta.md:

### Above the Divide (Training)

Levels 1-6 define what needs independent training:

```python
# Each combination trains a separate model
for seed in range(100):
    for batch in ["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"]:
        for model_type in ["mlp", "lstm", "cnn"]:
            for use_mim in [False, True]:
                train_model(seed, batch, model_type, use_mim)
```

### Below the Divide (Testing)

Levels 7-9 are applied to trained models:

```python
# Same model tested with all combinations
for mode in ["MCAR", "MAR", "MNAR"]:
    for missing_rate in np.arange(0, 1.0, 0.05):
        for imputation in ["mean", "knn", "iterative", "zero"]:
            evaluate(model, mode, missing_rate, imputation)
```

## Extension Points

### Adding a New Model Architecture

1. Create file in `models/architectures/`
2. Implement `nn.Module` with `forward()` method
3. Add to `factory.py`
4. Add tests

### Adding a New Missing Generator

1. Implement `MissingGenerator` protocol
2. Add to `missing/generators/`
3. Register in `Evaluator`

### Adding a New Imputer

1. Inherit from `BaseImputer`
2. Implement `fit()` and `transform()`
3. Add tests

## Module Dependencies

```
core (no dependencies)
  ↑
data → core
  ↑
models → core
  ↑
missing → core
  ↑
training → core, models
  ↑
evaluation → core, missing, models
  ↑
utils → core
```

Dependencies only flow upward (no circular dependencies).

## Testing Strategy

- **Unit Tests**: Each module independently tested
- **Integration Tests**: End-to-end workflows
- **Type Checking**: mypy strict mode
- **Coverage Target**: >90%

## Performance Considerations

- **Vectorization**: NumPy operations over loops
- **Batch Processing**: Efficient DataLoader usage
- **Memory Management**: Streaming for large datasets
- **GPU Support**: Automatic device detection

## Future Extensions

Planned architectural improvements:

1. **Plugin System**: Dynamic loading of models/imputers
2. **Configuration Management**: YAML/JSON experiment configs
3. **Distributed Training**: Multi-GPU support
4. **Model Registry**: Versioned model management
