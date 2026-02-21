import sys
sys.path.insert(0, 'src')
from experiments.experiment_runner import ExperimentRunner
from config.experiment_config import ExperimentConfig

config = ExperimentConfig(
    batch='3C',
    n_repeats=1,
    epochs=2,
    early_stopping_patience=5,
    results_dir='./test_results'
)

runner = ExperimentRunner(config, timestamp='20260202_test')
runner.load_data()
print('Data loaded successfully')
print("Train shape:", runner.data['X_train'].shape)

# Test training one model
model_configs = config.get_model_configs()
print("Total model configs:", len(model_configs))

# Train only MLP baseline
mlp_config = [c for c in model_configs if c.name == 'MLP'][0]
print('\nTesting:', mlp_config.name)

model, history = runner.train_baseline_model(mlp_config, seed=42)
print('Training completed!')
print("History keys:", list(history.keys()))

# Test evaluation
print('\nEvaluating at missing rate 0.5...')
metrics, preds, targets = runner.evaluate_model(model, mlp_config, missing_rate=0.5, seed=42)
print('MAE:', metrics['mae'])
print('RMSE:', metrics['rmse'])
print('R2:', metrics['r2'])

print('\nQuick test PASSED!')
