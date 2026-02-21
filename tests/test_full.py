import sys
sys.path.insert(0, 'src')
from experiments.experiment_runner import ExperimentRunner
from config.experiment_config import ExperimentConfig

# 完整测试配置 - 只测试MLP模型，5次重复
config = ExperimentConfig(
    batch='3C',
    n_repeats=5,
    epochs=10,
    early_stopping_patience=5,
    results_dir='./test_results'
)

runner = ExperimentRunner(config, timestamp='20260202_full')

# 只保留MLP相关配置
all_configs = config.get_model_configs()
mlp_configs = [c for c in all_configs if 'MLP' in c.name]
config.get_model_configs = lambda: mlp_configs

print("Running full test with MLP models only")
print("Model configs:", [c.name for c in mlp_configs])
print()

results = runner.run_all_experiments()

print("\n" + "="*60)
print("FULL TEST COMPLETED!")
print("="*60)
print(f"Total results: {len(results)}")
print(f"Results saved to: {runner.exp_dir}")
