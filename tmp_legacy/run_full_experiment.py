"""运行完整实验 - 所有模型，指定重复次数"""
import sys
sys.path.insert(0, 'src')
import argparse
from experiments.experiment_runner import ExperimentRunner
from config.experiment_config import ExperimentConfig

def main():
    parser = argparse.ArgumentParser(description='运行完整SOH实验')
    parser.add_argument('--batch', type=str, default='3C',
                       choices=['2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite'])
    parser.add_argument('--n_repeats', type=int, default=10,
                       help='重复实验次数（完整实验建议100）')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--results_dir', type=str, default='./experiments')
    args = parser.parse_args()
    
    # 创建配置
    config = ExperimentConfig(
        batch=args.batch,
        n_repeats=args.n_repeats,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        results_dir=args.results_dir
    )
    
    print("=" * 70)
    print("SOH预测缺失数据处理实验")
    print("=" * 70)
    print(f"批次: {args.batch}")
    print(f"重复次数: {args.n_repeats}")
    print(f"训练轮数: {args.epochs}")
    print(f"早停耐心: {args.patience}")
    print(f"模型数量: 10 (5模型 × 2策略)")
    print(f"总实验次数: {args.n_repeats * 10}次训练 + {args.n_repeats * 10 * 9}次评估")
    print("=" * 70)
    print()
    
    # 运行实验
    runner = ExperimentRunner(config)
    results = runner.run_all_experiments()
    
    print()
    print("=" * 70)
    print("实验完成！")
    print("=" * 70)
    print(f"结果保存位置: {runner.exp_dir}")
    print(f"总结果数: {len(results)}")
    
    # 简单统计
    import pandas as pd
    df = pd.DataFrame(results)
    print("\n各模型平均MAE:")
    summary = df.groupby('model')['mae'].mean().sort_values()
    for model, mae in summary.items():
        print(f"  {model:15s}: {mae:.6f}")
    
    print("\nMIM vs Baseline对比:")
    for base_model in ['MLP', 'XGBoost', 'LSTM', 'GRU', 'CNN1D']:
        base_mae = df[df['model'] == base_model]['mae'].mean()
        mim_mae = df[df['model'] == f"{base_model}-MIM"]['mae'].mean()
        if not pd.isna(base_mae) and not pd.isna(mim_mae):
            improvement = (base_mae - mim_mae) / base_mae * 100
            print(f"  {base_model:10s}: Baseline={base_mae:.6f}, MIM={mim_mae:.6f}, 提升={improvement:+.1f}%")

if __name__ == '__main__':
    main()
