#!/usr/bin/env python3
"""
细粒度架构搜索入口脚本

使用示例:
    # 搜索所有模型的最佳架构
    python run_fine_grained_search.py --model all --batch 3C
    
    # 只搜索LSTM的细粒度配置
    python run_fine_grained_search.py --model lstm --batch 3C
    
    # 从指定索引恢复
    python run_fine_grained_search.py --model gru --start-idx 20
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='细粒度架构搜索 - 在当前最佳配置附近密集探索',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 搜索所有模型（推荐）
  python run_fine_grained_search.py --model all --batch 3C
  
  # 只搜索LSTM
  python run_fine_grained_search.py --model lstm --batch 3C
  
  # 搜索GRU并从中断处恢复
  python run_fine_grained_search.py --model gru --start-idx 25
  
  # 搜索CNN1D
  python run_fine_grained_search.py --model cnn1d --batch 3C
  
  # 搜索XGBoost（粗粒度被剪枝，细粒度再验证）
  python run_fine_grained_search.py --model xgboost --batch 3C
        """
    )
    
    parser.add_argument('--model', choices=['mlp', 'lstm', 'gru', 'cnn1d', 'xgboost', 'all'],
                       default='all',
                       help='要搜索的模型类型 (默认: all)')
    parser.add_argument('--batch', default='3C',
                       choices=['2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite'],
                       help='数据批次 (默认: 3C)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='起始架构索引，用于恢复 (默认: 0)')
    
    args = parser.parse_args()
    
    from src.experiments.fine_grained_search import (
        run_single_model_search, run_all_models_fine_grained
    )
    
    # 估算搜索空间大小
    search_space_sizes = {
        'mlp': '~120个配置',
        'lstm': '~80个配置',
        'gru': '~80个配置',
        'cnn1d': '~150个配置',
        'xgboost': '~150个配置',
        'all': '~580个配置（总耗时约8-10小时）'
    }
    
    print("=" * 70)
    print("细粒度架构搜索")
    print("=" * 70)
    print(f"目标模型: {args.model.upper()}")
    print(f"数据批次: {args.batch}")
    print(f"随机种子: {args.seed}")
    print(f"搜索空间: {search_space_sizes[args.model]}")
    print(f"配置: 50 epochs, 缺失率 [0.1, 0.5, 0.9]")
    print("=" * 70)
    print()
    
    confirm = input("确认开始搜索? (y/n): ")
    if confirm.lower() != 'y':
        print("已取消")
        return
    
    if args.model == 'all':
        results = run_all_models_fine_grained(args.batch, args.seed)
        
        # 汇总最佳结果
        print("\n" + "=" * 70)
        print("各模型最佳架构汇总")
        print("=" * 70)
        for model_type, df in results.items():
            if len(df) > 0:
                best = df.loc[df['mae_mean'].idxmin()]
                print(f"\n【{model_type.upper()}】")
                print(f"  最佳MAE: {best['mae_mean']:.4f}")
                print(f"  参数量: {best['param_count']:,}")
                print(f"  配置: {best['config']}")
    else:
        df = run_single_model_search(args.model, args.batch, args.seed)
        
        if len(df) > 0:
            # 显示Top 5
            print("\n" + "=" * 70)
            print(f"{args.model.upper()} Top 5 架构")
            print("=" * 70)
            top5 = df.nsmallest(5, 'mae_mean')[['config_name', 'mae_mean', 'param_count', 'config']]
            for idx, row in top5.iterrows():
                print(f"\n{row['config_name']}:")
                print(f"  MAE: {row['mae_mean']:.4f}")
                print(f"  参数量: {row['param_count']:,}")
                print(f"  配置: {row['config']}")


if __name__ == '__main__':
    main()
