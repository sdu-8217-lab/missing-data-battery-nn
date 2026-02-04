#!/usr/bin/env python3
"""
模型架构搜索实验入口脚本

使用示例:
    # 阶段1: 粗粒度搜索
    python run_architecture_search.py --stage coarse --batch 3C
    
    # 阶段2: 细粒度搜索
    python run_architecture_search.py --stage fine --batch 3C --coarse-results experiments_v2/arch_search_3C_coarse_*/results.csv
    
    # 分析结果
    python run_architecture_search.py --analyze experiments_v2/arch_search_*/results.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='模型架构搜索 - 寻找最佳参数量与架构配置',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行粗粒度搜索（快速筛选）
  python run_architecture_search.py --stage coarse --batch 3C --seed 42
  
  # 运行细粒度搜索（精确验证）
  python run_architecture_search.py --stage fine --batch 3C --coarse-results results.csv
  
  # 分析已有结果
  python run_architecture_search.py --analyze results.csv --budget 50000
        """
    )
    
    parser.add_argument('--stage', choices=['coarse', 'fine'],
                       help='搜索阶段: coarse=粗粒度(10epochs), fine=细粒度(50epochs)')
    parser.add_argument('--batch', default='3C',
                       choices=['2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite'],
                       help='数据批次 (默认: 3C)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='起始架构索引，用于从中断处恢复 (默认: 0)')
    parser.add_argument('--coarse-results',
                       help='粗粒度结果路径（细粒度阶段使用）')
    parser.add_argument('--analyze',
                       help='分析已有结果文件')
    parser.add_argument('--budget', type=int, default=50000,
                       help='参数量预算，用于推荐最佳架构 (默认: 50000)')
    
    args = parser.parse_args()
    
    # 分析模式
    if args.analyze:
        from src.visualization.architecture_analysis import ArchitectureAnalyzer
        analyzer = ArchitectureAnalyzer(args.analyze)
        analyzer.analyze_all()
        analyzer.recommend_best_balance(args.budget)
        return
    
    # 检查必须参数
    if not args.stage:
        parser.error("必须指定 --stage (coarse 或 fine)")
    
    # 导入并运行搜索
    from src.experiments.architecture_search import (
        run_coarse_search, run_fine_search
    )
    
    if args.stage == 'coarse':
        print("=" * 70)
        print("模型架构搜索 - 阶段1: 粗粒度筛选")
        print("=" * 70)
        print(f"批次: {args.batch}")
        print(f"种子: {args.seed}")
        print(f"配置: 10 epochs, 只测缺失率0.5, 自动剪枝")
        print(f"起始索引: {args.start_idx}")
        print("=" * 70)
        print()
        
        df = run_coarse_search(args.batch, args.seed, args.start_idx)
        
        print("\n粗粒度搜索完成!")
        print(f"测试架构数: {len(df) if df is not None else 0}")
        print("\n下一步建议:")
        print("1. 查看分析图表: python run_architecture_search.py --analyze <results_path>")
        print("2. 运行细粒度搜索: python run_architecture_search.py --stage fine --coarse-results <results_path>")
        
    else:  # fine stage
        if not args.coarse_results:
            parser.error("细粒度阶段需要提供 --coarse-results 参数")
        
        print("=" * 70)
        print("模型架构搜索 - 阶段2: 细粒度验证")
        print("=" * 70)
        print(f"批次: {args.batch}")
        print(f"种子: {args.seed}")
        print(f"配置: 50 epochs, 测缺失率0.1/0.5/0.9, 3个种子")
        print("=" * 70)
        print()
        
        df = run_fine_search(args.coarse_results, args.batch, args.seed)
        
        print("\n细粒度搜索完成!")
        print(f"测试架构数: {len(df) if df is not None else 0}")


if __name__ == '__main__':
    main()
