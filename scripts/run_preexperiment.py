#!/usr/bin/env python3
"""
预实验命令行入口

自动搜索给定参数量预算下的最优模型架构

用法:
    # 运行所有模型的所有预算
    python scripts/run_preexperiment.py --all
    
    # 运行特定模型
    python scripts/run_preexperiment.py --model mlp --budgets 8192,16384,32768,65536
    
    # CPU 模式快速测试
    python scripts/run_preexperiment.py --model mlp --budgets 8192 --no-gpu --smoke-test
    
    # 指定 trials 数
    python scripts/run_preexperiment.py --model cnn1d --budgets 65536 --n-trials 50
    
    # 显示详细计时统计
    python scripts/run_preexperiment.py --all --timing
"""
import argparse
import json
import sys
import time
from pathlib import Path
from datetime import timedelta

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preexperiment.runner import run_preexperiment_for_budget
from src.preexperiment.search_space import SearchSpace


# 四个预算等级
ALL_BUDGETS = SearchSpace.BUDGETS  # [8192, 16384, 32768, 65536]
ALL_MODELS = ["mlp", "lstm", "gru", "cnn1d"]


def parse_budgets(budget_str: str) -> list:
    """解析预算字符串"""
    if budget_str.lower() == "all":
        return ALL_BUDGETS
    return [int(b.strip()) for b in budget_str.split(",")]


def parse_models(model_str: str) -> list:
    """解析模型字符串"""
    if model_str.lower() == "all":
        return ALL_MODELS
    return [m.strip().lower() for m in model_str.split(",")]


def run_experiment_with_timing(model_type: str, budget: int, n_trials: int, 
                               output_dir: Path, device: str, smoke_test: bool):
    """运行实验并记录时间"""
    print(f"\n{'='*60}")
    print(f"Experiment: {model_type.upper()} @ {budget:,} params")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        results = run_preexperiment_for_budget(
            model_type=model_type,
            param_budget=budget,
            output_dir=output_dir,
            n_trials=n_trials,
            device=device,
            smoke_test=smoke_test,
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ 完成!")
        print(f"  参数量: {results['n_params']:,}")
        print(f"  Test MAE: {results['performance']['test_mae_mean']:.4f} ± {results['performance']['test_mae_std']:.4f}")
        print(f"  耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
        
        return {
            "model": model_type,
            "budget": budget,
            "success": True,
            "duration": elapsed,
            "n_params": results["n_params"],
            "test_mae": results["performance"]["test_mae_mean"],
            "test_mae_std": results["performance"]["test_mae_std"],
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ 失败!")
        print(f"  错误: {e}")
        print(f"  耗时: {elapsed:.1f}秒")
        
        return {
            "model": model_type,
            "budget": budget,
            "success": False,
            "duration": elapsed,
            "error": str(e),
        }


def print_summary(all_results: list, total_duration: float, output_dir: Path):
    """打印实验汇总"""
    print("\n" + "=" * 70)
    print("实验汇总")
    print("=" * 70)
    print(f"总耗时: {timedelta(seconds=int(total_duration))}")
    
    success_count = sum(1 for r in all_results if r["success"])
    print(f"成功: {success_count}/{len(all_results)}")
    print()
    
    # 打印表格
    print(f"{'Model':<10} {'Budget':<10} {'Status':<8} {'Duration':<10} {'Params':<10} {'Test MAE'}")
    print("-" * 70)
    for r in all_results:
        status = "✓" if r["success"] else "✗"
        duration = f"{r['duration']:.0f}s"
        if r["success"]:
            params = f"{r['n_params']:,}"
            mae = f"{r['test_mae']:.4f}±{r['test_mae_std']:.4f}"
        else:
            params = "-"
            mae = "FAILED"
        print(f"{r['model']:<10} {r['budget']:<10,} {status:<8} {duration:<10} {params:<10} {mae}")
    
    print()
    print("=" * 70)
    print("下一步:")
    print("=" * 70)
    
    if success_count == len(all_results):
        print("✓ 所有实验成功！")
        print(f"  结果保存在: {output_dir}")
        print("  运行分析脚本生成报告:")
        print(f"    python scripts/analyze_preexperiment.py --input-dir {output_dir}")
    else:
        print("✗ 部分实验失败，请检查错误信息")
        failed = [r for r in all_results if not r["success"]]
        for r in failed:
            print(f"  - {r['model']} @ {r['budget']}: {r.get('error', 'Unknown')}")
    
    # 保存汇总
    summary_file = output_dir / "_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "total_duration": total_duration,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\n汇总已保存: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="预实验：搜索最优模型架构",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --all                              # 运行所有模型和预算
  %(prog)s --model mlp --budgets 8192         # 只跑 MLP 8K 预算
  %(prog)s --model mlp,lstm --budgets all     # MLP+LSTM，所有预算
  %(prog)s --model cnn1d --budgets 65536 --no-gpu --smoke-test  # CPU快速测试
  %(prog)s --all --timing                     # 显示详细计时统计
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="运行所有模型 (mlp, lstm, gru, cnn1d) 的所有预算 (8K, 16K, 32K, 65K)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="模型类型，逗号分隔 (mlp, lstm, gru, cnn1d) 或 'all'"
    )
    parser.add_argument(
        "--budgets",
        type=str,
        help="参数量预算，逗号分隔 (8192, 16384, 32768, 65536) 或 'all'"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Optuna 搜索 trials 数 (默认: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/preexperiment",
        help="结果输出目录 (默认: results/preexperiment)"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="强制使用 CPU"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="快速测试模式（减少 trials 和 epochs）"
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="显示详细计时统计和结果汇总"
    )
    
    args = parser.parse_args()
    
    # 确定运行的模型和预算
    if args.all:
        models = ALL_MODELS
        budgets = ALL_BUDGETS
    else:
        if not args.model:
            parser.error("需要指定 --model 或使用 --all")
        if not args.budgets:
            parser.error("需要指定 --budgets 或使用 --all")
        
        models = parse_models(args.model)
        budgets = parse_budgets(args.budgets)
    
    # 验证输入
    for m in models:
        if m not in ALL_MODELS:
            parser.error(f"未知模型: {m}，可用: {ALL_MODELS}")
    
    for b in budgets:
        if b not in ALL_BUDGETS:
            parser.error(f"未知预算: {b}，可用: {ALL_BUDGETS}")
    
    # 确定设备
    import torch
    if args.no_gpu or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"
    
    print("=" * 70)
    print("预实验：自动架构搜索")
    print("=" * 70)
    print(f"模型: {models}")
    print(f"预算: {budgets}")
    print(f"Trials: {args.n_trials}")
    print(f"设备: {device}")
    print(f"输出目录: {args.output_dir}")
    print(f"测试模式: {'是' if args.smoke_test else '否'}")
    print(f"计时统计: {'是' if args.timing else '否'}")
    print("=" * 70)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行实验
    start_time = time.time()
    all_results = []
    
    for model_type in models:
        for budget in budgets:
            result = run_experiment_with_timing(
                model_type=model_type,
                budget=budget,
                n_trials=args.n_trials,
                output_dir=output_dir,
                device=device,
                smoke_test=args.smoke_test,
            )
            all_results.append(result)
    
    total_duration = time.time() - start_time
    
    # 打印汇总
    if args.timing or args.all:
        print_summary(all_results, total_duration, output_dir)
    else:
        print("\n" + "=" * 70)
        print("预实验完成！")
        print(f"结果保存在: {output_dir}")
        print("=" * 70)


if __name__ == "__main__":
    main()
