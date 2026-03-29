#!/usr/bin/env python3
"""
模型评估脚本 - 并行化版本
支持并行评估，最大化CPU利用率
"""
import sys
import argparse
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import pandas as pd

from src.config.experiment_config import ExperimentConfig
from src.data.loader import XJTUDatasetLoader
from src.evaluation.tester import ModelTester
from src.utils.logger import setup_logger
from src.models.factory import create_model


def load_model_info(model_path: Path) -> dict:
    """加载模型信息"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    return {
        'config': checkpoint['config'],
        'model_arch': checkpoint['model_arch'],
        'seed': checkpoint['seed'],
        'use_mim': checkpoint['config']['training']['use_mim'],
        'batch': checkpoint['config']['data']['batch']
    }


def evaluate_single_model(
    model_path: Path,
    data: dict,
    test_config: dict,
    logger=None
) -> pd.DataFrame:
    """评估单个模型 - 支持并行调用"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model_arch = checkpoint['model_arch']
    use_mim = checkpoint['config']['training']['use_mim']
    model_type = model_arch['model_type']
    
    is_sequence_model = model_type in ['lstm', 'gru', 'cnn1d', 'cnn']
    seq_len = checkpoint['config']['training'].get('seq_len', 5)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(
        model_type=model_type,
        input_dim=16,
        use_mim=use_mim,
        device=device,
        **{k: v for k, v in model_arch.items() if k not in ['model_type', 'name', 'level']}
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    tester = ModelTester(
        model, 
        device=device,
        seq_len=seq_len,
        is_sequence_model=is_sequence_model
    )
    
    results = tester.test_all_conditions(
        X_test=data['X_test'],
        y_test=data['y_test'],
        use_mim=use_mim,
        missing_modes=test_config.get('missing_modes', ['MCAR', 'MAR', 'MNAR']),
        missing_rates=test_config.get('missing_rates', [i * 0.05 for i in range(20)]),
        imputation_methods=test_config.get('imputation_methods', ['zero', 'mean', 'knn', 'iterative']),
        base_seed=checkpoint['seed'],
        batch_size=32
    )
    
    results['seed'] = checkpoint['seed']
    results['batch'] = checkpoint['config']['data']['batch']
    results['model_type'] = model_arch['model_type']
    results['use_mim'] = use_mim
    results['model_path'] = str(model_path)
    
    return results


def evaluate_model_parallel(args_tuple):
    """并行评估的工作函数"""
    model_path, test_config, data_dir = args_tuple
    model_path = Path(model_path)
    
    try:
        # 加载模型信息
        first_model_info = load_model_info(model_path)
        batch = first_model_info['batch']
        
        # 加载数据
        data_loader = XJTUDatasetLoader(
            data_dir=data_dir,
            batch=batch
        )
        
        data = data_loader.prepare_data(
            feature_cols=first_model_info['config']['data']['feature_cols'],
            target_col=first_model_info['config']['data']['target_col'],
            test_size=first_model_info['config']['data']['test_size'],
            val_size=first_model_info['config']['data']['val_size'],
            random_seed=first_model_info['seed']
        )
        
        # 评估
        results = evaluate_single_model(model_path, data, test_config)
        
        return {
            'success': True,
            'results': results,
            'model': model_path.name
        }
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'model': model_path.name
        }


def parallel_evaluate_models(
    model_files: list,
    test_config: dict,
    data_dir: str,
    output_dir: Path,
    workers: int = 8
) -> pd.DataFrame:
    """
    并行评估多个模型
    
    Args:
        model_files: 模型文件列表
        test_config: 测试配置
        data_dir: 数据目录
        output_dir: 输出目录
        workers: 并行进程数
    
    Returns:
        合并的评估结果
    """
    all_results = []
    
    print(f"\n并行评估: {len(model_files)} 个模型")
    print(f"并行进程数: {workers}")
    print("="*60)
    
    # 构建任务列表
    tasks = [(str(mf), test_config, data_dir) for mf in model_files]
    
    # 并行评估
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_model_parallel, task): i 
                  for i, task in enumerate(tasks)}
        
        for future in as_completed(futures):
            i = futures[future]
            try:
                result = future.result(timeout=1800)  # 30分钟超时
                if result['success']:
                    all_results.append(result['results'])
                    print(f"[{i+1}/{len(model_files)}] ✓ {result['model']}: {len(result['results'])} 条记录")
                else:
                    print(f"[{i+1}/{len(model_files)}] ✗ {result['model']}: {result['error'][:100]}")
            except Exception as e:
                print(f"[{i+1}/{len(model_files)}] ✗ 异常: {e}")
    
    # 合并结果
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        output_path = output_dir / "test_results.csv"
        final_results.to_csv(output_path, index=False)
        print(f"\n结果已保存: {output_path}")
        return final_results
    else:
        print("\n警告: 没有成功评估的模型")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="评估模型 - 并行化版本")
    parser.add_argument("--models-dir", help="模型文件目录")
    parser.add_argument("--batch-dir", help="批量评估：扫描此目录下所有子目录的models文件夹")
    parser.add_argument("--output-dir", help="输出目录（默认与models-dir相同）")
    parser.add_argument("--data-dir", default="../data/raw/XJTU", help="数据目录")
    parser.add_argument("--missing-modes", nargs="+", default=["MCAR", "MAR", "MNAR"])
    parser.add_argument("--imputation-methods", nargs="+", default=["zero", "mean", "knn", "iterative"])
    parser.add_argument("--missing-rates", type=float, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=8,
                       help="并行评估进程数 (默认: 8)")
    
    args = parser.parse_args()
    
    # 参数检查
    if not args.models_dir and not args.batch_dir:
        print("错误: 必须指定 --models-dir 或 --batch-dir")
        return 1
    
    # 批量评估模式
    if args.batch_dir:
        return evaluate_batch(args)
    
    # 单一模型目录评估
    return evaluate_single_dir(args)


def evaluate_single_dir(args) -> int:
    """评估单一模型目录"""
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir) if args.output_dir else models_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(
        name="eval",
        log_file=str(output_dir / "eval.log"),
        level="INFO"
    )
    
    # 查找所有模型
    model_files = list(models_dir.glob("model_*.pt"))
    if not model_files:
        logger.error(f"未找到模型文件: {models_dir}")
        return 1
    
    logger.info(f"模型数: {len(model_files)}")
    
    # 测试配置
    missing_rates = args.missing_rates if args.missing_rates else [i * 0.05 for i in range(20)]
    test_config = {
        'missing_modes': args.missing_modes,
        'imputation_methods': args.imputation_methods,
        'missing_rates': missing_rates
    }
    
    # 并行评估
    final_results = parallel_evaluate_models(
        model_files=model_files,
        test_config=test_config,
        data_dir=args.data_dir,
        output_dir=output_dir,
        workers=args.workers
    )
    
    if not final_results.empty:
        logger.info(f"总记录数: {len(final_results)}")
        return 0
    return 1


def evaluate_batch(args) -> int:
    """批量评估多个模型目录"""
    batch_dir = Path(args.batch_dir)
    model_dirs = list(batch_dir.rglob("*/models"))
    
    if not model_dirs:
        print(f"未找到模型目录: {batch_dir}")
        return 1
    
    print(f"找到 {len(model_dirs)} 个模型目录")
    
    # 收集所有模型文件
    all_model_files = []
    for model_dir in model_dirs:
        all_model_files.extend(list(model_dir.glob("model_*.pt")))
    
    if not all_model_files:
        print("未找到任何模型文件")
        return 1
    
    print(f"总共 {len(all_model_files)} 个模型")
    
    # 测试配置
    missing_rates = args.missing_rates if args.missing_rates else [i * 0.05 for i in range(20)]
    test_config = {
        'missing_modes': args.missing_modes,
        'imputation_methods': args.imputation_methods,
        'missing_rates': missing_rates
    }
    
    output_dir = batch_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 并行评估所有模型
    final_results = parallel_evaluate_models(
        model_files=all_model_files,
        test_config=test_config,
        data_dir=args.data_dir,
        output_dir=output_dir,
        workers=args.workers
    )
    
    if not final_results.empty:
        print(f"\n评估完成！总记录数: {len(final_results)}")
        return 0
    return 1


if __name__ == "__main__":
    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)
    sys.exit(main())
