#!/usr/bin/env python3
"""
模型评估脚本
负责分界线以下的测试阶段（L7-L9）：3种缺失模式 × 20种MR × 4种插补 = 240组合

用法:
    # 串行评估（默认）
    python evaluate_models.py --models-dir results/phase1/3C/20260327/models
    
    # 并行评估（8进程，推荐用于多模型）
    python evaluate_models.py --models-dir results/phase1/3C/20260327/models --workers 8
"""
import sys
import argparse
import multiprocessing as mp
from pathlib import Path
from glob import glob

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
    """
    评估单个模型
    
    执行240种组合的完整测试
    """
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model_arch = checkpoint['model_arch']
    use_mim = checkpoint['config']['training']['use_mim']
    model_type = model_arch['model_type']
    
    # 判断是否为序列模型
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
    
    # 创建测试器（传递序列模型参数）
    tester = ModelTester(
        model, 
        device=device,
        seq_len=seq_len,
        is_sequence_model=is_sequence_model
    )
    
    # 执行完整测试
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
    
    # 添加模型信息
    results['seed'] = checkpoint['seed']
    results['batch'] = checkpoint['config']['data']['batch']
    results['model_type'] = model_arch['model_type']
    results['use_mim'] = use_mim
    results['param_count'] = checkpoint['param_count']
    
    return results


def evaluate_single_model_parallel(args_tuple):
    """
    并行评估单个模型的包装函数（用于多进程）
    
    每个进程独立加载数据和模型，避免共享内存问题
    """
    model_path, test_config, data_dir = args_tuple
    model_path = Path(model_path)
    
    try:
        # 加载模型
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        model_arch = checkpoint['model_arch']
        use_mim = checkpoint['config']['training']['use_mim']
        model_type = model_arch['model_type']
        seq_len = checkpoint['config']['training'].get('seq_len', 5)
        is_sequence_model = model_type in ['lstm', 'gru', 'cnn1d', 'cnn']
        batch = checkpoint['config']['data']['batch']
        
        # 独立加载数据
        data_loader = XJTUDatasetLoader(
            data_dir=data_dir,
            batch=batch
        )
        
        data = data_loader.prepare_data(
            feature_cols=checkpoint['config']['data']['feature_cols'],
            target_col=checkpoint['config']['data']['target_col'],
            test_size=checkpoint['config']['data']['test_size'],
            val_size=checkpoint['config']['data']['val_size'],
            random_seed=checkpoint['seed']
        )
        
        # 创建模型（CPU模式，避免GPU竞争）
        model = create_model(
            model_type=model_type,
            input_dim=16,
            use_mim=use_mim,
            device='cpu',
            **{k: v for k, v in model_arch.items() if k not in ['model_type', 'name', 'level']}
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 创建测试器
        tester = ModelTester(
            model, 
            device='cpu',
            seq_len=seq_len,
            is_sequence_model=is_sequence_model
        )
        
        # 执行测试
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
        
        # 添加模型信息
        results['seed'] = checkpoint['seed']
        results['batch'] = batch
        results['model_type'] = model_arch['model_type']
        results['use_mim'] = use_mim
        results['param_count'] = checkpoint['param_count']
        
        return {'success': True, 'results': results, 'model': model_path.name}
        
    except Exception as e:
        import traceback
        return {'success': False, 'error': str(e), 'traceback': traceback.format_exc(), 'model': model_path.name}


def main():
    parser = argparse.ArgumentParser(description="评估模型")
    parser.add_argument("--models-dir", required=True, help="模型文件目录")
    parser.add_argument("--output-dir", help="输出目录（默认与models-dir相同）")
    parser.add_argument("--data-dir", default="../data/raw/XJTU", help="数据目录")
    parser.add_argument("--missing-modes", nargs="+", default=["MCAR", "MAR", "MNAR"])
    parser.add_argument("--imputation-methods", nargs="+", default=["zero", "mean", "knn", "iterative"])
    parser.add_argument("--missing-rates", type=float, nargs="+", default=None,
                       help="缺失率列表，默认使用0.05步长的20个值")
    parser.add_argument("--workers", type=int, default=1,
                       help="并行进程数 (默认: 1串行, >1启用并行)")
    
    args = parser.parse_args()
    
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
    
    logger.info("="*60)
    logger.info("模型测试阶段 (分界线以下)")
    logger.info("="*60)
    logger.info(f"模型数: {len(model_files)}")
    n_mrs = len(args.missing_rates) if args.missing_rates else 20
    logger.info(f"测试组合: {len(args.missing_modes)} modes × {n_mrs} MRs × {len(args.imputation_methods)} imputations = {len(args.missing_modes) * n_mrs * len(args.imputation_methods)}")
    logger.info("="*60)
    
    # 加载第一个模型以获取配置
    first_model_info = load_model_info(model_files[0])
    batch = first_model_info['batch']
    
    # 加载测试数据
    logger.info(f"加载测试数据 - 批次: {batch}")
    data_loader = XJTUDatasetLoader(
        data_dir=args.data_dir,
        batch=batch
    )
    
    # 使用第一个模型的种子加载数据
    data = data_loader.prepare_data(
        feature_cols=first_model_info['config']['data']['feature_cols'],
        target_col=first_model_info['config']['data']['target_col'],
        test_size=first_model_info['config']['data']['test_size'],
        val_size=first_model_info['config']['data']['val_size'],
        random_seed=first_model_info['seed']
    )
    
    # 评估所有模型
    missing_rates = args.missing_rates if args.missing_rates else [i * 0.05 for i in range(20)]
    test_config = {
        'missing_modes': args.missing_modes,
        'imputation_methods': args.imputation_methods,
        'missing_rates': missing_rates
    }
    
    all_results = []
    
    if args.workers > 1:
        # 并行评估模式
        logger.info(f"启用并行评估: {args.workers} 进程")
        
        # 构建任务列表
        tasks = [(str(mf), test_config, args.data_dir) for mf in model_files]
        
        with mp.Pool(processes=args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(evaluate_single_model_parallel, tasks), 1):
                if result['success']:
                    all_results.append(result['results'])
                    logger.info(f"[{i}/{len(model_files)}] ✓ {result['model']}: {len(result['results'])} 条记录")
                else:
                    logger.error(f"[{i}/{len(model_files)}] ✗ {result['model']}: {result['error'][:100]}")
    else:
        # 串行评估模式（默认）
        for i, model_path in enumerate(model_files, 1):
            logger.info(f"[{i}/{len(model_files)}] 评估: {model_path.name}")
            
            try:
                results = evaluate_single_model(model_path, data, test_config, logger)
                all_results.append(results)
                logger.info(f"  ✓ 完成: {len(results)} 条记录")
            except Exception as e:
                logger.error(f"  ✗ 失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # 合并并保存结果
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        output_path = output_dir / "test_results.csv"
        final_results.to_csv(output_path, index=False)
        
        logger.info("="*60)
        logger.info(f"测试阶段完成")
        logger.info(f"总记录数: {len(final_results)}")
        logger.info(f"结果已保存: {output_path}")
        logger.info("="*60)
        
        # 打印摘要
        summary = final_results.groupby(['model_type', 'use_mim', 'missing_mode', 'imputation_method']).agg({
            'mae': ['mean', 'std']
        }).reset_index()
        
        logger.info("\n性能摘要 (MAE):")
        for _, row in summary.head(10).iterrows():
            logger.info(f"  {row['model_type']}-MIM{row['use_mim']}, "
                       f"{row['missing_mode']}/{row['imputation_method']}: "
                       f"{row['mae']['mean']:.4f}±{row['mae']['std']:.4f}")
    else:
        logger.error("没有成功完成任何评估")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
