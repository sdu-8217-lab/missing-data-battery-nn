#!/usr/bin/env python3
"""
模型参数量验证脚本
使用 PyTorch 官方方式计算参数量，与论文目标值对比

使用方法:
    python scripts/verify_model_params.py
"""

import sys
import torch
import torch.nn as nn

# 添加 src 到路径
sys.path.insert(0, 'src')

from models.mlp import MLP
from models.lstm import LSTM
from models.gru import GRU
from models.cnn1d import CNN1D


def count_parameters(model: nn.Module) -> int:
    """
    使用 PyTorch 官方方式计算可训练参数量
    
    Args:
        model: PyTorch 模型
        
    Returns:
        可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_model(model_class, model_name: str, input_dim: int, 
                 target_params: int, **kwargs) -> dict:
    """
    验证单个模型的参数量
    
    Args:
        model_class: 模型类
        model_name: 模型名称
        input_dim: 输入维度
        target_params: 论文目标参数量
        **kwargs: 模型其他参数
        
    Returns:
        验证结果字典
    """
    try:
        model = model_class(input_dim=input_dim, **kwargs)
        actual_params = count_parameters(model)
        
        return {
            'model': model_name,
            'input_dim': input_dim,
            'target': target_params,
            'actual': actual_params,
            'diff': actual_params - target_params,
            'match': actual_params == target_params,
            'error': None
        }
    except Exception as e:
        return {
            'model': model_name,
            'input_dim': input_dim,
            'target': target_params,
            'actual': None,
            'diff': None,
            'match': False,
            'error': str(e)
        }


def print_results(results: list, title: str):
    """打印验证结果表格"""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    
    print(f"{'Model':<15} {'Input':<8} {'Target':<10} {'Actual':<10} {'Diff':<10} {'Status':<10}")
    print("-" * 70)
    
    for r in results:
        status = "✓ MATCH" if r['match'] else "✗ DIFF" if r['error'] is None else "✗ ERROR"
        actual_str = f"{r['actual']:,}" if r['actual'] is not None else "N/A"
        diff_str = f"{r['diff']:+d}" if r['diff'] is not None else "N/A"
        
        print(f"{r['model']:<15} {r['input_dim']:<8} {r['target']:<10,} {actual_str:<10} {diff_str:<10} {status:<10}")
    
    # 统计
    total = len(results)
    matched = sum(1 for r in results if r['match'])
    print(f"\n匹配统计: {matched}/{total} ({matched/total*100:.1f}%)")


def detailed_analysis(model_class, model_name: str, input_dim: int, **kwargs):
    """详细分析模型各层参数量"""
    print(f"\n{'='*70}")
    print(f"详细分析: {model_name} (input_dim={input_dim})")
    print(f"{'='*70}")
    
    model = model_class(input_dim=input_dim, **kwargs)
    
    print(f"\n{'Layer':<30} {'Shape':<30} {'Parameters':<15}")
    print("-" * 80)
    
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total += num_params
            shape_str = str(tuple(param.shape))
            print(f"{name:<30} {shape_str:<30} {num_params:<15,}")
    
    print("-" * 80)
    print(f"{'Total':<30} {'':<30} {total:<15,}")
    
    return total


def main():
    """主函数：验证所有模型配置"""
    
    print("=" * 70)
    print("模型参数量验证工具 (PyTorch 官方计算方式)")
    print("=" * 70)
    print("\n验证论文 Table 1 (tab:model_config) 中的参数量配置")
    print("Baseline: input_dim=16, MIM: input_dim=32")
    
    results = []
    
    # ========== MLP ==========
    print("\n\n[1/4] 验证 MLP...")
    mlp_baseline = verify_model(
        MLP, "MLP-Baseline", 16, 27649,
        hidden_dims=[192, 96, 48, 24], dropout=0.15
    )
    mlp_mim = verify_model(
        MLP, "MLP-MIM", 32, 36865,
        hidden_dims=[192, 96, 48, 24], dropout=0.15
    )
    results.extend([mlp_baseline, mlp_mim])
    
    # MLP 详细分析
    detailed_analysis(MLP, "MLP", 16, hidden_dims=[192, 96, 48, 24], dropout=0.15)
    detailed_analysis(MLP, "MLP", 32, hidden_dims=[192, 96, 48, 24], dropout=0.15)
    
    # ========== LSTM ==========
    print("\n\n[2/4] 验证 LSTM...")
    lstm_baseline = verify_model(
        LSTM, "LSTM-Baseline", 16, 31537,
        hidden_size=48, num_layers=2, dropout=0.2
    )
    lstm_mim = verify_model(
        LSTM, "LSTM-MIM", 32, 40753,
        hidden_size=48, num_layers=2, dropout=0.2
    )
    results.extend([lstm_baseline, lstm_mim])
    
    # LSTM 详细分析
    detailed_analysis(LSTM, "LSTM", 16, hidden_size=48, num_layers=2, dropout=0.2)
    detailed_analysis(LSTM, "LSTM", 32, hidden_size=48, num_layers=2, dropout=0.2)
    
    # ========== GRU ==========
    print("\n\n[3/4] 验证 GRU...")
    gru_baseline = verify_model(
        GRU, "GRU-Baseline", 16, 40769,
        hidden_size=64, num_layers=2, dropout=0.2
    )
    gru_mim = verify_model(
        GRU, "GRU-MIM", 32, 49985,
        hidden_size=64, num_layers=2, dropout=0.2
    )
    results.extend([gru_baseline, gru_mim])
    
    # GRU 详细分析
    detailed_analysis(GRU, "GRU", 16, hidden_size=64, num_layers=2, dropout=0.2)
    detailed_analysis(GRU, "GRU", 32, hidden_size=64, num_layers=2, dropout=0.2)
    
    # ========== CNN1D ==========
    print("\n\n[4/4] 验证 CNN1D...")
    # CNN1D 需要特殊处理：它期望的输入是序列数据
    # 论文中 channels=[72,32]，输入通道是 input_dim
    cnn_baseline = verify_model(
        CNN1D, "CNN-Baseline", 16, 16713,
        channels=[72, 32], kernel_size=4, dropout=0.1
    )
    cnn_mim = verify_model(
        CNN1D, "CNN-MIM", 32, 30537,
        channels=[72, 32], kernel_size=4, dropout=0.1
    )
    results.extend([cnn_baseline, cnn_mim])
    
    # CNN1D 详细分析
    detailed_analysis(CNN1D, "CNN1D", 16, channels=[72, 32], kernel_size=4, dropout=0.1)
    detailed_analysis(CNN1D, "CNN1D", 32, channels=[72, 32], kernel_size=4, dropout=0.1)
    
    # ========== 总结果汇总 ==========
    print_results(results, "验证结果汇总")
    
    # 返回非零退出码如果有不匹配
    mismatches = [r for r in results if not r['match'] and r['error'] is None]
    if mismatches:
        print(f"\n⚠️  发现 {len(mismatches)} 个不匹配的配置")
        return 1
    else:
        print("\n✓ 所有模型参数量与论文目标值完全匹配！")
        return 0


if __name__ == "__main__":
    exit(main())
