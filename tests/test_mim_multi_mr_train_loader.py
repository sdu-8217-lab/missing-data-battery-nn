"""
Test MIM multi-missing-rate train loader behavior

验证点：
1. 合并后的训练集总样本数是否为 10×N
2. mim_input shape 是否为 (B, 32)
3. 每个缺失率的实际缺失率是否接近目标值
"""
import sys
sys.path.insert(0, '.')

import torch
from omegaconf import OmegaConf
from src.trainers.neural_network_trainer import _create_mim_train_loader
from src.missing_data.mcar import simulate_mcar


def test_mim_train_loader():
    """测试 MIM 训练 DataLoader"""
    print("=" * 70)
    print("Testing MIM Multi-Missing-Rate Train Loader")
    print("=" * 70)
    
    # 1. 创建合成数据集
    N, D = 100, 16
    X_train = torch.randn(N, D)
    y_train = torch.randn(N)
    
    print(f"\n[1] Synthetic dataset created:")
    print(f"    X_train shape: {X_train.shape}")
    print(f"    y_train shape: {y_train.shape}")
    
    # 2. 创建配置
    cfg = OmegaConf.create({
        "missing": {
            "mode": "mar",
            "beta": 2.0,
            "gamma": 0.05,
            "alpha": None
        }
    })
    
    # 3. 创建 DataLoader
    batch_size = 32
    seed = 42
    train_loader = _create_mim_train_loader(cfg, X_train, y_train, batch_size, seed)
    
    # 4. 遍历 DataLoader 收集统计信息
    total_samples = 0
    all_mim_inputs = []
    
    for batch_x, batch_y in train_loader:
        total_samples += len(batch_x)
        all_mim_inputs.append(batch_x)
        
        # 验证 shape
        assert batch_x.shape[1] == 32, f"Expected input dim 32, got {batch_x.shape[1]}"
    
    print(f"\n[2] DataLoader traversal completed:")
    print(f"    Total samples collected: {total_samples}")
    print(f"    Expected: {10 * N} (10 copies × {N})")
    print(f"    Match: {'[OK]' if total_samples == 10 * N else '[FAIL]'}")
    
    # 5. 验证 mim_input shape
    sample_batch = all_mim_inputs[0]
    print(f"\n[3] MIM input shape verification:")
    print(f"    Batch shape: {sample_batch.shape}")
    print(f"    Expected: (B, 32) where 32 = 16 features + 16 masks")
    print(f"    Features part shape: {sample_batch[:, :16].shape}")
    print(f"    Mask part shape: {sample_batch[:, 16:].shape}")
    
    # 6. 验证每个缺失率的实际缺失率
    print(f"\n[4] Missing rate verification per copy:")
    print(f"    {'Target MR':<12} {'Actual MR':<12} {'Error':<12} {'Status'}")
    print(f"    {'-'*12} {'-'*12} {'-'*12} {'-'*6}")
    
    expected_mrs = [i / 10.0 for i in range(10)]
    all_pass = True
    
    for i, target_mr in enumerate(expected_mrs):
        # 重新生成该副本以计算实际缺失率
        mr_seed = seed + i * 100
        X_imp, mask, mim_input = simulate_mcar(X_train, target_mr, mr_seed)
        
        # 计算实际缺失率（mask=0 的比例）
        actual_mr = 1.0 - mask.mean().item()
        error = abs(actual_mr - target_mr)
        status = "[PASS]" if error < 0.02 else "[FAIL]"
        
        if error >= 0.02:
            all_pass = False
            
        print(f"    {target_mr:<12.1f} {actual_mr:<12.3f} {error:<12.3f} {status}")
    
    # 7. 汇总
    print(f"\n[5] Summary:")
    print(f"    Total samples: {total_samples} (expected: {10*N})")
    print(f"    Input dimension: 32 (16 features + 16 masks)")
    print(f"    Missing rate accuracy: {'All within ±0.02' if all_pass else 'Some exceed ±0.02'}")
    
    # 8. 验证 MIM 输入结构
    print(f"\n[6] MIM input structure verification (first sample of copy 0):")
    X_imp_0, mask_0, mim_input_0 = simulate_mcar(X_train, 0.0, seed)
    print(f"    Original X[0]:    {X_train[0][:4].numpy()}... (first 4 dims)")
    print(f"    Mask[0]:          {mask_0[0].numpy()}")
    print(f"    X_imputed[0]:     {X_imp_0[0][:4].numpy()}... (first 4 dims)")
    print(f"    MIM input[0,:4]:  {mim_input_0[0,:4].numpy()}... (should = X_imputed)")
    print(f"    MIM input[0,16:20]: {mim_input_0[0,16:20].numpy()}... (should = 1-mask)")
    
    # 验证拼接正确性
    assert torch.allclose(mim_input_0[0, :16], X_imp_0[0]), "First half should be X_imputed"
    assert torch.allclose(mim_input_0[0, 16:], 1.0 - mask_0[0]), "Second half should be 1-mask"
    print(f"    [OK] Structure verified: [X_imputed | 1-mask]")
    
    print(f"\n{'='*70}")
    if total_samples == 10 * N and all_pass:
        print("ALL TESTS PASSED [OK]")
    else:
        print("SOME TESTS FAILED [FAIL]")
    print(f"{'='*70}")
    
    return total_samples == 10 * N and all_pass


if __name__ == "__main__":
    try:
        success = test_mim_train_loader()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
