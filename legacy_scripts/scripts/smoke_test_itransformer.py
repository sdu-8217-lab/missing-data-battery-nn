"""iTransformerModel 烟雾测试"""
import sys
from pathlib import Path

# 将项目根目录加入路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.models.neural_network_models import iTransformerModel
from src.models.base_model import BaseModel


def run_test(device: str):
    """在指定设备上运行前向传播测试"""
    print(f"\nRunning on device: {device}")
    
    batch_size = 4
    seq_len = 5
    input_dim = 16
    
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    
    model = iTransformerModel(input_dim=input_dim, seq_len=seq_len)
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        out = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == torch.Size([batch_size]), (
        f"Expected output shape [{batch_size}], got {out.shape}"
    )
    print("Shape check passed.")
    
    # 同时验证 BaseModel 包装器可用
    wrapped = BaseModel(iTransformerModel(input_dim=input_dim, seq_len=seq_len),
                        device=device, model_type='pytorch')
    with torch.no_grad():
        out_wrapped = wrapped.predict(x)
    print(f"BaseModel output shape: {out_wrapped.shape}")
    assert out_wrapped.shape == torch.Size([batch_size])
    print("BaseModel wrapper check passed.")


def main():
    print("iTransformerModel smoke test")
    
    # CPU 测试
    run_test('cpu')
    
    # CUDA 测试（如果可用）
    if torch.cuda.is_available():
        run_test('cuda')
    else:
        print("\nCUDA not available, skipping GPU test.")
    
    print("\nAll tests passed!")


if __name__ == '__main__':
    main()
