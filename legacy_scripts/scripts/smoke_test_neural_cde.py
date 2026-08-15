"""NeuralCDEModel 快速冒烟测试"""
import sys
import torch

# 直接加载模型文件，避免触发包级相对导入
sys.path.insert(0, '/home/chen/research/missing-data-battery-nn/src/models')

from neural_cde_model import NeuralCDEModel


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    batch_size = 4
    seq_len = 5
    input_dim = 16

    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    model = NeuralCDEModel(input_dim=input_dim).to(device)

    model.eval()
    with torch.no_grad():
        out = model(x)

    print(f'Input shape:  {x.shape}')
    print(f'Output shape: {out.shape}')
    assert out.shape == torch.Size([batch_size]), f'Expected [4], got {out.shape}'
    print('Smoke test passed.')


if __name__ == '__main__':
    main()
