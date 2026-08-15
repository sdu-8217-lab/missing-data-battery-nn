"""物理约束损失函数

为电池 SOH 估计提供可微的物理正则项：
- 容量单调性：SOH 随循环数不增
- 退化平滑性：SOH 轨迹二阶差分小
"""
import torch
import torch.nn as nn


class PhysicsLoss(nn.Module):
    """物理约束损失
    
    Args:
        loss_types: 启用的物理损失类型列表，如 ['monotonicity', 'smoothness']
        weights: 各项损失的权重字典，如 {'monotonicity': 0.01, 'smoothness': 0.001}
    """
    
    def __init__(self, loss_types=None, weights=None):
        super().__init__()
        self.loss_types = loss_types or []
        self.weights = weights or {}
    
    def forward(self, predictions, indices):
        """
        Args:
            predictions: [N] SOH 预测值
            indices: [N, 2] 排序信息，[:,0]=block_id, [:,1]=global_index
        Returns:
            dict: 各项物理损失值
        """
        if len(self.loss_types) == 0:
            return {}
        
        # 按 block_id 然后 global_index 排序
        sorted_idx = torch.argsort(indices[:, 0] * 1e9 + indices[:, 1])
        y_sorted = predictions[sorted_idx]
        
        losses = {}
        if 'monotonicity' in self.loss_types:
            # SOH 不应随循环上升：penalize max(0, y[t+1] - y[t])
            diff = y_sorted[1:] - y_sorted[:-1]
            mono_loss = torch.mean(torch.clamp(diff, min=0.0) ** 2)
            w = self.weights.get('monotonicity', 1.0)
            losses['monotonicity'] = w * mono_loss
        
        if 'smoothness' in self.loss_types:
            # 二阶差分平滑
            second_diff = y_sorted[2:] - 2 * y_sorted[1:-1] + y_sorted[:-2]
            smooth_loss = torch.mean(second_diff ** 2)
            w = self.weights.get('smoothness', 1.0)
            losses['smoothness'] = w * smooth_loss
        
        if 'soh_bound' in self.loss_types:
            # SOH 应在 [0, 1] 之间（若已归一化）
            lower_viol = torch.clamp(-y_sorted, min=0.0) ** 2
            upper_viol = torch.clamp(y_sorted - 1.0, min=0.0) ** 2
            bound_loss = torch.mean(lower_viol + upper_viol)
            w = self.weights.get('soh_bound', 1.0)
            losses['soh_bound'] = w * bound_loss
        
        return losses
    
    def total_loss(self, predictions, indices):
        """返回总物理损失（标量 Tensor）"""
        losses = self.forward(predictions, indices)
        if len(losses) == 0:
            return torch.tensor(0.0, device=predictions.device)
        return sum(losses.values())
