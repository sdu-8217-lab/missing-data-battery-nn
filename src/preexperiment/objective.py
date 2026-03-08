"""
Optuna 目标函数定义

包含：
1. 参数量计算和约束检查
2. 模型创建工厂
3. Optuna objective 函数
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import optuna


def count_parameters(model: nn.Module) -> int:
    """计算模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_param_penalty(model: nn.Module, param_budget: int) -> float:
    """
    计算参数量超预算的惩罚项
    
    Args:
        model: 模型实例
        param_budget: 参数预算上限
        
    Returns:
        惩罚值（0 表示满足约束，越大表示超预算越多）
    """
    n_params = count_parameters(model)
    if n_params <= param_budget:
        return 0.0
    else:
        # 软性惩罚：超预算比例 × 惩罚系数
        excess_ratio = (n_params - param_budget) / param_budget
        return excess_ratio * 10.0  # 10倍惩罚系数


def create_model(
    model_type: str,
    input_dim: int,
    config: Dict[str, Any],
    device: str = "cpu"
) -> nn.Module:
    """
    根据配置创建模型
    
    Args:
        model_type: 模型类型 (mlp, lstm, gru, cnn1d)
        input_dim: 输入维度
        config: 模型配置参数字典
        device: 计算设备
        
    Returns:
        创建的模型实例
    """
    model_type = model_type.lower()
    
    if model_type == "mlp":
        from src.models.mlp import MLP
        model = MLP(
            input_dim=input_dim,
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"]
        )
    
    elif model_type == "lstm":
        from src.models.lstm import LSTM
        model = LSTM(
            input_dim=input_dim,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"]
        )
    
    elif model_type == "gru":
        from src.models.gru import GRU
        model = GRU(
            input_dim=input_dim,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"]
        )
    
    elif model_type == "cnn1d":
        from src.models.cnn1d import CNN1D
        # CNN1D 的原始接口需要适配
        # 我们的搜索空间产生的是简化版参数，需要转换
        model = CNN1D(
            input_dim=input_dim,
            channels=config["channels"],
            kernel_size=config["kernel_size"],
            dropout=config["dropout"]
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def _is_sequence_model(model: nn.Module) -> bool:
    """检查模型是否为序列模型（需要3D输入）"""
    model_name = model.__class__.__name__.lower()
    return any(x in model_name for x in ["lstm", "gru", "cnn1d"])


def _add_sequence_dim(x: torch.Tensor, seq_len: int = 5) -> torch.Tensor:
    """
    为2D输入添加序列维度
    
    输入: [batch, features]
    输出: [batch, seq_len, features]（通过重复扩展）
    """
    # 重复特征 seq_len 次作为简单序列
    # 实际应用中应该使用真实的滑动窗口序列
    return x.unsqueeze(1).repeat(1, seq_len, 1)


def train_and_evaluate(
    model: nn.Module,
    data: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    device: str = "cpu",
    epochs: int = 50,
    patience: int = 10,
    seq_len: int = 5,
) -> Dict[str, float]:
    """
    训练并评估模型
    
    Args:
        model: 模型实例
        data: 包含 X_train, y_train, X_val, y_val, X_test, y_test 的字典
        config: 包含 lr, batch_size 等的训练配置
        device: 计算设备
        epochs: 最大训练轮数
        patience: 早停耐心值
        seq_len: 序列长度（用于序列模型）
        
    Returns:
        包含 test_mae, test_rmse, test_r2, val_mae_best 的字典
    """
    model = model.to(device)
    
    # 训练配置
    lr = config.get("lr", 1e-3)
    batch_size = config.get("batch_size", 64)
    weight_decay = config.get("weight_decay", 0.0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # 数据准备
    X_train = data["X_train"].to(device)
    y_train = data["y_train"].to(device)
    X_val = data["X_val"].to(device)
    y_val = data["y_val"].to(device)
    X_test = data["X_test"].to(device)
    y_test = data["y_test"].to(device)
    
    # 序列模型需要3D输入 [batch, seq_len, features]
    if _is_sequence_model(model):
        X_train = _add_sequence_dim(X_train, seq_len)
        X_val = _add_sequence_dim(X_val, seq_len)
        X_test = _add_sequence_dim(X_test, seq_len)
    
    # DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    # 训练循环
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        
        train_loss /= len(train_dataset)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).squeeze()
            val_loss = criterion(val_outputs, y_val).item()
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # 加载最佳模型并测试
    if best_state is not None:
        model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test).squeeze()
        
        # MAE
        test_mae = torch.mean(torch.abs(test_outputs - y_test)).item()
        
        # RMSE
        test_mse = torch.mean((test_outputs - y_test) ** 2).item()
        test_rmse = test_mse ** 0.5
        
        # R2
        y_mean = y_test.mean()
        ss_tot = torch.sum((y_test - y_mean) ** 2).item()
        ss_res = torch.sum((y_test - test_outputs) ** 2).item()
        test_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # 验证集MAE（用于 Optuna 优化目标）
        val_outputs = model(X_val).squeeze()
        val_mae = torch.mean(torch.abs(val_outputs - y_val)).item()
    
    return {
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "val_mae_best": val_mae,
        "best_epoch": epoch - patience_counter,
    }


class ObjectiveFactory:
    """
    Optuna 目标函数工厂
    
    创建闭包函数作为 optuna 的 objective，捕获所有必要的上下文
    """
    
    def __init__(
        self,
        model_type: str,
        input_dim: int,
        param_budget: int,
        data: Dict[str, torch.Tensor],
        search_space_func,
        training_config: Dict[str, Any],
        device: str = "cpu",
        trial_queue: Optional[list] = None,  # 用于记录所有 trial 结果
    ):
        self.model_type = model_type
        self.input_dim = input_dim
        self.param_budget = param_budget
        self.data = data
        self.search_space_func = search_space_func
        self.training_config = training_config
        self.device = device
        self.trial_queue = trial_queue
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Optuna 目标函数
        
        返回值越小越好（验证MAE + 参数量惩罚）
        """
        # 1. 采样模型配置
        model_config = self.search_space_func(
            trial=trial,
            input_dim=self.input_dim,
            param_budget=self.param_budget
        )
        
        # 2. 创建模型
        try:
            model = create_model(
                model_type=self.model_type,
                input_dim=self.input_dim,
                config=model_config,
                device=self.device
            )
        except Exception as e:
            # 模型创建失败（如配置不合理），返回一个很大的值
            return float("inf")
        
        # 3. 检查参数量约束
        n_params = count_parameters(model)
        param_penalty = calculate_param_penalty(model, self.param_budget)
        
        # 设置用户属性以便后续分析
        trial.set_user_attr("n_params", n_params)
        trial.set_user_attr("model_config", model_config)
        
        # 如果严重超预算，直接剪枝
        if n_params > self.param_budget * 1.5:  # 超预算50%直接放弃
            raise optuna.TrialPruned(f"Params {n_params} exceeds budget {self.param_budget} by >50%")
        
        # 4. 训练和评估
        try:
            results = train_and_evaluate(
                model=model,
                data=self.data,
                config=self.training_config,
                device=self.device,
                epochs=self.training_config.get("epochs", 50),
                patience=self.training_config.get("patience", 10),
            )
        except Exception as e:
            # 训练失败
            return float("inf")
        
        # 5. 记录结果
        trial.set_user_attr("test_mae", results["test_mae"])
        trial.set_user_attr("test_r2", results["test_r2"])
        trial.set_user_attr("best_epoch", results["best_epoch"])
        
        if self.trial_queue is not None:
            self.trial_queue.append({
                "trial_number": trial.number,
                "model_config": model_config,
                "n_params": n_params,
                **results,
            })
        
        # 6. 计算目标值（验证MAE + 惩罚）
        objective_value = results["val_mae_best"] + param_penalty
        
        return objective_value
