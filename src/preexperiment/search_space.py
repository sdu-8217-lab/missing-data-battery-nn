"""
模型架构搜索空间定义

为不同参数量预算（2^13 ~ 2^16）定义合理的搜索空间边界
"""
from typing import Dict, List, Any
import optuna


class SearchSpace:
    """参数化搜索空间生成器"""
    
    BUDGETS = [8192, 16384, 32768, 65536]
    
    @staticmethod
    def get_budget_factor(param_budget: int) -> float:
        return param_budget / 8192
    
    @staticmethod
    def _estimate_mlp_params(input_dim: int, hidden_dims: List[int]) -> int:
        """估算MLP参数量"""
        layers = [input_dim] + hidden_dims + [1]
        return sum(layers[i] * layers[i+1] + layers[i+1] for i in range(len(layers)-1))
    
    @classmethod
    def suggest_mlp_params(cls, trial: optuna.Trial, input_dim: int = 16,
                          param_budget: int = 8192) -> Dict[str, Any]:
        """MLP搜索空间（2-8层）"""
        budget_factor = cls.get_budget_factor(param_budget)
        
        # 层数: 2-8层
        max_layers = min(2 + int(budget_factor * 2), 8)
        n_layers = trial.suggest_int("n_layers", 2, max_layers)
        
        # 计算每层最大可行维度（简化估算）
        # 假设每层相同: params ≈ n_layers * hidden^2 + hidden * input_dim
        approx_hidden = int((param_budget / n_layers) ** 0.5)
        upper_bound = min(1024, max(approx_hidden * 2, 64))
        
        hidden_dims = []
        for i in range(n_layers):
            dim = trial.suggest_int(f"hidden_{i}", 16, upper_bound, log=True)
            hidden_dims.append(dim)
        
        max_dropout = min(0.05 + 0.08 * budget_factor, 0.4)
        dropout = trial.suggest_float("dropout", 0.0, max_dropout, step=0.05)
        
        return {"hidden_dims": hidden_dims, "dropout": dropout}
    
    @staticmethod
    def _estimate_lstm_params(input_dim: int, hidden_size: int, num_layers: int) -> int:
        """估算LSTM参数量"""
        total = 0
        for layer in range(num_layers):
            in_size = input_dim if layer == 0 else hidden_size
            total += 4 * (in_size * hidden_size + hidden_size * hidden_size + hidden_size)
        return total + hidden_size + 1
    
    @classmethod
    def suggest_lstm_params(cls, trial: optuna.Trial, input_dim: int = 16,
                           param_budget: int = 8192) -> Dict[str, Any]:
        """LSTM搜索空间（1-3层）"""
        budget_factor = cls.get_budget_factor(param_budget)
        
        max_layers = min(1 + int(budget_factor), 3)
        num_layers = trial.suggest_int("num_layers", 1, max_layers)
        
        # 计算最大可行hidden_size
        approx_hidden = int((param_budget / (4 * num_layers)) ** 0.5)
        max_hidden = min(512, max(approx_hidden * 2, 48))
        
        hidden_size = trial.suggest_int("hidden_size", 16, max_hidden, log=True)
        
        dropout = trial.suggest_float("dropout", 0.0, 
                                     min(0.05 + 0.1 * budget_factor, 0.3), step=0.05) if num_layers > 1 else 0.0
        
        return {"hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout}
    
    @staticmethod
    def _estimate_gru_params(input_dim: int, hidden_size: int, num_layers: int) -> int:
        """估算GRU参数量"""
        total = 0
        for layer in range(num_layers):
            in_size = input_dim if layer == 0 else hidden_size
            total += 3 * (in_size * hidden_size + hidden_size * hidden_size + hidden_size)
        return total + hidden_size + 1
    
    @classmethod
    def suggest_gru_params(cls, trial: optuna.Trial, input_dim: int = 16,
                          param_budget: int = 8192) -> Dict[str, Any]:
        """GRU搜索空间（1-3层）"""
        budget_factor = cls.get_budget_factor(param_budget)
        
        max_layers = min(1 + int(budget_factor), 3)
        num_layers = trial.suggest_int("num_layers", 1, max_layers)
        
        approx_hidden = int((param_budget / (3 * num_layers)) ** 0.5)
        max_hidden = min(640, max(approx_hidden * 2, 48))
        
        hidden_size = trial.suggest_int("hidden_size", 16, max_hidden, log=True)
        
        dropout = trial.suggest_float("dropout", 0.0,
                                     min(0.05 + 0.1 * budget_factor, 0.3), step=0.05) if num_layers > 1 else 0.0
        
        return {"hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout}
    
    @staticmethod
    def _estimate_cnn1d_params(input_dim: int, channels: List[int], kernel_size: int) -> int:
        """估算CNN1D参数量"""
        total = 0
        in_ch = input_dim
        for out_ch in channels:
            total += out_ch * in_ch * kernel_size + out_ch
            in_ch = out_ch
        return total + channels[-1] + 1
    
    @classmethod
    def suggest_cnn1d_params(cls, trial: optuna.Trial, input_dim: int = 16,
                            param_budget: int = 8192) -> Dict[str, Any]:
        """CNN1D搜索空间（1-3层）"""
        budget_factor = cls.get_budget_factor(param_budget)
        
        max_conv_layers = min(1 + int(budget_factor), 3)
        n_conv_layers = trial.suggest_int("n_conv_layers", 1, max_conv_layers)
        
        kernel_size = trial.suggest_int("kernel_size", 3, 5)
        
        # 计算全局最大通道数
        approx_ch = int((param_budget / (n_conv_layers * kernel_size * input_dim)) ** 0.5)
        upper_bound = min(512, max(approx_ch * 2, 32))
        
        channels = []
        for i in range(n_conv_layers):
            out_ch = trial.suggest_int(f"channel_{i}", 8, upper_bound, log=True)
            channels.append(out_ch)
        
        max_dropout = min(0.05 + 0.05 * budget_factor, 0.35)
        dropout = trial.suggest_float("dropout", 0.0, max_dropout, step=0.05)
        
        return {"channels": channels, "kernel_size": kernel_size, "dropout": dropout}
    
    @classmethod
    def get_suggest_func(cls, model_type: str):
        funcs = {
            "mlp": cls.suggest_mlp_params,
            "lstm": cls.suggest_lstm_params,
            "gru": cls.suggest_gru_params,
            "cnn1d": cls.suggest_cnn1d_params,
        }
        return funcs.get(model_type.lower())
