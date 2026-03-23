"""神经网络训练器模块

提供神经网络训练和完整实验运行的功能。
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

# 导入项目模块
from src.data.loader import load_dataset
from src.evaluation.metrics import compute_metrics
from src.evaluation.result_writer import append_result_row
from src.missing_data.mar import simulate_mar
from src.missing_data.mcar import simulate_mcar
from src.models.model_factory import create_model
from src.trainers.lightning_trainer import get_trainer
from src.utils.seed_manager import set_seed

# 配置日志记录器
logger = logging.getLogger(__name__)


class NeuralNetworkTrainer:
    """
    传统神经网络训练器（不依赖 PyTorch Lightning）
    
    提供基本的神经网络训练功能，包括早停、学习率调度等。
    """
    
    def __init__(self, model, device: str = 'cpu'):
        """
        Args:
            model: PyTorch模型
            device: 计算设备
        """
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.MSELoss()
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader = None,
        epochs: int = 100, 
        lr: float = 0.001, 
        weight_decay: float = 0.0, 
        patience: int = 15
    ) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
            
        Returns:
            dict: 训练历史，包含 train_loss 和 val_loss
        """
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                # 确保输出和目标维度一致
                outputs = outputs.view(-1)
                batch_y = batch_y.view(-1)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(batch_x)
            
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            # 验证阶段
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_x)
                        # 确保输出和目标维度一致
                        outputs = outputs.view(-1)
                        batch_y = batch_y.view(-1)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item() * len(batch_x)
                
                val_loss /= len(val_loader.dataset)
                history['val_loss'].append(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型状态
                    best_model_state = {
                        k: v.cpu().clone() 
                        for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"早停于 epoch {epoch+1}")
                    break
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f", Val Loss: {val_loss:.6f}"
                logger.info(msg)
        
        training_time = time.time() - start_time
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        history['training_time'] = training_time
        history['best_epoch'] = len(history['train_loss']) - patience_counter
        
        return history


def run_experiment(cfg: DictConfig) -> None:
    """
    运行完整实验
    
    根据配置执行完整实验流程，包括:
    1. 加载数据
    2. 对每个 missing_rate 和 seed:
       - 生成缺失数据（MIM模式下训练集使用多缺失率混合）
       - 训练模型
       - 评估并记录结果
    
    Args:
        cfg: OmegaConf 配置对象，包含完整的实验配置
        
    Returns:
        None，结果将保存到 CSV 文件
        
    Raises:
        ValueError: 当配置无效或缺失必要字段时
        FileNotFoundError: 当数据文件不存在时
        Exception: 当实验执行过程中发生错误时
        
    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.load("config.yaml")
        >>> run_experiment(cfg)
    """
    start_time = time.time()
    
    try:
        # 验证配置
        _validate_config(cfg)
        
        # 加载数据
        logger.info("Loading dataset...")
        data = load_dataset(cfg)
        logger.info(
            f"Dataset loaded: train={len(data['X_train'])}, "
            f"val={len(data['X_val'])}, test={len(data['X_test'])}"
        )
        
        # 确定要运行的 missing_rates
        missing_rates = _get_missing_rates(cfg)
        logger.info(f"Will run experiments for missing_rates: {missing_rates}")
        
        # 获取种子列表
        seeds = cfg.experiments.training.get("seeds", [42])
        if isinstance(seeds, int):
            seeds = [seeds]
        
        total_runs = len(missing_rates) * len(seeds)
        run_count = 0
        
        # 检查是否使用 MIM 方法
        use_mim = cfg.experiments.experiment.get("use_mim", False)
        if use_mim:
            logger.info("[MIM Mode] Training set will use multi-missing-rate augmentation (0.0, 0.1, ..., 0.9)")
        
        for mr in missing_rates:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running experiments with missing_rate = {mr}")
            logger.info(f"{'='*60}\n")
            
            for seed in seeds:
                run_count += 1
                logger.info(f"[{run_count}/{total_runs}] Seed: {seed}")
                
                try:
                    # 设置随机种子
                    set_seed(seed)
                    
                    # 准备数据
                    X_train, y_train = data["X_train"], data["y_train"]
                    X_val, y_val = data["X_val"], data["y_val"]
                    X_test, y_test = data["X_test"], data["y_test"]
                    
                    # 创建 DataLoader
                    batch_size = cfg.experiments.training.get("batch_size", 32)
                    
                    if use_mim:
                        # MIM模式：训练集使用多缺失率混合（0.0, 0.1, ..., 0.9）
                        train_loader = _create_mim_train_loader(
                            cfg, X_train, y_train, batch_size, seed
                        )
                        # 验证集和测试集使用当前 missing_rate
                        _, val_input, test_input = _apply_missing_mechanism(
                            cfg, X_train, y_train, X_val, y_val, X_test, y_test, mr, seed
                        )
                    else:
                        # Baseline模式：所有数据集使用当前 missing_rate
                        train_input, val_input, test_input = _apply_missing_mechanism(
                            cfg, X_train, y_train, X_val, y_val, X_test, y_test, mr, seed
                        )
                        train_loader = DataLoader(
                            TensorDataset(train_input, y_train),
                            batch_size=batch_size,
                            shuffle=True,
                        )
                    
                    val_loader = DataLoader(
                        TensorDataset(val_input, y_val),
                        batch_size=batch_size,
                        shuffle=False,
                    )
                    test_loader = DataLoader(
                        TensorDataset(test_input, y_test),
                        batch_size=batch_size,
                        shuffle=False,
                    )
                    
                    # 创建模型
                    model = create_model(cfg)
                    logger.debug(f"Model created: {cfg.models.name}")
                    
                    # 创建 trainer 并训练
                    trainer = get_trainer(cfg)
                    trainer.fit(model, train_loader, val_loader)
                    
                    # 测试
                    test_results = trainer.test(model, test_loader)
                    
                    if test_results and len(test_results) > 0:
                        test_results = test_results[0]
                    else:
                        test_results = {}
                        logger.warning("No test results returned from trainer")
                    
                    # 添加元信息
                    result_row = {
                        "seed": seed,
                        "missing_rate": mr,
                        "model": cfg.models.name,
                        "missing_mode": cfg.missing.mode,
                        "use_mim": use_mim,
                    }
                    result_row.update(test_results)
                    
                    # 保存结果
                    append_result_row(result_row, cfg)
                    logger.info(f"Results saved for seed={seed}, mr={mr}")
                    
                except Exception as e:
                    logger.error(f"Error in run with seed={seed}, mr={mr}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # 继续下一个种子，不中断整个实验
                    continue
        
        elapsed_time = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment completed! Total time: {elapsed_time:.2f}s")
        logger.info(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def _validate_config(cfg: DictConfig) -> None:
    """
    验证配置对象是否包含必要字段
    
    Args:
        cfg: 配置对象
        
    Raises:
        ValueError: 当配置缺失必要字段时
    """
    required_sections = ["data", "models", "experiments", "missing"]
    
    for section in required_sections:
        if not hasattr(cfg, section):
            raise ValueError(f"配置缺少必要部分: {section}")
    
    # 验证缺失模式
    valid_modes = ["mcar", "mar"]
    if cfg.missing.mode not in valid_modes:
        raise ValueError(f"未知的缺失模式: {cfg.missing.mode}. 支持: {valid_modes}")
    
    # 验证 MAR 参数
    if cfg.missing.mode == "mar":
        if not hasattr(cfg.missing, "beta") or not hasattr(cfg.missing, "gamma"):
            raise ValueError("MAR 模式需要配置 beta 和 gamma 参数")
    
    logger.debug("Config validation passed")


def _get_missing_rates(cfg: DictConfig) -> List[float]:
    """
    从配置中提取缺失率列表
    
    Args:
        cfg: 配置对象
        
    Returns:
        缺失率列表
    """
    if hasattr(cfg.experiments.experiment, 'missing_rate'):
        # 单一 MR 场景
        mr = cfg.experiments.experiment.missing_rate
        return [mr] if not isinstance(mr, (list, tuple)) else list(mr)
    else:
        # 多 MR 场景
        return cfg.missing.get("missing_rates", [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])


def _create_mim_train_loader(
    cfg: DictConfig,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    batch_size: int,
    seed: int
) -> DataLoader:
    """
    创建 MIM 模式的训练 DataLoader
    
    将训练集复制多份，每份应用不同的缺失率（0.0, 0.1, ..., 0.9），
    然后合并成一个大训练集。
    
    Args:
        cfg: 配置对象
        X_train: 原始训练特征 [N, D]
        y_train: 原始训练目标 [N]
        batch_size: 批次大小
        seed: 随机种子
        
    Returns:
        训练用的 DataLoader
    """
    # MIM 训练缺失率：0.0, 0.1, 0.2, ..., 0.9（共10份）
    training_missing_rates = [i / 10.0 for i in range(10)]  # [0.0, 0.1, ..., 0.9]
    
    all_train_inputs = []
    all_train_targets = []
    
    # 获取 MAR 参数（如果使用 MAR 模式）
    beta = cfg.missing.get('beta', 2.0)
    gamma = cfg.missing.get('gamma', 0.05)
    alpha = cfg.missing.get('alpha', None)
    
    for i, mr in enumerate(training_missing_rates):
        # 为每份使用不同的种子，确保多样性
        mr_seed = seed + i * 100
        
        # 应用 MCAR 缺失（训练时使用 MCAR 即可）
        # MIM 的核心是 mask，缺失机制本身不重要
        X_imp, mask, mim_input = simulate_mcar(X_train, mr, mr_seed)
        
        all_train_inputs.append(mim_input)
        all_train_targets.append(y_train)
        
        logger.debug(f"MIM training copy {i}: missing_rate={mr:.1f}, shape={mim_input.shape}")
    
    # 合并所有训练集
    combined_train_input = torch.cat(all_train_inputs, dim=0)  # [10*N, 2D]
    combined_train_target = torch.cat(all_train_targets, dim=0)  # [10*N]
    
    logger.info(
        f"MIM training set created: {len(training_missing_rates)} copies, "
        f"original size={len(X_train)}, augmented size={len(combined_train_input)}"
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        TensorDataset(combined_train_input, combined_train_target),
        batch_size=batch_size,
        shuffle=True,
    )
    
    return train_loader


def _apply_missing_mechanism(
    cfg: DictConfig,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    missing_rate: float,
    seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    应用缺失机制到数据集
    
    Args:
        cfg: 配置对象
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        X_test, y_test: 测试数据
        missing_rate: 缺失率
        seed: 随机种子
        
    Returns:
        train_input, val_input, test_input: 处理后的输入数据
        
    Raises:
        ValueError: 当缺失机制未知时
    """
    use_mim = cfg.experiments.experiment.get("use_mim", False)
    
    if cfg.missing.mode == "mcar":
        # MCAR: 完全随机缺失
        logger.debug(f"Applying MCAR with rate={missing_rate}, seed={seed}")
        
        Xtr_imp, mask_tr, mim_tr = simulate_mcar(X_train, missing_rate, seed)
        Xval_imp, mask_val, mim_val = simulate_mcar(X_val, missing_rate, seed + 999)
        Xte_imp, mask_te, mim_te = simulate_mcar(X_test, missing_rate, seed + 1000)
        
    elif cfg.missing.mode == "mar":
        # MAR: 依赖 SOH 的缺失
        logger.debug(f"Applying MAR with rate={missing_rate}, seed={seed}")
        
        alpha = cfg.missing.get('alpha', None)
        beta = cfg.missing.beta
        gamma = cfg.missing.gamma
        
        Xtr_imp, mask_tr, mim_tr = simulate_mar(
            X_train, y_train, missing_rate, alpha, beta, gamma, seed
        )
        Xval_imp, mask_val, mim_val = simulate_mar(
            X_val, y_val, missing_rate, alpha, beta, gamma, seed + 999
        )
        Xte_imp, mask_te, mim_te = simulate_mar(
            X_test, y_test, missing_rate, alpha, beta, gamma, seed + 1000
        )
    else:
        raise ValueError(f"Unknown missing mode: {cfg.missing.mode}")
    
    # 选择输入（MIM 或仅插补）
    if use_mim:
        logger.debug("Using MIM input (doubled dimensions)")
        train_input, val_input, test_input = mim_tr, mim_val, mim_te
    else:
        logger.debug("Using imputed input only")
        train_input, val_input, test_input = Xtr_imp, Xval_imp, Xte_imp
    
    return train_input, val_input, test_input
