import os
import datetime
import logging
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =========================
# 1. 日志配置
# =========================
# 配置日志处理器以支持UTF-8编码
class UTF8FileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
        super().__init__(filename, mode, encoding, delay)

# 创建日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 清除现有处理器
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# 控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# 文件处理器（UTF-8编码）
log_file = f'soh_estimation_experiment_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
file_handler = UTF8FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# =========================
# 2. 命令行参数与配置
# =========================
parser = argparse.ArgumentParser(description='优化SOH估计模型，处理缺失数据')
parser.add_argument('--missing_rates', type=float, nargs='+', default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                   help='要测试的缺失率列表 (0.05-0.95)')
parser.add_argument('--training_missing_rates', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                   help='训练缺失指示器模型时使用的缺失率列表，包含0.0(完整数据)')
parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
parser.add_argument('--data_dir', type=str, default='./data/XJTU data',
                   help='数据集根目录')
parser.add_argument('--batch', type=str, default='3C', choices=['2C','3C','R2.5','R3','RW','satellite'],
                   help='电池批次')
parser.add_argument('--seed', type=int, default=42, help='用于可复现性的随机种子')
parser.add_argument('--batch_size', type=int, default=32, help='训练批次大小')
parser.add_argument('--results_dir', type=str, default='optimized_results', help='结果保存目录')
args = parser.parse_args()

# =========================
# 3. 设置随机种子（全面）
# =========================
def set_seeds(seed):
    """设置所有随机种子以确保可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seeds(args.seed)

# 确保结果目录存在
results_dir = Path(args.results_dir)
results_dir.mkdir(parents=True, exist_ok=True)

# 生成全局唯一标识符（时间戳）
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logger.info(f"实验时间戳: {timestamp}")

# =========================
# 4. 优化的数据清洗函数
# =========================
def clean_data_optimized(df, feature_cols, target_col):
    """优化的数据清洗函数，使用向量化操作提高效率"""
    logger.info("正在清洗数据: 移除inf值和异常值...")
    
    # 1. 替换无穷大值为NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 2. 删除包含NaN的行
    original_shape = df.shape
    df = df.dropna(subset=feature_cols + [target_col], how='any')
    logger.info(f"  已移除 {original_shape[0] - df.shape[0]} 行包含NaN/inf值的数据")
    
    # 3. 使用向量化操作进行3-sigma异常值检测
    outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in feature_cols + [target_col]:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    # 移除异常值
    outlier_mask = outlier_mask.any(axis=1)
    if outlier_mask.any():
        df = df[~outlier_mask]
        logger.info(f"  移除 {outlier_mask.sum()} 行异常值")
    
    logger.info(f"  清洗后数据形状: {df.shape}")
    return df

# =========================
# 5. 训练集扩充函数 (高效实现)
# =========================
def expand_training_set_efficient(X_train, y_train, missing_rates):
    """
    高效扩充训练集以适应带缺失指示器的神经网络
    根据给定的缺失率列表生成多个缺失副本，包含完整数据(缺失率=0)
    """
    logger.info(f"正在扩充训练集，缺失率范围: {missing_rates}")
    
    expanded_X = []
    expanded_y = []
    
    # 为每个缺失率生成缺失副本
    for mr in missing_rates:
        logger.info(f"  生成缺失率为 {mr*100:.0f}% 的副本...")
        
        # 使用高效的方式生成缺失掩码
        mask = np.random.rand(*X_train.shape) > mr  # 更高效的方法
        
        # 应用缺失（将缺失位置设为0，对应标准化后的均值）
        X_with_missing = np.where(mask, X_train, 0)
        
        # 生成缺失指示器向量（1表示缺失，0表示存在）
        missing_indicators = 1 - mask
        
        # 将原始特征与缺失指示器拼接
        X_combined = np.concatenate([X_with_missing, missing_indicators], axis=1)
        
        expanded_X.append(X_combined)
        expanded_y.append(y_train)  # 标签保持不变
        
        logger.info(f"    缺失副本形状: {X_combined.shape}")
    
    # 合并所有数据
    X_expanded = np.vstack(expanded_X)
    y_expanded = np.concatenate(expanded_y)
    
    logger.info(f"扩展训练集完成 - 特征: {X_expanded.shape}, 标签: {y_expanded.shape}")
    return X_expanded, y_expanded

# =========================
# 6. Dataset 定义 (优化)
# =========================
class OptimizedBatteryDatasetFixedMissing(Dataset):
    """优化的固定缺失率数据集，用于验证和测试"""
    def __init__(self, X, y, missing_rate=0.0, include_missing_indicators=True, missing_mask=None):
        self.X = X
        self.y = y
        self.missing_rate = missing_rate
        self.include_missing_indicators = include_missing_indicators
        
        # 如果没有提供预定义的缺失掩码，则生成新的
        if missing_mask is None:
            self.missing_mask = np.random.rand(*X.shape) > missing_rate  # 更高效
        else:
            self.missing_mask = missing_mask
        
        # 创建缺失指示器 (1表示缺失, 0表示存在)
        self.missing_indicators = 1 - self.missing_mask
        
        # 应用缺失 (将缺失位置设为0)
        self.X_with_missing = np.where(self.missing_mask, self.X, 0)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # 获取特征值
        feature_values = torch.tensor(self.X_with_missing[idx], dtype=torch.float32)
        target = torch.tensor(self.y[idx], dtype=torch.float32)
        
        # 如果需要，添加缺失指示器
        if self.include_missing_indicators:
            missing_indicators = torch.tensor(self.missing_indicators[idx], dtype=torch.float32)
            combined_features = torch.cat([feature_values, missing_indicators])
            return combined_features, target
        else:
            return feature_values, target

class OptimizedMeanFillDataset(Dataset):
    """优化的均值填充方法数据集，使用0填充（标准化后的均值）"""
    def __init__(self, X, y, missing_mask=None):
        self.X = X.copy()
        self.y = y
        
        # 使用提供的缺失掩码
        self.missing_mask = missing_mask
        
        # 应用缺失
        self.features_with_missing = np.where(self.missing_mask, self.X, np.nan)
        
        # 使用0填充（标准化后的均值）
        self.features_filled = np.where(np.isnan(self.features_with_missing), 0.0, self.features_with_missing)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features_filled[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# =========================
# 7. 模型定义 (优化)
# =========================
class OptimizedSOHNetwork(nn.Module):
    """优化的SOH估计神经网络，支持不同输入大小"""
    def __init__(self, input_size):
        super(OptimizedSOHNetwork, self).__init__()
        
        # 根据输入大小动态设计网络结构
        if input_size == 16:  # 均值填充方法和完整数据基线(无指示器)
            self.net = nn.Sequential(
                nn.Linear(16, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        elif input_size == 32:  # 缺失指示器方法
            self.net = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        else:
            # 通用结构，处理其他输入大小
            hidden1 = min(128, max(32, input_size * 2))
            hidden2 = min(64, max(16, hidden1 // 2))
            hidden3 = min(32, max(8, hidden2 // 2))
            
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, hidden3),
                nn.ReLU(),
                nn.Linear(hidden3, 1)
            )
    
    def forward(self, x):
        output = self.net(x)
        # 确保输出至少是一维的，防止0维张量问题
        if output.dim() == 0:
            output = output.unsqueeze(0)
        elif output.size() == torch.Size([]):
            output = output.view(1)
        return output.squeeze()

# =========================
# 8. 优化的训练函数（包含早停机制和学习率调度）
# =========================
def train_model_optimized(model, train_loader, val_loader, epochs, device, save_path=None, patience=15):
    """
    优化的训练模型函数，包含早停机制和安全模型保存
    
    参数:
    model: 要训练的模型
    train_loader: 训练数据加载器
    val_loader: 验证数据加载器
    epochs: 最大训练轮数
    device: 训练设备
    save_path: 模型保存路径
    patience: 早停耐心值
    
    返回:
    model: 训练好的模型
    train_losses: 训练损失历史
    val_losses: 验证损失历史
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    logger.info(f"开始训练模型，总轮数: {epochs}")
    start_time = datetime.datetime.now()
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        num_batches = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            num_batches += 1
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_path:
                # 使用安全的模型保存方式
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"在第 {epoch+1} 轮提前停止训练，最佳验证损失: {best_val_loss:.6f}")
                break
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            logger.info(f"轮次 {epoch+1}/{epochs}, 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
    
    end_time = datetime.datetime.now()
    training_duration = end_time - start_time
    logger.info(f"模型训练完成，总耗时: {training_duration}, 最佳验证损失: {best_val_loss:.6f}")
    
    # 加载最佳模型（使用安全加载）
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True))
    
    return model, train_losses, val_losses

# =========================
# 9. 评估函数（优化版）
# =========================
def evaluate_model_optimized(model, loader, device):
    """评估模型性能（使用sklearn.metrics）"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # 确保输出是正确的维度
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            elif outputs.dim() == 1 and outputs.size(0) == 1:
                outputs = outputs.unsqueeze(0) if outputs.numel() == 1 else outputs
            predictions.extend(outputs.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 使用sklearn标准指标计算
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)  # 自动处理分母为0情况
    
    return mae, rmse, r2, predictions, targets

def evaluate_mean_filling_optimized(model, X_test, y_test, missing_mask, device, batch_size=32):
    """评估基线模型在均值填充数据上的表现，复用预训练模型"""
    # 创建均值填充数据集
    fill_test_dataset = OptimizedMeanFillDataset(X_test, y_test, missing_mask=missing_mask)
    test_loader_fill = DataLoader(fill_test_dataset, batch_size=batch_size)
    
    # 评估
    mae, rmse, r2, preds, _ = evaluate_model_optimized(model, test_loader_fill, device)
    
    return mae, rmse, r2, preds

# =========================
# 10. 可视化和结果保存函数 (优化)
# =========================
def plot_training_results_optimized(train_losses, val_losses, missing_rate, method_name, save_dir):
    """绘制训练过程的损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.plot(val_losses, 'r--', label='Validation Loss')
    plt.title(f'Training Curves ({method_name}, Missing Rate: {missing_rate*100:.0f}%)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/training_curves_{method_name}_mr_{missing_rate*100:.0f}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison_results_optimized(true_values, indicator_preds, fill_preds, baseline_preds, missing_rate, save_dir):
    """绘制不同方法的预测结果对比"""
    plt.figure(figsize=(12, 8))
    
    # 预测结果对比
    plt.plot(true_values, 'k-', label='True SOH', linewidth=2.5)
    plt.plot(indicator_preds, 'r--', label='Missing Indicators', linewidth=1.5, alpha=0.9)
    plt.plot(fill_preds, 'g-.', label='Mean Filling (Base Model)', linewidth=1.5, alpha=0.9)
    plt.plot(baseline_preds, 'b:', label='Baseline (Complete Data)', linewidth=1.5, alpha=0.9)
    
    plt.title(f'SOH Estimation Comparison (Missing Rate: {missing_rate*100:.0f}%)')
    plt.xlabel('Sample Index')
    plt.ylabel('SOH')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prediction_comparison_mr_{missing_rate*100:.0f}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_distribution_optimized(indicator_errors, fill_errors, missing_rate, save_dir):
    """绘制误差分布"""
    plt.figure(figsize=(10, 6))
    plt.boxplot([indicator_errors, fill_errors], tick_labels=['Missing Indicators', 'Mean Filling (Base Model)'])
    plt.title(f'Absolute Error Distribution (Missing Rate: {missing_rate*100:.0f}%)')
    plt.ylabel('Absolute Error')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/error_distribution_mr_{missing_rate*100:.0f}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_comprehensive_results_optimized(results, save_dir):
    """生成综合比较图表"""
    missing_rates = results['missing_rates']
    
    plt.figure(figsize=(15, 12))
    
    # MAE随缺失率变化趋势
    plt.subplot(2, 2, 1)
    plt.plot(missing_rates, results['indicator_mae'], 'r-o', label='Missing Indicators', linewidth=2)
    plt.plot(missing_rates, results['fill_mae'], 'g--s', label='Mean Filling (Base Model)', linewidth=2)
    plt.axhline(y=results['baseline_mae'], color='k', linestyle='-', label='Baseline (Complete Data)', linewidth=3)
    plt.xlabel('Missing Rate')
    plt.ylabel('MAE')
    plt.title('MAE vs Missing Rate')
    plt.legend()
    plt.grid(True)
    
    # RMSE随缺失率变化趋势
    plt.subplot(2, 2, 2)
    plt.plot(missing_rates, results['indicator_rmse'], 'r-o', label='Missing Indicators', linewidth=2)
    plt.plot(missing_rates, results['fill_rmse'], 'g--s', label='Mean Filling (Base Model)', linewidth=2)
    plt.axhline(y=results['baseline_rmse'], color='k', linestyle='-', label='Baseline (Complete Data)', linewidth=3)
    plt.xlabel('Missing Rate')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Missing Rate')
    plt.legend()
    plt.grid(True)
    
    # R²随缺失率变化趋势
    plt.subplot(2, 2, 3)
    plt.plot(missing_rates, results['indicator_r2'], 'r-o', label='Missing Indicators', linewidth=2)
    plt.plot(missing_rates, results['fill_r2'], 'g--s', label='Mean Filling (Base Model)', linewidth=2)
    plt.axhline(y=results['baseline_r2'], color='k', linestyle='-', label='Baseline (Complete Data)', linewidth=3)
    plt.xlabel('Missing Rate')
    plt.ylabel('R² Score')
    plt.title('R² Score vs Missing Rate')
    plt.legend()
    plt.grid(True)
    
    # 改进百分比
    plt.subplot(2, 2, 4)
    improvements = results['improvement']
    bars = plt.bar(range(len(missing_rates)), improvements, 
                  color=['green' if x > 0 else 'red' for x in improvements],
                  alpha=0.7, tick_label=[f'{mr*100:.0f}%' for mr in missing_rates])
    plt.xlabel('Missing Rate')
    plt.ylabel('Improvement (%)')
    plt.title('Improvement of Missing Indicators over Mean Filling')
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.grid(True, alpha=0.3)
    
    # 在柱子上添加数值
    for i, improvement in enumerate(improvements):
        plt.text(i, improvement + (0.5 if improvement >= 0 else -0.5), 
                f'{improvement:.1f}%', ha='center', 
                va='bottom' if improvement >= 0 else 'top',
                fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comprehensive_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results_to_csv_optimized(results, save_dir):
    """将结果保存到CSV文件"""
    results_df = pd.DataFrame({
        'Missing_Rate': results['missing_rates'],
        'Baseline_MAE': [results['baseline_mae']] * len(results['missing_rates']),
        'Baseline_RMSE': [results['baseline_rmse']] * len(results['missing_rates']),
        'Baseline_R2': [results['baseline_r2']] * len(results['missing_rates']),
        'Indicator_MAE': results['indicator_mae'],
        'Indicator_RMSE': results['indicator_rmse'],
        'Indicator_R2': results['indicator_r2'],
        'Fill_MAE': results['fill_mae'],
        'Fill_RMSE': results['fill_rmse'],
        'Fill_R2': results['fill_r2'],
        'Improvement_Percent': results['improvement']
    })
    csv_path = f'{save_dir}/experiment_results_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info(f"详细结果已保存至 {csv_path}")
    return csv_path

# =========================
# 11. 主程序 (重构核心逻辑)
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 定义特征和目标列
    feature_cols = ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness', 
                   'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
                   'current mean', 'current std', 'current kurtosis', 'current skewness',
                   'CV Q', 'CV charge time', 'current slope', 'current entropy']
    target_col = 'capacity'
    
    # 获取数据文件列表
    data_dir = Path(args.data_dir)
    logger.info(f"正在从目录加载数据: {data_dir}, 批次: {args.batch}")
    
    # 获取该批次下所有电池文件
    pattern = f"{args.batch}_battery-*.csv"
    all_files = list(data_dir.glob(pattern))
    if not all_files:
        raise ValueError(f"在目录 {data_dir} 下未找到匹配 {pattern} 的文件")
    all_files.sort()  # 确保顺序
    logger.info(f"找到 {len(all_files)} 个电池文件: {[f.name for f in all_files]}")
    
    # ===== 完全随机划分训练/测试集 (20%测试) =====
    # 使用随机种子确保可复现性，但划分是随机的
    train_files, test_files = train_test_split(
        all_files, test_size=0.2, random_state=args.seed
    )
    
    logger.info(f"训练电池文件 ({len(train_files)}): {[f.name for f in train_files]}")
    logger.info(f"测试电池文件 ({len(test_files)}): {[f.name for f in test_files]}")
    
    if not train_files or not test_files:
        raise ValueError("训练集或测试集为空，请检查数据文件数量")

    # 加载并处理训练电池数据
    X_train_list = []
    y_train_list = []
    
    for file_path in train_files:
        df = pd.read_csv(file_path)
        logger.info(f"\n加载训练电池: {file_path.name}, 原始形状: {df.shape}")
        df = clean_data_optimized(df, feature_cols, target_col)
        if df.empty:
            logger.warning(f"  警告: 电池 {file_path.name} 在清洗后无数据，跳过")
            continue
            
        # 计算该电池的SOH（基于初始容量）
        initial_capacity = df[target_col].iloc[0]
        y_soh = df[target_col].values / initial_capacity
        X = df[feature_cols].values
        
        X_train_list.append(X)
        y_train_list.append(y_soh)

    # 合并训练电池数据
    if not X_train_list:
        raise ValueError("没有可用的训练数据，请检查数据文件")
        
    X_train_all = np.vstack(X_train_list)
    y_train_all = np.concatenate(y_train_list)
    logger.info(f"\n训练电池合并后形状 - 特征: {X_train_all.shape}, SOH: {y_train_all.shape}")
    
    # 加载并处理测试电池数据
    X_test_list = []
    y_test_list = []
    
    for file_path in test_files:
        df = pd.read_csv(file_path)
        logger.info(f"\n加载测试电池: {file_path.name}, 原始形状: {df.shape}")
        df = clean_data_optimized(df, feature_cols, target_col)
        if df.empty:
            logger.warning(f"  警告: 电池 {file_path.name} 在清洗后无数据，跳过")
            continue
            
        initial_capacity = df[target_col].iloc[0]
        y_soh = df[target_col].values / initial_capacity
        X = df[feature_cols].values
        
        X_test_list.append(X)
        y_test_list.append(y_soh)

    # 合并测试电池数据
    if not X_test_list:
        raise ValueError("没有可用的测试数据，请检查数据文件")
        
    X_test_all = np.vstack(X_test_list)
    y_test_all = np.concatenate(y_test_list)
    logger.info(f"\n测试电池合并后形状 - 特征: {X_test_all.shape}, SOH: {y_test_all.shape}")
    
    # 特征标准化（使用训练集参数）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_all)
    X_test_scaled = scaler.transform(X_test_all)
    
    # 从训练集中划分验证集（20%）
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_scaled, y_train_all, test_size=0.2, random_state=args.seed
    )
    X_test = X_test_scaled
    y_test = y_test_all
    
    logger.info(f"数据集划分完成 - 训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")
    
    # 存储所有结果
    results = {
        'missing_rates': args.missing_rates,
        'baseline_mae': 0, 'baseline_rmse': 0, 'baseline_r2': 0,  # 基线结果只有单一值
        'indicator_mae': [], 'indicator_rmse': [], 'indicator_r2': [],
        'fill_mae': [], 'fill_rmse': [], 'fill_r2': [],
        'improvement': []
    }
    
    # ===== 1. 训练基线模型（完整数据）=====
    logger.info("\n" + "="*60)
    logger.info("正在训练基线模型（完整数据）")
    logger.info("="*60)
    
    # 完整数据基线 - 使用固定的缺失掩码（全1，没有缺失）
    full_train_mask = np.ones_like(X_train)
    full_val_mask = np.ones_like(X_val)
    full_test_mask = np.ones_like(X_test)
    
    full_train_dataset = OptimizedBatteryDatasetFixedMissing(
        X_train, y_train, missing_rate=0.0, 
        include_missing_indicators=False, missing_mask=full_train_mask
    )
    full_val_dataset = OptimizedBatteryDatasetFixedMissing(
        X_val, y_val, missing_rate=0.0, 
        include_missing_indicators=False, missing_mask=full_val_mask
    )
    full_test_dataset = OptimizedBatteryDatasetFixedMissing(
        X_test, y_test, missing_rate=0.0, 
        include_missing_indicators=False, missing_mask=full_test_mask
    )
    
    train_loader = DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(full_val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(full_test_dataset, batch_size=args.batch_size)
    
    model_baseline = OptimizedSOHNetwork(input_size=16).to(device)
    baseline_model_path = f'{args.results_dir}/best_model_baseline_{timestamp}.pth'
    model_baseline, baseline_train_losses, baseline_val_losses = train_model_optimized(
        model_baseline, train_loader, val_loader, args.epochs, device, baseline_model_path
    )
    
    # 评估完整数据基线
    baseline_mae, baseline_rmse, baseline_r2, baseline_preds, _ = evaluate_model_optimized(
        model_baseline, test_loader, device
    )
    
    # 保存基线结果 (单一值，不是列表)
    results['baseline_mae'] = baseline_mae
    results['baseline_rmse'] = baseline_rmse
    results['baseline_r2'] = baseline_r2
    
    logger.info(f"基线模型（完整数据）- MAE: {baseline_mae:.4f}, RMSE: {baseline_rmse:.4f}, R²: {baseline_r2:.4f}")
    plot_training_results_optimized(baseline_train_losses, baseline_val_losses, 0.0, "baseline", args.results_dir)
    
    # ===== 2. 训练缺失指示器模型（一次性训练，通用模型）=====
    logger.info("\n" + "="*60)
    logger.info("正在训练缺失指示器模型（通用模型）")
    logger.info("="*60)
    
    # 一次性扩充训练集，包含完整数据和多种缺失率
    X_train_expanded, y_train_expanded = expand_training_set_efficient(X_train, y_train, args.training_missing_rates)
    
    # 创建扩展训练集的Dataset
    class BatteryDatasetExpanded(Dataset):
        """使用已扩充数据的训练集"""
        def __init__(self, X, y):
            self.X = X
            self.y = y
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            x = torch.tensor(self.X[idx], dtype=torch.float32)
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return x, y
    
    expanded_train_dataset = BatteryDatasetExpanded(X_train_expanded, y_train_expanded)
    
    # 为验证集创建固定缺失率的Dataset (使用中等缺失率0.5作为代表)
    val_mask_indicator = np.random.rand(*X_val.shape) > 0.5  # 50%缺失率作为验证，更高效
    val_dataset_indicator = OptimizedBatteryDatasetFixedMissing(
        X_val, y_val, missing_rate=0.5,
        include_missing_indicators=True, missing_mask=val_mask_indicator
    )
    
    train_loader = DataLoader(expanded_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_indicator, batch_size=args.batch_size)
    
    # 训练单一缺失指示器模型
    model_indicator = OptimizedSOHNetwork(input_size=32).to(device)
    indicator_model_path = f'{args.results_dir}/best_model_indicator_general_{timestamp}.pth'
    model_indicator, indicator_train_losses, indicator_val_losses = train_model_optimized(
        model_indicator, train_loader, val_loader, args.epochs, device, indicator_model_path
    )
    
    logger.info("缺失指示器通用模型训练完成")
    plot_training_results_optimized(indicator_train_losses, indicator_val_losses, 0.5, "indicators_general", args.results_dir)
    
    # ===== 3. 为每个缺失率生成固定的测试掩码，确保可比性 =====
    test_masks = {}
    for i, mr in enumerate(args.missing_rates):
        # 使用不同的随机种子确保不同缺失率的掩码独立
        np.random.seed(args.seed + i)
        mask = np.random.rand(*X_test.shape) > mr  # 更高效的方法
        test_masks[mr] = mask
    
    # ===== 4. 为每个缺失率评估所有方法 =====
    logger.info("\n" + "="*60)
    logger.info("正在测试不同缺失率")
    logger.info("="*60)
    
    for i, missing_rate in enumerate(args.missing_rates):
        logger.info(f"\n" + "-"*60)
        logger.info(f"正在测试缺失率: {missing_rate*100:.0f}%")
        logger.info("-"*60)
        
        # 获取当前缺失率的测试掩码
        test_mask = test_masks[missing_rate]
        
        # ===== 4.1 评估缺失指示器模型 (使用已训练的通用模型) =====
        logger.info(f"\n--- 评估缺失指示器通用模型 (缺失率: {missing_rate*100:.0f}%) ---")
        
        test_dataset_indicator = OptimizedBatteryDatasetFixedMissing(
            X_test, y_test, missing_rate=missing_rate,
            include_missing_indicators=True, missing_mask=test_mask
        )
        
        test_loader_indicator = DataLoader(test_dataset_indicator, batch_size=args.batch_size)
        
        indicator_mae, indicator_rmse, indicator_r2, indicator_preds, _ = evaluate_model_optimized(
            model_indicator, test_loader_indicator, device
        )
        
        results['indicator_mae'].append(indicator_mae)
        results['indicator_rmse'].append(indicator_rmse)
        results['indicator_r2'].append(indicator_r2)
        
        logger.info(f"缺失指示器通用模型 - MAE: {indicator_mae:.4f}, RMSE: {indicator_rmse:.4f}, R²: {indicator_r2:.4f}")
        
        # ===== 4.2 评估均值填充方法（复用基线模型）=====
        logger.info(f"\n--- 评估均值填充方法 (复用基线模型, 缺失率: {missing_rate*100:.0f}%) ---")
        
        # 直接用基线模型评估均值填充数据
        fill_mae, fill_rmse, fill_r2, fill_preds = evaluate_mean_filling_optimized(
            model_baseline, X_test, y_test, test_mask, device, batch_size=args.batch_size
        )
        
        results['fill_mae'].append(fill_mae)
        results['fill_rmse'].append(fill_rmse)
        results['fill_r2'].append(fill_r2)
        
        # 计算改进百分比
        improvement = (fill_mae - indicator_mae) / fill_mae * 100 if fill_mae > 0 else 0.0
        results['improvement'].append(improvement)
        
        logger.info(f"均值填充方法 (复用基线) - MAE: {fill_mae:.4f}, RMSE: {fill_rmse:.4f}, R²: {fill_r2:.4f}")
        logger.info(f"缺失指示器通用模型相对于均值填充方法的改进: {improvement:.1f}%")
        
        # ===== 4.3 保存详细对比图 =====
        logger.info("\n--- 正在保存详细对比图表 ---")
        
        # 预测结果对比
        plot_comparison_results_optimized(y_test, indicator_preds, fill_preds, baseline_preds, missing_rate, args.results_dir)
        
        # 误差分布
        indicator_errors = np.abs(indicator_preds - y_test)
        fill_errors = np.abs(fill_preds - y_test)
        plot_error_distribution_optimized(indicator_errors, fill_errors, missing_rate, args.results_dir)
        
        logger.info(f"缺失率 {missing_rate*100:.0f}% 的详细图表已保存至 {args.results_dir}")
    
    # ===== 5. 生成综合比较图 =====
    logger.info("\n" + "="*60)
    logger.info("正在生成综合对比图表")
    logger.info("="*60)
    plot_comprehensive_results_optimized(results, args.results_dir)
    
    # ===== 6. 保存结果到CSV =====
    logger.info("\n" + "="*60)
    logger.info("正在将结果保存至CSV文件")
    logger.info("="*60)
    csv_path = save_results_to_csv_optimized(results, args.results_dir)
    
    # ===== 7. 打印结果表格 =====
    logger.info("\n" + "="*100)
    logger.info("实验结果摘要")
    logger.info("="*100)
    logger.info(f"{'缺失率':<12} {'方法':<25} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'改进率':<12}")
    logger.info("-"*100)
    
    # 基线结果（单一值）
    logger.info(f"{'完整数据':<12} {'基线模型':<25} {results['baseline_mae']:<10.4f} {results['baseline_rmse']:<10.4f} {results['baseline_r2']:<10.4f} {'-':<12}")
    logger.info("-"*100)
    
    for i, mr in enumerate(results['missing_rates']):
        logger.info(f"{mr*100:>10.0f}%  {'缺失指示器通用模型':<25} {results['indicator_mae'][i]:<10.4f} {results['indicator_rmse'][i]:<10.4f} {results['indicator_r2'][i]:<10.4f} {'-':<12}")
        logger.info(f"{'':<12} {'均值填充 (复用基线)':<25} {results['fill_mae'][i]:<10.4f} {results['fill_rmse'][i]:<10.4f} {results['fill_r2'][i]:<10.4f} {results['improvement'][i]:<10.1f}%")
        logger.info("-"*100)
    
    logger.info("\n" + "="*100)
    logger.info("实验成功完成！")
    logger.info("="*100)
    logger.info(f"结果时间戳: {timestamp}")
    logger.info(f"结果保存目录: {os.path.abspath(args.results_dir)}")
    logger.info(f"详细结果文件: {csv_path}")
    logger.info(f"日志文件: {log_file}")
    logger.info("="*100)

if __name__ == "__main__":
    main()
