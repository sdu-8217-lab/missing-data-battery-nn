import os
import datetime
import logging
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import itertools
import random

# =========================
# 1. 常量定义
# =========================
NUM_FEATURES = 16  # 特征数量常量

# =========================
# 2. 日志配置
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================
# 3. 命令行参数与配置
# =========================
parser = argparse.ArgumentParser(description='Optimized Fixed Missing Pattern Experiment for SOH Estimation')
parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
parser.add_argument('--data_dir', type=str, default='./data/XJTU data',
                   help='数据集根目录')
parser.add_argument('--batch', type=str, default='3C', choices=['2C','3C','R2.5','R3','RW','satellite'],
                   help='电池批次')
parser.add_argument('--seed', type=int, default=42, help='用于可复现性的随机种子')
parser.add_argument('--batch_size', type=int, default=32, help='训练批次大小')
parser.add_argument('--results_dir', type=str, default='optimized_fixed_missing_results',
                   help='结果保存目录')
parser.add_argument('--pretrained_model', type=str, default=None,
                   help='预训练通用模型路径 (可选)')
parser.add_argument('--training_missing_rates', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                   help='训练缺失指示器模型时使用的缺失率列表 (范围: [0.0, 1.0])')
args = parser.parse_args()

# 确保结果目录存在
os.makedirs(args.results_dir, exist_ok=True)

# 生成全局唯一标识符
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logger.info(f"固定缺失实验启动 (时间戳: {timestamp})")

# =========================
# 4. 设置随机种子（全面）
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

# =========================
# 5. 优化的数据处理函数
# =========================
def clean_data_optimized(df, feature_cols, target_col):
    """优化的数据清洗函数，使用向量化操作提高效率"""
    logger.info("正在清洗数据: 移除inf值和异常值...")
    
    # 1. 替换无穷大值为NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 2. 删除包含NaN的行
    original_shape = df.shape
    df = df.dropna()
    df = df.reset_index(drop=True)
    logger.info(f"  已移除 {original_shape[0] - df.shape[0]} 行包含NaN/inf值的数据")
    
    # 3. 应用3-sigma原则移除异常值 (向量化实现)
    outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in feature_cols + [target_col]:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    # 合并异常标记并移除
    outlier_mask = outlier_mask.any(axis=1)
    if outlier_mask.any():
        df = df[~outlier_mask]
        logger.info(f"  正在移除 {outlier_mask.sum()} 行极端异常值")
    
    logger.info(f"  清洗后数据形状: {df.shape}")
    return df

def expand_training_set_efficient(X_train, y_train, missing_rates):
    """高效扩充训练集以适应带缺失指示器的神经网络"""
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
# 6. 优化的数据集类定义
# =========================
class OptimizedBatteryDatasetFixedMissing(Dataset):
    """优化的固定缺失率数据集，用于验证和测试"""
    def __init__(self, X, y, fixed_missing_mask):
        self.X = X
        self.y = y
        self.fixed_missing_mask = fixed_missing_mask  # 固定缺失掩码 (NUM_FEATURES,)
        
        # 创建缺失指示器 (1表示缺失, 0表示存在)
        self.missing_indicators = 1 - fixed_missing_mask
        
        # 应用缺失 (将缺失位置设为0)
        self.X_with_missing = np.where(fixed_missing_mask == 1, self.X, 0)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # 获取特征值
        feature_values = torch.tensor(self.X_with_missing[idx], dtype=torch.float32)
        target = torch.tensor(self.y[idx], dtype=torch.float32)
        
        # 添加缺失指示器
        missing_indicators = torch.tensor(self.missing_indicators, dtype=torch.float32)
        combined_features = torch.cat([feature_values, missing_indicators])
        
        return combined_features, target

class OptimizedBatteryDatasetExpanded(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

class OptimizedBatteryDatasetReduced(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# =========================
# 7. 优化的模型定义
# =========================
class OptimizedSOHNetwork(nn.Module):
    """优化的SOH估计神经网络，支持不同输入大小"""
    def __init__(self, input_size):
        super(OptimizedSOHNetwork, self).__init__()
        
        # 通用网络结构
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
        return self.net(x).squeeze()

# =========================
# 8. 优化的训练函数
# =========================
def train_model_optimized(model, train_loader, val_loader, epochs, device, save_path=None, patience=15):
    """优化的训练模型函数，包含早停机制和安全模型保存"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
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
                logger.info(f"在第 {epoch+1} 轮提前停止训练")
                break
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            logger.info(f"轮次 {epoch+1}/{epochs}, 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
    
    # 加载最佳模型（使用安全加载）
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True))
    
    return model, train_losses, val_losses

# =========================
# 9. 优化的评估函数
# =========================
def evaluate_model_optimized(model, loader, device):
    """优化的模型评估函数，使用sklearn指标"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 使用sklearn标准指标计算
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    
    return mae, rmse, r2, predictions, targets

# =========================
# 10. 优化的可视化和结果保存函数
# =========================
def plot_comparison_results_optimized(true_values, indicator_preds, reduced_preds, missing_comb, save_dir):
    """优化的固定缺失特征预测结果对比图"""
    plt.figure(figsize=(12, 8))
    
    # 预测结果对比
    plt.plot(true_values, 'k-', label='True SOH', linewidth=2.5)
    plt.plot(indicator_preds, 'r--', label='Missing Indicators', linewidth=1.5, alpha=0.9)
    plt.plot(reduced_preds, 'b-.', label='Reduced Model', linewidth=1.5, alpha=0.9)
    
    plt.title(f'SOH Estimation Comparison (Fixed Missing: {missing_comb})')
    plt.xlabel('Sample Index')
    plt.ylabel('SOH')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/missing_comb_{str(missing_comb).replace(" ", "")}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results_to_csv_optimized(results, save_dir):
    """优化的结果保存函数"""
    results_df = pd.DataFrame({
        'Missing_Combination': results['missing_combinations'],
        'Indicator_MAE': results['indicator_mae'],
        'Indicator_RMSE': results['indicator_rmse'],
        'Indicator_R2': results['indicator_r2'],
        'Reduced_MAE': results['reduced_mae'],
        'Reduced_RMSE': results['reduced_rmse'],
        'Reduced_R2': results['reduced_r2'],
        'Improvement_Percent': results['improvement']
    })
    csv_path = f'{save_dir}/all_missing_combinations_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info(f"详细结果已保存至 {csv_path}")
    return csv_path

def plot_overall_comparison_optimized(results, save_dir, timestamp):
    """优化的所有缺失组合的MAE对比总图"""
    plt.figure(figsize=(14, 8))
    
    # 提取数据
    missing_combinations = results['missing_combinations']
    indicator_mae = results['indicator_mae']
    reduced_mae = results['reduced_mae']
    
    # 创建柱状图
    x = np.arange(len(missing_combinations))
    width = 0.35
    
    # 绘制柱状图
    plt.bar(x - width/2, indicator_mae, width, label='Missing Indicators', color='#1f77b4', alpha=0.8)
    plt.bar(x + width/2, reduced_mae, width, label='Reduced Model', color='#ff7f0e', alpha=0.8)
    
    # 添加标签和标题
    plt.xlabel('Missing Combination', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('MAE Comparison Across All Missing Combinations', fontsize=14, fontweight='bold')
    
    # 优化x轴标签
    plt.xticks(x, [str(comb) for comb in missing_combinations], rotation=45, ha='right', fontsize=8)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for i, (ind_mae, red_mae) in enumerate(zip(indicator_mae, reduced_mae)):
        plt.text(i - width/2, ind_mae + 0.0005, f'{ind_mae:.4f}', ha='center', fontsize=6)
        plt.text(i + width/2, red_mae + 0.0005, f'{red_mae:.4f}', ha='center', fontsize=6)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/overall_mae_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"总图已保存至 {save_dir}/overall_mae_comparison_{timestamp}.png")

# =========================
# 11. 主程序 (核心逻辑)
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
    
    # ===== 完全随机划分训练/测试集 =====
    # 随机打乱文件列表（使用设置的种子确保可复现）
    np.random.shuffle(all_files)
    
    # 确保至少有1个测试文件（如果文件数量少于5，则使用20%但至少1个）
    test_size = max(1, min(int(len(all_files) * 0.2), len(all_files) - 1))
    train_size = len(all_files) - test_size
    
    train_files = all_files[:train_size]
    test_files = all_files[train_size:]
    
    logger.info(f"随机划分训练/测试集: 训练集 {train_size} 个文件, 测试集 {test_size} 个文件")
    logger.info(f"训练电池文件: {[f.name for f in train_files]}")
    logger.info(f"测试电池文件: {[f.name for f in test_files]}")
    
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
    
    # ===== 1. 训练通用缺失指示器模型 (32维输入) =====
    logger.info("\n" + "="*60)
    logger.info("正在训练通用缺失指示器模型 (32维输入)")
    logger.info("="*60)
    
    # 扩充训练集
    X_train_expanded, y_train_expanded = expand_training_set_efficient(X_train, y_train, args.training_missing_rates)
    
    # 创建扩展训练集的Dataset
    expanded_train_dataset = OptimizedBatteryDatasetExpanded(X_train_expanded, y_train_expanded)
    
    # 创建验证集 (使用随机缺失掩码)
    feature_missing_vector = np.random.rand(NUM_FEATURES) > 0.5  # 更高效的方法
    val_dataset = OptimizedBatteryDatasetFixedMissing(X_val, y_val, fixed_missing_mask=feature_missing_vector)

    train_loader = DataLoader(expanded_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # 加载预训练模型或重新训练
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        logger.info(f"加载预训练模型: {args.pretrained_model}")
        model_indicator = OptimizedSOHNetwork(input_size=2 * NUM_FEATURES).to(device)
        model_indicator.load_state_dict(torch.load(args.pretrained_model, weights_only=True))
    else:
        logger.info("未提供预训练模型，重新训练通用模型")
        model_indicator = OptimizedSOHNetwork(input_size=2 * NUM_FEATURES).to(device)
        model_path = f'{args.results_dir}/best_model_indicator_{timestamp}.pth'
        model_indicator, _, _ = train_model_optimized(
            model_indicator, train_loader, val_loader, args.epochs, device, model_path
        )
    
    # ===== 2. 生成所有缺失组合 (1-15个特征缺失) =====
    logger.info("\n" + "="*60)
    logger.info("正在生成所有缺失组合 (1-15个特征缺失，每组10个随机组合)")
    logger.info("="*60)
    
    # 生成所有可能的缺失组合 (每种缺失数量k取10个随机组合)
    all_missing_combinations = []
    for k in range(1, NUM_FEATURES):  # k=1 to 15
        # 生成所有可能的组合
        all_combinations = list(itertools.combinations(range(NUM_FEATURES), k))
        
        # 随机选择10个组合 (如果组合数不足10则全部选择)
        if len(all_combinations) > 10:
            selected_combinations = random.sample(all_combinations, 10)
        else:
            selected_combinations = all_combinations
        
        all_missing_combinations.extend(selected_combinations)
    
    logger.info(f"共生成 {len(all_missing_combinations)} 个缺失组合 (每种缺失数量10个)")
    
    # 存储所有结果
    results = {
        'missing_combinations': [],
        'indicator_mae': [], 'indicator_rmse': [], 'indicator_r2': [],
        'reduced_mae': [], 'reduced_rmse': [], 'reduced_r2': [],
        'improvement': []
    }
    
    # ===== 3. 为每个缺失组合测试 =====
    for idx, missing_comb in enumerate(all_missing_combinations):
        logger.info("\n" + "="*60)
        logger.info(f"测试缺失组合 ({idx+1}/{len(all_missing_combinations)}): {missing_comb}")
        logger.info("="*60)
        
        # 创建固定缺失掩码 (当前组合中的特征缺失)
        fixed_missing_mask = np.ones(NUM_FEATURES, dtype=int)
        for feature_idx in missing_comb:
            fixed_missing_mask[feature_idx] = 0  # 设置为缺失
        
        # 创建测试集 (固定缺失)
        test_dataset_indicator = OptimizedBatteryDatasetFixedMissing(
            X_test, y_test, fixed_missing_mask=fixed_missing_mask
        )
        test_loader_indicator = DataLoader(test_dataset_indicator, batch_size=args.batch_size)
        
        # ===== 3.1 评估缺失指示器通用模型 =====
        indicator_mae, indicator_rmse, indicator_r2, indicator_preds, _ = evaluate_model_optimized(
            model_indicator, test_loader_indicator, device
        )
        
        logger.info(f"缺失指示器通用模型: "
                    f"MAE={indicator_mae:.4f}, RMSE={indicator_rmse:.4f}, R²={indicator_r2:.4f}")
        
        # ===== 3.2 训练缩减模型 (移除当前缺失组合的特征) =====
        logger.info(f"\n--- 训练缩减模型 (移除特征 {missing_comb}) ---")
        
        # 创建缩减数据集 (移除缺失组合中的特征)
        # 按索引从大到小排序，避免删除后索引变化
        indices_to_remove = sorted(missing_comb, reverse=True)
        X_train_reduced = np.delete(X_train, indices_to_remove, axis=1)
        X_val_reduced = np.delete(X_val, indices_to_remove, axis=1)
        X_test_reduced = np.delete(X_test, indices_to_remove, axis=1)
        
        # 重新标准化
        scaler_reduced = StandardScaler().fit(X_train_reduced)
        X_train_reduced = scaler_reduced.transform(X_train_reduced)
        X_val_reduced = scaler_reduced.transform(X_val_reduced)
        X_test_reduced = scaler_reduced.transform(X_test_reduced)
        
        # 创建缩减训练集Dataset
        train_dataset_reduced = OptimizedBatteryDatasetReduced(X_train_reduced, y_train)
        val_dataset_reduced = OptimizedBatteryDatasetReduced(X_val_reduced, y_val)
        
        train_loader_reduced = DataLoader(train_dataset_reduced, batch_size=args.batch_size, shuffle=True)
        val_loader_reduced = DataLoader(val_dataset_reduced, batch_size=args.batch_size)
        
        # 训练缩减模型
        input_size = NUM_FEATURES - len(missing_comb)
        model_reduced = OptimizedSOHNetwork(input_size=input_size).to(device)
        model_path_reduced = f'{args.results_dir}/best_model_reduced_{str(missing_comb).replace(" ", "")}_{timestamp}.pth'
        model_reduced, _, _ = train_model_optimized(
            model_reduced, train_loader_reduced, val_loader_reduced, args.epochs, device, model_path_reduced
        )
        
        # 评估缩减模型
        reduced_mae, reduced_rmse, reduced_r2, reduced_preds, _ = evaluate_model_optimized(
            model_reduced, 
            DataLoader(OptimizedBatteryDatasetReduced(X_test_reduced, y_test), batch_size=args.batch_size), 
            device
        )
        
        logger.info(f"缩减模型 (移除特征 {missing_comb}): "
                    f"MAE={reduced_mae:.4f}, RMSE={reduced_rmse:.4f}, R²={reduced_r2:.4f}")
        
        # 计算改进百分比
        improvement = (reduced_mae - indicator_mae) / reduced_mae * 100 if reduced_mae > 0 else 0.0
        
        # 保存结果
        results['missing_combinations'].append(str(missing_comb))
        results['indicator_mae'].append(indicator_mae)
        results['indicator_rmse'].append(indicator_rmse)
        results['indicator_r2'].append(indicator_r2)
        results['reduced_mae'].append(reduced_mae)
        results['reduced_rmse'].append(reduced_rmse)
        results['reduced_r2'].append(reduced_r2)
        results['improvement'].append(improvement)
        
        # 保存详细对比图
        plot_comparison_results_optimized(
            y_test, indicator_preds, reduced_preds, missing_comb, args.results_dir
        )
    
    # ===== 4. 生成综合比较结果 =====
    logger.info("\n" + "="*60)
    logger.info("正在生成综合比较图表")
    logger.info("="*60)
    
    # 保存结果到CSV
    csv_path = save_results_to_csv_optimized(results, args.results_dir)
    
    # 生成总图
    plot_overall_comparison_optimized(results, args.results_dir, timestamp)
    
    # 打印结果表格
    logger.info("\n" + "="*100)
    logger.info("固定缺失实验结果摘要")
    logger.info("="*100)
    logger.info(f"{'组合':<25} {'方法':<25} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'改进率':<12}")
    logger.info("-"*100)
    
    for i in range(len(results['missing_combinations'])):
        comb = results['missing_combinations'][i]
        logger.info(f"{comb:<25} {'缺失指示器':<25} {results['indicator_mae'][i]:<10.4f} "
                    f"{results['indicator_rmse'][i]:<10.4f} {results['indicator_r2'][i]:<10.4f} {'-':<12}")
        logger.info(f"{'':<25} {'缩减模型':<25} {results['reduced_mae'][i]:<10.4f} "
                    f"{results['reduced_rmse'][i]:<10.4f} {results['reduced_r2'][i]:<10.4f} "
                    f"{results['improvement'][i]:<10.1f}%")
        logger.info("-"*100)
    
    logger.info("\n" + "="*100)
    logger.info("实验成功完成！")
    logger.info("="*100)
    logger.info(f"结果时间戳: {timestamp}")
    logger.info(f"结果保存目录: {os.path.abspath(args.results_dir)}")
    logger.info(f"详细结果文件: {csv_path}")
    logger.info(f"核心改进: 结合了最佳实践 - 安全模型加载、高效缺失生成、全面随机种子设置")
    logger.info("="*100)

if __name__ == "__main__":
    main()