from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import datetime
import glob

# =========================
# 1. 命令行参数与配置
# =========================
parser = argparse.ArgumentParser(description='SOH estimation with missing data handling')
parser.add_argument('--missing_rates', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                   help='要测试的缺失率列表 (0.1-0.9)')
parser.add_argument('--max_missing_rate', type=float, default=0.9,
                   help='动态训练的最大缺失率')
parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
parser.add_argument('--data_dir', type=str, default='./data/XJTU data',
                   help='数据集根目录')
parser.add_argument('--batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'],
                   help='电池批次')
parser.add_argument('--seed', type=int, default=42, help='用于可复现性的随机种子')
parser.add_argument('--batch_size', type=int, default=32, help='训练批次大小')
parser.add_argument('--results_dir', type=str, default='results', help='结果保存目录')
args = parser.parse_args()

# 设置随机种子保证可复现性
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# 确保结果目录存在
os.makedirs(args.results_dir, exist_ok=True)

# 生成全局唯一标识符（时间戳）
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"实验时间戳: {timestamp}")

# =========================
# 2. 数据清洗函数
# =========================
def clean_data(df, feature_cols, target_col):
    """清洗数据：处理inf值、NaN值和异常值"""
    print("正在清洗数据: 移除inf值和异常值...")
    
    # 1. 替换无穷大值为NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 2. 删除包含NaN的行
    original_shape = df.shape
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(f"  已移除 {original_shape[0] - df.shape[0]} 行包含NaN/inf值的数据")
    
    # 3. 应用3-sigma原则移除异常值
    out_index = []
    for col in feature_cols + [target_col]:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            # 避免除以零
            if std > 0:
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                out_index.extend(outliers.tolist())
    
    # 去重并删除异常值
    out_index = list(set(out_index))
    if out_index:
        print(f"  正在移除 {len(out_index)} 行极端异常值")
        df = df.drop(out_index)
        df = df.reset_index(drop=True)
    
    print(f"  清洗后数据形状: {df.shape}")
    return df

# =========================
# 3. 训练集扩充函数
# =========================
def expand_training_set(X_train, y_train, missing_rates):
    """
    扩充训练集以适应带缺失指示器的神经网络
    根据给定的缺失率列表生成多个缺失副本
    """
    print(f"正在扩充训练集，缺失率范围: {missing_rates}")
    
    expanded_X = []
    expanded_y = []
    
    # 为每个缺失率生成缺失副本（只生成缺失指示器格式的副本）
    for mr in missing_rates:
        print(f"  生成缺失率为 {mr*100:.0f}% 的副本...")
        
        # 生成随机缺失掩码
        mask = np.random.binomial(1, 1-mr, size=X_train.shape)
        
        # 应用缺失（将缺失位置设为0）
        X_with_missing = np.where(mask == 1, X_train, 0)
        
        # 生成缺失指示器向量（缺失位置为1，否则为0）
        missing_indicators = 1 - mask
        
        # 将原始特征与缺失指示器拼接
        X_combined = np.concatenate([X_with_missing, missing_indicators], axis=1)
        
        expanded_X.append(X_combined)
        expanded_y.append(y_train)  # 标签保持不变
        
        print(f"    缺失副本形状: {X_combined.shape}")
    
    # 合并所有数据
    X_expanded = np.vstack(expanded_X)
    y_expanded = np.concatenate(expanded_y)
    
    print(f"扩展训练集完成 - 特征: {X_expanded.shape}, 标签: {y_expanded.shape}")
    return X_expanded, y_expanded

# =========================
# 4. Dataset 定义
# =========================
class BatteryDatasetFixedMissing(Dataset):
    """固定缺失率的数据集，用于验证和测试"""
    def __init__(self, X, y, missing_rate=0.0, include_missing_indicators=True, missing_mask=None):
        self.X = X
        self.y = y
        self.missing_rate = missing_rate
        self.include_missing_indicators = include_missing_indicators
        
        # 如果没有提供预定义的缺失掩码，则生成新的
        if missing_mask is None:
            self.missing_mask = np.random.binomial(1, 1-missing_rate, size=X.shape)
        else:
            self.missing_mask = missing_mask
        
        # 创建缺失指示器 (1表示缺失, 0表示存在)
        self.missing_indicators = 1 - self.missing_mask
        
        # 应用缺失 (将缺失位置设为0)
        self.X_with_missing = np.where(self.missing_mask == 1, self.X, 0)
        
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

class BatteryDatasetDynamicMissing(Dataset):
    """动态缺失率的数据集，用于训练"""
    def __init__(self, X, y, max_missing_rate=0.9, include_missing_indicators=True):
        self.X = X
        self.y = y
        self.max_missing_rate = max_missing_rate
        self.include_missing_indicators = include_missing_indicators

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        # 随机生成当前样本的缺失率
        mr = np.random.uniform(0.0, self.max_missing_rate)
        mask = np.random.binomial(1, 1 - mr, size=x.shape)

        x_miss = np.where(mask == 1, x, 0.0)
        indicator = 1 - mask

        x_tensor = torch.tensor(x_miss, dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
        
        if self.include_missing_indicators:
            m_tensor = torch.tensor(indicator, dtype=torch.float32)
            return torch.cat([x_tensor, m_tensor]), y_tensor
        else:
            return x_tensor, y_tensor

class MeanFillDataset(Dataset):
    """均值填充方法的数据集"""
    def __init__(self, X, y, missing_rate=0.0, missing_mask=None):
        self.X = X.copy()
        self.y = y
        self.missing_rate = missing_rate
        
        # 生成或使用提供的缺失掩码
        if missing_mask is None:
            missing_mask = np.random.binomial(1, 1-missing_rate, size=X.shape)
        
        # 应用缺失
        features_with_missing = np.where(missing_mask == 1, self.X, np.nan)
        
        # 均值填充
        feature_means = np.nanmean(features_with_missing, axis=0)
        # 处理全为NaN的列（虽然理论上不会发生）
        feature_means = np.nan_to_num(feature_means, nan=0.0)
        self.features_filled = np.where(np.isnan(features_with_missing), feature_means, features_with_missing)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features_filled[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# =========================
# 5. 模型定义
# =========================
class SOHNetwork(nn.Module):
    """SOH估计神经网络，支持不同输入大小"""
    def __init__(self, input_size):
        super(SOHNetwork, self).__init__()
        
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
        return self.net(x).squeeze()

# =========================
# 6. 训练函数（包含早停机制）
# =========================
def train_model(model, train_loader, val_loader, epochs, device, save_path=None, patience=15):
    """
    训练模型，包含早停机制
    
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
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"在第 {epoch+1} 轮提前停止训练")
                break
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"轮次 {epoch+1}/{epochs}, 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
    
    # 加载最佳模型
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    
    return model, train_losses, val_losses

# =========================
# 7. 评估函数
# =========================
def evaluate_model(model, loader, device):
    """评估模型性能"""
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
    
    # 计算评估指标
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    r2 = 1 - (np.sum((predictions - targets) ** 2) / 
              np.sum((targets - np.mean(targets)) ** 2)) if np.var(targets) > 1e-8 else 0.0
    
    return mae, rmse, r2, predictions, targets

# =========================
# 8. 可视化和结果保存函数
# =========================
def plot_training_results(train_losses, val_losses, missing_rate, method_name, save_dir):
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

def plot_comparison_results(true_values, indicator_preds, fill_preds, baseline_preds, missing_rate, save_dir):
    """绘制不同方法的预测结果对比"""
    plt.figure(figsize=(12, 8))
    
    # 预测结果对比
    plt.plot(true_values, 'k-', label='True SOH', linewidth=2.5)
    plt.plot(indicator_preds, 'r--', label='Missing Indicators', linewidth=1.5, alpha=0.9)
    plt.plot(fill_preds, 'g-.', label='Mean Filling', linewidth=1.5, alpha=0.9)
    plt.plot(baseline_preds, 'b:', label='Baseline (Complete Data)', linewidth=1.5, alpha=0.9)
    
    plt.title(f'SOH Estimation Comparison (Missing Rate: {missing_rate*100:.0f}%)')
    plt.xlabel('Sample Index')
    plt.ylabel('SOH')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prediction_comparison_mr_{missing_rate*100:.0f}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_distribution(indicator_errors, fill_errors, missing_rate, save_dir):
    """绘制误差分布"""
    plt.figure(figsize=(10, 6))
    plt.boxplot([indicator_errors, fill_errors], tick_labels=['Missing Indicators', 'Mean Filling'])
    plt.title(f'Absolute Error Distribution (Missing Rate: {missing_rate*100:.0f}%)')
    plt.ylabel('Absolute Error')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/error_distribution_mr_{missing_rate*100:.0f}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_comprehensive_results(results, save_dir):
    """生成综合比较图表"""
    missing_rates = results['missing_rates']
    
    plt.figure(figsize=(15, 12))
    
    # MAE随缺失率变化趋势
    plt.subplot(2, 2, 1)
    plt.plot(missing_rates, results['baseline_mae'], 'k-', label='Baseline (Complete Data)', linewidth=3)
    plt.plot(missing_rates, results['indicator_mae'], 'r-o', label='Missing Indicators', linewidth=2)
    plt.plot(missing_rates, results['fill_mae'], 'g--s', label='Mean Filling', linewidth=2)
    plt.xlabel('Missing Rate')
    plt.ylabel('MAE')
    plt.title('MAE vs Missing Rate')
    plt.legend()
    plt.grid(True)
    
    # RMSE随缺失率变化趋势
    plt.subplot(2, 2, 2)
    plt.plot(missing_rates, results['baseline_rmse'], 'k-', label='Baseline (Complete Data)', linewidth=3)
    plt.plot(missing_rates, results['indicator_rmse'], 'r-o', label='Missing Indicators', linewidth=2)
    plt.plot(missing_rates, results['fill_rmse'], 'g--s', label='Mean Filling', linewidth=2)
    plt.xlabel('Missing Rate')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Missing Rate')
    plt.legend()
    plt.grid(True)
    
    # R²随缺失率变化趋势
    plt.subplot(2, 2, 3)
    plt.plot(missing_rates, results['baseline_r2'], 'k-', label='Baseline (Complete Data)', linewidth=3)
    plt.plot(missing_rates, results['indicator_r2'], 'r-o', label='Missing Indicators', linewidth=2)
    plt.plot(missing_rates, results['fill_r2'], 'g--s', label='Mean Filling', linewidth=2)
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

def save_results_to_csv(results, save_dir):
    """将结果保存到CSV文件"""
    results_df = pd.DataFrame({
        'Missing_Rate': results['missing_rates'],
        'Baseline_MAE': results['baseline_mae'],
        'Baseline_RMSE': results['baseline_rmse'],
        'Baseline_R2': results['baseline_r2'],
        'Indicator_MAE': results['indicator_mae'],
        'Indicator_RMSE': results['indicator_rmse'],  # 修复：将 'indicator_rm2' 改为 'indicator_rmse'
        'Indicator_R2': results['indicator_r2'],
        'Fill_MAE': results['fill_mae'],
        'Fill_RMSE': results['fill_rmse'],
        'Fill_R2': results['fill_r2'],
        'Improvement_Percent': results['improvement']
    })
    csv_path = f'{save_dir}/experiment_results_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"详细结果已保存至 {csv_path}")
    return csv_path

# =========================
# 9. 主程序
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 定义特征和目标列
    feature_cols = ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness', 
                   'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
                   'current mean', 'current std', 'current kurtosis', 'current skewness',
                   'CV Q', 'CV charge time', 'current slope', 'current entropy']
    target_col = 'capacity'
    
    # 获取数据文件列表
    data_dir = Path(args.data_dir)
    print(f"正在从目录加载数据: {data_dir}, 批次: {args.batch}")
    
    # 获取该批次下所有电池文件
    pattern = f"{args.batch}_battery-*.csv"
    all_files = list(data_dir.glob(pattern))
    if not all_files:
        raise ValueError(f"在目录 {data_dir} 下未找到匹配 {pattern} 的文件")
    all_files.sort()  # 确保顺序
    print(f"找到 {len(all_files)} 个电池文件: {[f.name for f in all_files]}")
    
    # 按电池ID划分训练/测试集（4号和8号电池作为测试集）
    train_files = [f for f in all_files if not ('4' in f.name or '8' in f.name)]
    test_files = [f for f in all_files if '4' in f.name or '8' in f.name]
    print(f"训练电池文件 ({len(train_files)}): {[f.name for f in train_files]}")
    print(f"测试电池文件 ({len(test_files)}): {[f.name for f in test_files]}")
    
    if not train_files:
        raise ValueError("没有找到训练电池文件")
    if not test_files:
        print("警告: 没有找到测试电池文件（包含'4'或'8'），将使用20%的训练电池作为测试集")
        # 从训练电池中随机选择20%作为测试集
        np.random.shuffle(train_files)
        split_idx = int(len(train_files) * 0.8)
        test_files = train_files[split_idx:]
        train_files = train_files[:split_idx]
    
    # 加载并处理训练电池数据
    X_train_list = []
    y_train_list = []
    train_battery_info = []
    
    for file_path in train_files:
        df = pd.read_csv(file_path)
        print(f"\n加载训练电池: {file_path.name}, 原始形状: {df.shape}")
        df = clean_data(df, feature_cols, target_col)
        if df.empty:
            print(f"  警告: 电池 {file_path.name} 在清洗后无数据，跳过")
            continue
            
        # 计算该电池的SOH（基于初始容量）
        initial_capacity = df[target_col].iloc[0]
        y_soh = df[target_col].values / initial_capacity
        X = df[feature_cols].values
        
        X_train_list.append(X)
        y_train_list.append(y_soh)
        train_battery_info.append((file_path.name, initial_capacity, len(X)))
        print(f"  电池 {file_path.name} 保留 {X.shape[0]} 个样本，初始容量: {initial_capacity:.4f}, SOH范围: [{y_soh.min():.4f}, {y_soh.max():.4f}]")
    
    # 合并训练电池数据
    X_train_all = np.vstack(X_train_list)
    y_train_all = np.concatenate(y_train_list)
    print(f"\n训练电池合并后形状 - 特征: {X_train_all.shape}, SOH: {y_train_all.shape}")
    
    # 加载并处理测试电池数据
    X_test_list = []
    y_test_list = []
    test_battery_info = []
    
    for file_path in test_files:
        df = pd.read_csv(file_path)
        print(f"\n加载测试电池: {file_path.name}, 原始形状: {df.shape}")
        df = clean_data(df, feature_cols, target_col)
        if df.empty:
            print(f"  警告: 电池 {file_path.name} 在清洗后无数据，跳过")
            continue
            
        initial_capacity = df[target_col].iloc[0]
        y_soh = df[target_col].values / initial_capacity
        X = df[feature_cols].values
        
        X_test_list.append(X)
        y_test_list.append(y_soh)
        test_battery_info.append((file_path.name, initial_capacity, len(X)))
        print(f"  电池 {file_path.name} 保留 {X.shape[0]} 个样本，初始容量: {initial_capacity:.4f}, SOH范围: [{y_soh.min():.4f}, {y_soh.max():.4f}]")
    
    X_test_all = np.vstack(X_test_list)
    y_test_all = np.concatenate(y_test_list)
    print(f"\n测试电池合并后形状 - 特征: {X_test_all.shape}, SOH: {y_test_all.shape}")
    
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
    
    print(f"数据集划分完成 - 训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")
    
    # 存储所有结果
    results = {
        'missing_rates': args.missing_rates,
        'baseline_mae': [], 'baseline_rmse': [], 'baseline_r2': [],
        'indicator_mae': [], 'indicator_rmse': [], 'indicator_r2': [],
        'fill_mae': [], 'fill_rmse': [], 'fill_r2': [],
        'improvement': []
    }
    
    print("\n" + "="*60)
    print("正在训练基线模型（完整数据）")
    print("="*60)
    
    # 完整数据基线 - 使用固定的缺失掩码（全1，没有缺失）
    full_train_mask = np.ones_like(X_train)
    full_val_mask = np.ones_like(X_val)
    full_test_mask = np.ones_like(X_test)
    
    # 修正：使用include_missing_indicators=False，因为这是"普通神经网络"
    full_train_dataset = BatteryDatasetFixedMissing(
        X_train, y_train, missing_rate=0.0, 
        include_missing_indicators=False, missing_mask=full_train_mask
    )
    full_val_dataset = BatteryDatasetFixedMissing(
        X_val, y_val, missing_rate=0.0, 
        include_missing_indicators=False, missing_mask=full_val_mask
    )
    full_test_dataset = BatteryDatasetFixedMissing(
        X_test, y_test, missing_rate=0.0, 
        include_missing_indicators=False, missing_mask=full_test_mask
    )
    
    train_loader = DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(full_val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(full_test_dataset, batch_size=args.batch_size)
    
    # 修正：基线模型使用input_size=16（普通神经网络，无缺失指示器）
    model_baseline = SOHNetwork(input_size=16).to(device)
    baseline_model_path = f'{args.results_dir}/best_model_baseline_{timestamp}.pth'
    model_baseline, baseline_train_losses, baseline_val_losses = train_model(
        model_baseline, train_loader, val_loader, args.epochs, device, baseline_model_path
    )
    
    # 评估完整数据基线
    baseline_mae, baseline_rmse, baseline_r2, baseline_preds, _ = evaluate_model(
        model_baseline, test_loader, device
    )
    
    # 为所有缺失率保存相同的基线结果
    for _ in args.missing_rates:
        results['baseline_mae'].append(baseline_mae)
        results['baseline_rmse'].append(baseline_rmse)
        results['baseline_r2'].append(baseline_r2)
    
    print(f"基线模型（完整数据）- MAE: {baseline_mae:.4f}, RMSE: {baseline_rmse:.4f}, R²: {baseline_r2:.4f}")
    plot_training_results(baseline_train_losses, baseline_val_losses, 0.0, "baseline", args.results_dir)
    
    print("\n" + "="*60)
    print("正在测试不同缺失率")
    print("="*60)
    
    # 为每个缺失率生成固定的测试掩码，确保可比性
    test_masks = {}
    for mr in args.missing_rates:
        mask = np.random.binomial(1, 1-mr, size=X_test.shape)
        test_masks[mr] = mask
    
    # 对每个缺失率进行测试
    for i, missing_rate in enumerate(args.missing_rates):
        print(f"\n" + "-"*60)
        print(f"正在测试缺失率: {missing_rate*100:.0f}%")
        print("-"*60)
        
        # 为当前缺失率设置随机种子，确保可复现性
        torch.manual_seed(args.seed + i)
        np.random.seed(args.seed + i)
        
        # 生成训练和验证的缺失掩码
        train_mask = np.random.binomial(1, 1-missing_rate, size=X_train.shape)
        val_mask = np.random.binomial(1, 1-missing_rate, size=X_val.shape)
        test_mask = test_masks[missing_rate]  # 使用预定义的测试掩码
        
        # ===== 缺失指示器方法 =====
        print(f"\n--- 缺失指示器方法 (缺失率: {missing_rate*100:.0f}%) ---")
        
        # 使用训练集扩充功能
        X_train_expanded, y_train_expanded = expand_training_set(X_train, y_train, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        # 修复：直接使用扩充后的训练集，不需要再次添加缺失指示器
        # 因为X_train_expanded已经包含了缺失指示器，维度是32
        # 所以使用BatteryDatasetFixedMissing类，不添加额外的缺失指示器
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
        
        # 固定缺失率验证
        val_dataset = BatteryDatasetFixedMissing(
            X_val, y_val, missing_rate=missing_rate,
            include_missing_indicators=True, missing_mask=val_mask
        )
        
        # 固定缺失率测试
        test_dataset = BatteryDatasetFixedMissing(
            X_test, y_test, missing_rate=missing_rate,
            include_missing_indicators=True, missing_mask=test_mask
        )
        
        train_loader = DataLoader(expanded_train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # 训练缺失指示器模型
        model_indicator = SOHNetwork(input_size=32).to(device)
        indicator_model_path = f'{args.results_dir}/best_model_indicator_mr_{missing_rate*100:.0f}_{timestamp}.pth'
        model_indicator, train_losses, val_losses = train_model(
            model_indicator, train_loader, val_loader, args.epochs, device, indicator_model_path
        )
        
        # 评估缺失指示器方法
        indicator_mae, indicator_rmse, indicator_r2, indicator_preds, _ = evaluate_model(
            model_indicator, test_loader, device
        )
        
        results['indicator_mae'].append(indicator_mae)
        results['indicator_rmse'].append(indicator_rmse)
        results['indicator_r2'].append(indicator_r2)
        
        print(f"缺失指示器方法 - MAE: {indicator_mae:.4f}, RMSE: {indicator_rmse:.4f}, R²: {indicator_r2:.4f}")
        
        # 绘制训练曲线
        plot_training_results(train_losses, val_losses, missing_rate, "indicators", args.results_dir)
        
        # ===== 均值填充方法 =====
        print(f"\n--- 均值填充方法 (缺失率: {missing_rate*100:.0f}%) ---")
        
        fill_train_dataset = MeanFillDataset(X_train, y_train, missing_rate=missing_rate, missing_mask=train_mask)
        fill_val_dataset = MeanFillDataset(X_val, y_val, missing_rate=missing_rate, missing_mask=val_mask)
        fill_test_dataset = MeanFillDataset(X_test, y_test, missing_rate=missing_rate, missing_mask=test_mask)
        
        train_loader = DataLoader(fill_train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(fill_val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(fill_test_dataset, batch_size=args.batch_size)
        
        # 均值填充方法只有16个输入特征
        model_fill = SOHNetwork(input_size=16).to(device)
        fill_model_path = f'{args.results_dir}/best_model_fill_mr_{missing_rate*100:.0f}_{timestamp}.pth'
        model_fill, fill_train_losses, fill_val_losses = train_model(
            model_fill, train_loader, val_loader, args.epochs, device, fill_model_path
        )
        
        # 评估均值填充方法
        fill_mae, fill_rmse, fill_r2, fill_preds, _ = evaluate_model(model_fill, test_loader, device)
        
        results['fill_mae'].append(fill_mae)
        results['fill_rmse'].append(fill_rmse)
        results['fill_r2'].append(fill_r2)
        
        # 计算改进百分比
        improvement = (fill_mae - indicator_mae) / fill_mae * 100 if fill_mae > 0 else 0.0
        results['improvement'].append(improvement)
        
        print(f"均值填充方法 - MAE: {fill_mae:.4f}, RMSE: {fill_rmse:.4f}, R²: {fill_r2:.4f}")
        print(f"缺失指示器方法相对于均值填充方法的改进: {improvement:.1f}%")
        
        # 绘制训练曲线
        plot_training_results(fill_train_losses, fill_val_losses, missing_rate, "mean_fill", args.results_dir)
        
        # ===== 保存详细对比图 =====
        print("\n--- 正在保存详细对比图表 ---")
        
        # 预测结果对比
        plot_comparison_results(y_test, indicator_preds, fill_preds, baseline_preds, missing_rate, args.results_dir)
        
        # 误差分布
        indicator_errors = np.abs(indicator_preds - y_test)
        fill_errors = np.abs(fill_preds - y_test)
        plot_error_distribution(indicator_errors, fill_errors, missing_rate, args.results_dir)
        
        print(f"缺失率 {missing_rate*100:.0f}% 的详细图表已保存至 {args.results_dir}")
    
    # ===== 生成综合比较图 =====
    print("\n" + "="*60)
    print("正在生成综合对比图表")
    print("="*60)
    plot_comprehensive_results(results, args.results_dir)
    
    # ===== 保存结果到CSV =====
    print("\n" + "="*60)
    print("正在将结果保存至CSV文件")
    print("="*60)
    csv_path = save_results_to_csv(results, args.results_dir)
    
    # ===== 打印结果表格 =====
    print("\n" + "="*100)
    print("实验结果摘要")
    print("="*100)
    print(f"{'缺失率':<12} {'方法':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'改进率':<12}")
    print("-"*100)
    
    for i, mr in enumerate(results['missing_rates']):
        print(f"{mr*100:>10.0f}%  {'基线模型':<20} {results['baseline_mae'][i]:<10.4f} {results['baseline_rmse'][i]:<10.4f} {results['baseline_r2'][i]:<10.4f} {'-':<12}")
        print(f"{'':<12} {'缺失指示器':<20} {results['indicator_mae'][i]:<10.4f} {results['indicator_rmse'][i]:<10.4f} {results['indicator_r2'][i]:<10.4f} {'-':<12}")
        print(f"{'':<12} {'均值填充':<20} {results['fill_mae'][i]:<10.4f} {results['fill_rmse'][i]:<10.4f} {results['fill_r2'][i]:<10.4f} {results['improvement'][i]:<10.1f}%")
        print("-"*100)
    
    print("\n" + "="*100)
    print("实验成功完成！")
    print("="*100)
    print(f"结果时间戳: {timestamp}")
    print(f"结果保存目录: {os.path.abspath(args.results_dir)}")
    print(f"详细结果文件: {csv_path}")
    print("="*100)

if __name__ == "__main__":
    main()