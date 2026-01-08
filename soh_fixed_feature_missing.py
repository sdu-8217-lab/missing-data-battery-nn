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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =========================
# 1. 常量定义 (避免魔法数字)
# =========================
NUM_FEATURES = 16  # 特征数量常量

# =========================
# 2. 日志配置 (避免重复输出)
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================
# 3. 命令行参数与配置
# =========================
parser = argparse.ArgumentParser(description='Fixed Missing Pattern Experiment for SOH Estimation')
parser.add_argument('--features', type=int, nargs='+', default=list(range(NUM_FEATURES)),
                   help='要测试的特征索引列表 (0-%d, 默认全部%d个特征)' % (NUM_FEATURES-1, NUM_FEATURES))
parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
parser.add_argument('--data_dir', type=str, default='./data/XJTU data',
                   help='数据集根目录')
parser.add_argument('--batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'],
                   help='电池批次')
parser.add_argument('--seed', type=int, default=8217, help='用于可复现性的随机种子')
parser.add_argument('--batch_size', type=int, default=32, help='训练批次大小')
parser.add_argument('--results_dir', type=str, default='fixed_missing_results',
                   help='结果保存目录')
parser.add_argument('--pretrained_model', type=str, default=None,
                   help='预训练通用模型路径 (可选)')
parser.add_argument('--training_missing_rates', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                   help='训练缺失指示器模型时使用的缺失率列表 (范围: [0.0, 1.0])')
args = parser.parse_args()

# 确保结果目录存在
os.makedirs(args.results_dir, exist_ok=True)

# 生成全局唯一标识符
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logger.info(f"固定缺失实验启动 (时间戳: {timestamp})")

# =========================
# 4. 数据处理函数 (修复文档字符串)
# =========================
def clean_data(df, feature_cols, target_col):
    """清洗数据：处理inf值、NaN值和异常值
    
    Args:
        df: 原始数据框
        feature_cols: 特征列名列表
        target_col: 目标列名
    
    Returns:
        pd.DataFrame: 清洗后的数据框
    """
    logger.info("正在清洗数据: 移除inf值和异常值...")
    
    # 1. 替换无穷大值为NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 2. 删除包含NaN的行
    original_shape = df.shape
    df = df.dropna()
    df = df.reset_index(drop=True)
    logger.info(f"  已移除 {original_shape[0] - df.shape[0]} 行包含NaN/inf值的数据")
    
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
        logger.info(f"  正在移除 {len(out_index)} 行极端异常值")
        df = df.drop(out_index)
        df = df.reset_index(drop=True)
    
    logger.info(f"  清洗后数据形状: {df.shape}")
    return df

def expand_training_set(X_train, y_train, missing_rates):
    """扩充训练集以适应带缺失指示器的神经网络
    
    Args:
        X_train: 训练特征 (n_samples, n_features)
        y_train: 训练标签 (n_samples,)
        missing_rates: 缺失率列表 (范围: [0.0, 1.0])
    
    Returns:
        tuple: (扩展后的特征, 扩展后的标签)
    """
    logger.info(f"正在扩充训练集，缺失率范围: {missing_rates}")
    
    expanded_X = []
    expanded_y = []
    
    # 为每个缺失率生成缺失副本
    for mr in missing_rates:
        logger.info(f"  生成缺失率为 {mr*100:.0f}% 的副本...")
        
        # 生成随机缺失掩码 (1表示保留，0表示缺失)
        mask = np.random.binomial(1, 1-mr, size=X_train.shape)
        
        # 应用缺失（将缺失位置设为0，对应标准化后的均值）
        X_with_missing = np.where(mask == 1, X_train, 0)
        
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
# 5. 固定缺失专用数据集类
# =========================
class BatteryDatasetFixedMissing(Dataset):
    """固定缺失率的数据集，用于验证和测试
    
    Args:
        X: 原始特征 (n_samples, n_features)
        y: 目标值 (n_samples,)
        fixed_missing_mask: 固定缺失掩码 (n_features,) - 1表示存在，0表示缺失
    """
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

# =========================
# 6. 模型定义 (修复网络结构)
# =========================
class SOHNetwork(nn.Module):
    """SOH估计神经网络，支持不同输入大小
    
    Args:
        input_size: 输入特征数量
    """
    def __init__(self, input_size):
        super(SOHNetwork, self).__init__()
        
        # 根据输入大小动态设计网络结构
        if input_size == NUM_FEATURES - 1:  # 15维输入 (砍掉1个特征)
            self.net = nn.Sequential(
                nn.Linear(NUM_FEATURES - 1, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        elif input_size == 2 * NUM_FEATURES:  # 缺失指示器方法 (16原始特征 + 16缺失指示器)
            self.net = nn.Sequential(
                nn.Linear(2 * NUM_FEATURES, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        else:
            # 通用结构
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
# 7. 训练函数 (复用)
# =========================
def train_model(model, train_loader, val_loader, epochs, device, save_path=None, patience=15):
    """训练模型，包含早停机制
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 最大训练轮数
        device: 训练设备
        save_path: 模型保存路径
        patience: 早停耐心值
    
    Returns:
        tuple: (训练好的模型, 训练损失历史, 验证损失历史)
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
                logger.info(f"在第 {epoch+1} 轮提前停止训练")
                break
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            logger.info(f"轮次 {epoch+1}/{epochs}, 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
    
    # 加载最佳模型
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    
    return model, train_losses, val_losses

# =========================
# 8. 评估函数 (复用)
# =========================
def evaluate_model(model, loader, device):
    """评估模型性能
    
    Args:
        model: 模型
        loader: 数据加载器
        device: 设备
    
    Returns:
        tuple: (MAE, RMSE, R², 预测值, 真实值)
    """
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
# 9. 可视化和结果保存函数 (优化)
# =========================
def plot_comparison_results(true_values, indicator_preds, reduced_preds, feature_index, save_dir):
    """绘制固定缺失特征的预测结果对比
    
    Args:
        true_values: 真实SOH值
        indicator_preds: 缺失指示器模型预测
        reduced_preds: 15维模型预测
        feature_index: 缺失特征索引
        save_dir: 保存目录
    """
    plt.figure(figsize=(12, 8))
    
    # 预测结果对比
    plt.plot(true_values, 'k-', label='True SOH', linewidth=2.5)
    plt.plot(indicator_preds, 'r--', label='Missing Indicators', linewidth=1.5, alpha=0.9)
    plt.plot(reduced_preds, 'b-.', label='Reduced Model (15 features)', linewidth=1.5, alpha=0.9)
    
    plt.title(f'SOH Estimation Comparison (Fixed Missing: Feature {feature_index})')
    plt.xlabel('Sample Index')
    plt.ylabel('SOH')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_{feature_index}_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results_to_csv(results, save_dir):
    """将结果保存到CSV文件
    
    Args:
        results: 结果字典
        save_dir: 保存目录
    
    Returns:
        str: CSV文件路径
    """
    results_df = pd.DataFrame({
        'Feature_Index': results['feature_indices'],
        'Indicator_MAE': results['indicator_mae'],
        'Indicator_RMSE': results['indicator_rmse'],
        'Indicator_R2': results['indicator_r2'],
        'Reduced_MAE': results['reduced_mae'],
        'Reduced_RMSE': results['reduced_rmse'],
        'Reduced_R2': results['reduced_r2'],
        'Improvement_Percent': results['improvement']
    })
    csv_path = f'{save_dir}/feature_comparison_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info(f"详细结果已保存至 {csv_path}")
    return csv_path

def plot_overall_comparison(results, save_dir, timestamp):
    """绘制所有特征的MAE对比总图
    
    Args:
        results: 结果字典
        save_dir: 保存目录
        timestamp: 时间戳
    """
    plt.figure(figsize=(14, 8))
    
    # 提取数据
    feature_indices = results['feature_indices']
    indicator_mae = results['indicator_mae']
    reduced_mae = results['reduced_mae']
    
    # 创建柱状图
    x = np.arange(len(feature_indices))
    width = 0.35
    
    # 绘制柱状图
    plt.bar(x - width/2, indicator_mae, width, label='Missing Indicators', color='#1f77b4', alpha=0.8)
    plt.bar(x + width/2, reduced_mae, width, label='Reduced Model (15 features)', color='#ff7f0e', alpha=0.8)
    
    # 添加标签和标题
    plt.xlabel('Feature Index', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('MAE Comparison Across All Features (Fixed Missing Pattern)', fontsize=14, fontweight='bold')
    plt.xticks(x, [str(i) for i in feature_indices], fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for i, (ind_mae, red_mae) in enumerate(zip(indicator_mae, reduced_mae)):
        plt.text(i - width/2, ind_mae + 0.0005, f'{ind_mae:.4f}', ha='center', fontsize=8)
        plt.text(i + width/2, red_mae + 0.0005, f'{red_mae:.4f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/overall_mae_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"总图已保存至 {save_dir}/overall_mae_comparison_{timestamp}.png")

# =========================
# 10. 主程序 (核心逻辑)
# =========================
def main():
    # =========================
    # 关键修复：设置随机种子确保所有随机操作可复现
    # =========================
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"设置随机种子: {args.seed}")
    
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
    
    # 按电池ID划分训练/测试集（4号和8号电池作为测试集）
    train_files = [f for f in all_files if not ('4' in f.name or '8' in f.name)]
    test_files = [f for f in all_files if '4' in f.name or '8' in f.name]
    logger.info(f"训练电池文件 ({len(train_files)}): {[f.name for f in train_files]}")
    logger.info(f"测试电池文件 ({len(test_files)}): {[f.name for f in test_files]}")
    
    if not train_files:
        raise ValueError("没有找到训练电池文件")
    if not test_files:
        logger.warning("警告: 没有找到测试电池文件（包含'4'或'8'），将使用20%的训练电池作为测试集")
        # 从训练电池中随机选择20%作为测试集（使用已设置的种子确保可复现）
        np.random.shuffle(train_files)
        split_idx = int(len(train_files) * 0.8)
        test_files = train_files[split_idx:]
        train_files = train_files[:split_idx]
    
    # 加载并处理训练电池数据
    X_train_list = []
    y_train_list = []
    
    for file_path in train_files:
        df = pd.read_csv(file_path)
        logger.info(f"\n加载训练电池: {file_path.name}, 原始形状: {df.shape}")
        df = clean_data(df, feature_cols, target_col)
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
    X_train_all = np.vstack(X_train_list)
    y_train_all = np.concatenate(y_train_list)
    logger.info(f"\n训练电池合并后形状 - 特征: {X_train_all.shape}, SOH: {y_train_all.shape}")
    
    # 加载并处理测试电池数据
    X_test_list = []
    y_test_list = []
    
    for file_path in test_files:
        df = pd.read_csv(file_path)
        logger.info(f"\n加载测试电池: {file_path.name}, 原始形状: {df.shape}")
        df = clean_data(df, feature_cols, target_col)
        if df.empty:
            logger.warning(f"  警告: 电池 {file_path.name} 在清洗后无数据，跳过")
            continue
            
        initial_capacity = df[target_col].iloc[0]
        y_soh = df[target_col].values / initial_capacity
        X = df[feature_cols].values
        
        X_test_list.append(X)
        y_test_list.append(y_soh)
    
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
        'feature_indices': [],
        'indicator_mae': [], 'indicator_rmse': [], 'indicator_r2': [],
        'reduced_mae': [], 'reduced_rmse': [], 'reduced_r2': [],
        'improvement': []
    }
    
    # ===== 1. 训练通用缺失指示器模型 (32维输入) =====
    logger.info("\n" + "="*60)
    logger.info("正在训练通用缺失指示器模型 (32维输入)")
    logger.info("="*60)
    
    # 扩充训练集
    X_train_expanded, y_train_expanded = expand_training_set(X_train, y_train, args.training_missing_rates)
    
    # 创建扩展训练集的Dataset
    class BatteryDatasetExpanded(Dataset):
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
    
    # 创建验证集 (使用随机缺失掩码 - 与测试集缺失模式不同，但这是合理的实验设计)
    feature_missing_vector = np.random.binomial(1, 0.5, size=NUM_FEATURES)
    val_dataset = BatteryDatasetFixedMissing(X_val, y_val, fixed_missing_mask=feature_missing_vector)

    train_loader = DataLoader(expanded_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # 加载预训练模型或重新训练
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        logger.info(f"加载预训练模型: {args.pretrained_model}")
        model_indicator = SOHNetwork(input_size=2 * NUM_FEATURES).to(device)
        model_indicator.load_state_dict(torch.load(args.pretrained_model))
    else:
        logger.info("未提供预训练模型，重新训练通用模型")
        model_indicator = SOHNetwork(input_size=2 * NUM_FEATURES).to(device)
        model_path = f'{args.results_dir}/best_model_indicator_{timestamp}.pth'
        model_indicator, _, _ = train_model(
            model_indicator, train_loader, val_loader, args.epochs, device, model_path
        )
    
    # ===== 2. 为每个特征测试固定缺失模式 =====
    for feature_index in args.features:
        logger.info("\n" + "="*60)
        logger.info(f"测试特征 {feature_index} 的固定缺失模式")
        logger.info("="*60)
        
        # 创建固定缺失掩码 (仅当前特征缺失)
        fixed_missing_mask = np.ones(NUM_FEATURES, dtype=int)
        fixed_missing_mask[feature_index] = 0  # 设置当前特征为缺失
        
        # 创建测试集 (固定缺失)
        test_dataset_indicator = BatteryDatasetFixedMissing(
            X_test, y_test, fixed_missing_mask=fixed_missing_mask
        )
        test_loader_indicator = DataLoader(test_dataset_indicator, batch_size=args.batch_size)
        
        # ===== 2.1 评估缺失指示器通用模型 =====
        indicator_mae, indicator_rmse, indicator_r2, indicator_preds, _ = evaluate_model(
            model_indicator, test_loader_indicator, device
        )
        
        logger.info(f"缺失指示器通用模型 (特征 {feature_index}): "
                    f"MAE={indicator_mae:.4f}, RMSE={indicator_rmse:.4f}, R²={indicator_r2:.4f}")
        
        # ===== 2.2 训练15维缩减模型 (移除当前特征) =====
        logger.info(f"\n--- 训练15维缩减模型 (移除特征 {feature_index}) ---")
        
        # 创建15维训练集/测试集 (移除特征)
        X_train_reduced = np.delete(X_train, feature_index, axis=1)
        X_val_reduced = np.delete(X_val, feature_index, axis=1)
        X_test_reduced = np.delete(X_test, feature_index, axis=1)
        
        # 修复: 15维数据重新标准化
        scaler_reduced = StandardScaler().fit(X_train_reduced)
        X_train_reduced = scaler_reduced.transform(X_train_reduced)
        X_val_reduced = scaler_reduced.transform(X_val_reduced)
        X_test_reduced = scaler_reduced.transform(X_test_reduced)
        
        # 创建15维训练集Dataset
        class BatteryDatasetReduced(Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                x = torch.tensor(self.X[idx], dtype=torch.float32)
                y = torch.tensor(self.y[idx], dtype=torch.float32)
                return x, y
        
        train_dataset_reduced = BatteryDatasetReduced(X_train_reduced, y_train)
        val_dataset_reduced = BatteryDatasetReduced(X_val_reduced, y_val)
        
        train_loader_reduced = DataLoader(train_dataset_reduced, batch_size=args.batch_size, shuffle=True)
        val_loader_reduced = DataLoader(val_dataset_reduced, batch_size=args.batch_size)
        
        # 训练15维模型
        model_reduced = SOHNetwork(input_size=NUM_FEATURES - 1).to(device)
        model_path_reduced = f'{args.results_dir}/best_model_reduced_{feature_index}_{timestamp}.pth'
        model_reduced, _, _ = train_model(
            model_reduced, train_loader_reduced, val_loader_reduced, args.epochs, device, model_path_reduced
        )
        
        # 评估15维模型
        reduced_mae, reduced_rmse, reduced_r2, reduced_preds, _ = evaluate_model(
            model_reduced, DataLoader(BatteryDatasetReduced(X_test_reduced, y_test), batch_size=args.batch_size), device
        )
        
        logger.info(f"15维缩减模型 (特征 {feature_index}): "
                    f"MAE={reduced_mae:.4f}, RMSE={reduced_rmse:.4f}, R²={reduced_r2:.4f}")
        
        # 计算改进百分比
        improvement = (reduced_mae - indicator_mae) / reduced_mae * 100 if reduced_mae > 0 else 0.0
        
        # 保存结果 (确保每个特征只添加一次)
        if feature_index not in results['feature_indices']:
            results['feature_indices'].append(feature_index)
            results['indicator_mae'].append(indicator_mae)
            results['indicator_rmse'].append(indicator_rmse)
            results['indicator_r2'].append(indicator_r2)
            results['reduced_mae'].append(reduced_mae)
            results['reduced_rmse'].append(reduced_rmse)
            results['reduced_r2'].append(reduced_r2)
            results['improvement'].append(improvement)
        
        # 保存详细对比图
        plot_comparison_results(
            y_test, indicator_preds, reduced_preds, feature_index, args.results_dir
        )
    
    # ===== 3. 生成综合比较结果 =====
    logger.info("\n" + "="*60)
    logger.info("正在生成综合比较图表")
    logger.info("="*60)
    
    # 保存结果到CSV
    csv_path = save_results_to_csv(results, args.results_dir)
    
    # 生成总图
    plot_overall_comparison(results, args.results_dir, timestamp)
    
    # 打印结果表格 (确保每个特征只打印一次)
    logger.info("\n" + "="*100)
    logger.info("固定缺失实验结果摘要")
    logger.info("="*100)
    logger.info(f"{'特征索引':<12} {'方法':<25} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'改进率':<12}")
    logger.info("-"*100)
    
    for i in range(len(results['feature_indices'])):
        feature_idx = results['feature_indices'][i]
        logger.info(f"{feature_idx:<12} {'缺失指示器':<25} {results['indicator_mae'][i]:<10.4f} "
                    f"{results['indicator_rmse'][i]:<10.4f} {results['indicator_r2'][i]:<10.4f} {'-':<12}")
        logger.info(f"{'':<12} {'15维缩减模型':<25} {results['reduced_mae'][i]:<10.4f} "
                    f"{results['reduced_rmse'][i]:<10.4f} {results['reduced_r2'][i]:<10.4f} "
                    f"{results['improvement'][i]:<10.1f}%")
        logger.info("-"*100)
    
    logger.info("\n" + "="*100)
    logger.info("实验成功完成！")
    logger.info("="*100)
    logger.info(f"结果时间戳: {timestamp}")
    logger.info(f"结果保存目录: {os.path.abspath(args.results_dir)}")
    logger.info(f"详细结果文件: {csv_path}")
    logger.info("="*100)

if __name__ == "__main__":
    main()