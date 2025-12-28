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
import os
import datetime  # 新增导入

# =====================
# 1. 命令行参数与配置
# =====================
parser = argparse.ArgumentParser(description='Minimal PyTorch prototype for SOH estimation with missing data')
parser.add_argument('--missing_rate', type=float, default=0.5, help='Missing rate (0.1-0.9)')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--data_path', type=str, default='./data/XJTU data/2C_battery-1.csv', help='Path to dataset')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()

# 设置随机种子保证可复现性
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# =====================
# 2. 数据处理
# =====================
# 定义数据清洗函数
def clean_data(df, feature_cols, target_col):
    print("Cleaning data: removing inf values and outliers...")
    
    # 1. 替换无穷大值为NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 2. 删除包含NaN的行
    original_shape = df.shape
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(f"  Removed {original_shape[0] - df.shape[0]} rows with NaN/inf values")
    
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
        print(f"  Removing {len(out_index)} rows with extreme outliers")
        df = df.drop(out_index)
        df = df.reset_index(drop=True)
    
    print(f"  Data shape after cleaning: {df.shape}")
    return df

# 定义特征和目标列
feature_cols = ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness', 
                'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
                'current mean', 'current std', 'current kurtosis', 'current skewness',
                'CV Q', 'CV charge time', 'current slope', 'current entropy']
target_col = 'capacity'

# 加载数据
df = pd.read_csv(args.data_path)
print(f"Loaded dataset with shape: {df.shape}")

# 应用数据清洗
df = clean_data(df, feature_cols, target_col)

# 提取特征和目标
X = df[feature_cols].values
y_capacity = df[target_col].values

# 计算SOH (归一化到0-1范围)
initial_capacity = y_capacity[0]
y_soh = y_capacity / initial_capacity
print(f"Initial capacity: {initial_capacity:.2f}, SOH range: [{y_soh.min():.4f}, {y_soh.max():.4f}]")

# 特征标准化 - 使用StandardScaler (z-score标准化，均值为0，标准差为1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_soh, test_size=0.3, random_state=args.seed)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=args.seed)

print(f"Dataset split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# =====================
# 3. 数据集定义
# =====================
class BatteryDataset(Dataset):
    def __init__(self, features, targets, missing_rate=0.0, include_missing_indicators=True, missing_mask=None):
        self.features = features
        self.targets = targets
        self.missing_rate = missing_rate
        self.include_missing_indicators = include_missing_indicators
        
        # 如果没有提供预定义的缺失掩码，则生成新的
        if missing_mask is None:
            self.missing_mask = np.random.binomial(1, 1-missing_rate, size=features.shape)
        else:
            self.missing_mask = missing_mask
        
        # 创建缺失指示器 (1表示缺失, 0表示存在)
        self.missing_indicators = 1 - self.missing_mask
        
        # 应用缺失 (将缺失位置设为0，但保留缺失指示器)
        # 注意: StandardScaler标准化后，0是均值，这是合理的处理方式
        self.features_with_missing = np.where(self.missing_mask == 1, self.features, 0)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # 返回特征值
        feature_values = torch.tensor(self.features_with_missing[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        # 如果需要，添加缺失指示器
        if self.include_missing_indicators:
            missing_indicators = torch.tensor(self.missing_indicators[idx], dtype=torch.float32)
            combined_features = torch.cat([feature_values, missing_indicators])
            return combined_features, target
        else:
            return feature_values, target

# 对比实验：均值填充方法
class MeanFillDataset(Dataset):
    def __init__(self, features, targets, missing_rate=0.0, missing_mask=None):
        self.features = features.copy()
        self.targets = targets
        self.missing_rate = missing_rate
        
        # 如果没有提供预定义的缺失掩码，则生成新的
        if missing_mask is None:
            missing_mask = np.random.binomial(1, 1-missing_rate, size=features.shape)
        
        # 应用缺失
        features_with_missing = np.where(missing_mask == 1, self.features, np.nan)
        
        # 均值填充
        feature_means = np.nanmean(features_with_missing, axis=0)
        self.features_filled = np.where(np.isnan(features_with_missing), feature_means, features_with_missing)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features_filled[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

# =====================
# 4. 模型定义
# =====================
class SOHNetwork(nn.Module):
    def __init__(self, input_size):
        super(SOHNetwork, self).__init__()
        # 根据输入大小动态构建网络结构
        if input_size == 16:
            # 对照组: 16-8-4-2-1
            self.net = nn.Sequential(
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 4),
                nn.ReLU(),
                nn.Linear(4, 2),
                nn.ReLU(),
                nn.Linear(2, 1)
            )
        elif input_size == 32:
            # 研究对象: 32-16-8-4-2-1
            self.net = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 4),
                nn.ReLU(),
                nn.Linear(4, 2),
                nn.ReLU(),
                nn.Linear(2, 1)
            )
        else:
            # 保留原始结构作为回退
            hidden_size = max(32, input_size)
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, max(8, hidden_size // 4)),
                nn.ReLU(),
                nn.Linear(max(8, hidden_size // 4), 1)
            )
    
    def forward(self, x):
        return self.net(x).squeeze()

# =====================
# 5. 训练与评估函数
# =====================
def train_model(model, train_loader, val_loader, epochs, device, model_path):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_loss = float('inf')
    patience = 10
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
            torch.save(model.state_dict(), model_path)  # 保存到指定路径
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(model_path))
    return model, train_losses, val_losses

# 修改评估函数，只返回预测值，使用外部统一的真实值
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:  # 不使用target，只获取预测
            inputs = inputs.to(device)
            output = model(inputs)
            predictions.extend(output.cpu().numpy())
    
    return np.array(predictions)

# =====================
# 6. 主程序
# =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 确保保存结果的目录存在
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 生成唯一标识符（时间戳 + 缺失率）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    missing_rate_str = str(args.missing_rate).replace('.', '_')
    
    # 为所有实验创建固定的缺失掩码，确保可复现性
    train_missing_mask = np.random.binomial(1, 1-args.missing_rate, size=X_train.shape)
    val_missing_mask = np.random.binomial(1, 1-args.missing_rate, size=X_val.shape)
    test_missing_mask = np.random.binomial(1, 1-args.missing_rate, size=X_test.shape)
    
    # 完整数据基线 - 不包含缺失指示器
    print("\n=== Training baseline model (complete data) ===")
    full_train_dataset = BatteryDataset(X_train, y_train, missing_rate=0.0, include_missing_indicators=False)
    full_val_dataset = BatteryDataset(X_val, y_val, missing_rate=0.0, include_missing_indicators=False)
    full_test_dataset = BatteryDataset(X_test, y_test, missing_rate=0.0, include_missing_indicators=False)
    
    train_loader = DataLoader(full_train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(full_val_dataset, batch_size=32)
    test_loader = DataLoader(full_test_dataset, batch_size=32)
    
    # 完整数据基线使用16个输入
    model = SOHNetwork(input_size=16).to(device)
    baseline_path = f'best_model_baseline_{missing_rate_str}_{timestamp}.pth'
    model, baseline_train_losses, baseline_val_losses = train_model(model, train_loader, val_loader, args.epochs, device, baseline_path)
    
    # 评估完整数据基线
    baseline_preds = evaluate_model(model, test_loader, device)
    baseline_mae = np.mean(np.abs(baseline_preds - y_test))
    baseline_rmse = np.sqrt(np.mean((baseline_preds - y_test) ** 2))
    denominator = np.sum((y_test - np.mean(y_test)) ** 2)
    baseline_r2 = 1 - (np.sum((y_test - baseline_preds) ** 2) / denominator) if denominator > 1e-8 else 0.0
    
    print(f"Baseline (complete data) - MAE: {baseline_mae:.4f}, RMSE: {baseline_rmse:.4f}, R²: {baseline_r2:.4f}")
    
    # 缺失数据实验 (带缺失指示器)
    print(f"\n=== Training model with missing data (rate={args.missing_rate}) with missing indicators ===")
    miss_train_dataset = BatteryDataset(X_train, y_train, missing_rate=args.missing_rate, 
                                       include_missing_indicators=True, missing_mask=train_missing_mask)
    miss_val_dataset = BatteryDataset(X_val, y_val, missing_rate=args.missing_rate, 
                                     include_missing_indicators=True, missing_mask=val_missing_mask)
    miss_test_dataset = BatteryDataset(X_test, y_test, missing_rate=args.missing_rate, 
                                      include_missing_indicators=True, missing_mask=test_missing_mask)
    
    train_loader = DataLoader(miss_train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(miss_val_dataset, batch_size=32)
    test_loader = DataLoader(miss_test_dataset, batch_size=32)
    
    # 带缺失指示器的模型，输入大小为32 (16原始特征 + 16指示器)
    model = SOHNetwork(input_size=32).to(device)
    indicator_path = f'best_model_indicator_{missing_rate_str}_{timestamp}.pth'
    model, indicator_train_losses, indicator_val_losses = train_model(model, train_loader, val_loader, args.epochs, device, indicator_path)
    
    # 评估缺失指示器方法
    indicator_preds = evaluate_model(model, test_loader, device)
    indicator_mae = np.mean(np.abs(indicator_preds - y_test))
    indicator_rmse = np.sqrt(np.mean((indicator_preds - y_test) ** 2))
    denominator = np.sum((y_test - np.mean(y_test)) ** 2)
    indicator_r2 = 1 - (np.sum((y_test - indicator_preds) ** 2) / denominator) if denominator > 1e-8 else 0.0
    
    print(f"Missing indicators method - MAE: {indicator_mae:.4f}, RMSE: {indicator_rmse:.4f}, R²: {indicator_r2:.4f}")
    
    # 均值填充方法
    print(f"\n=== Training model with missing data (rate={args.missing_rate}) using mean filling ===")
    fill_train_dataset = MeanFillDataset(X_train, y_train, missing_rate=args.missing_rate, missing_mask=train_missing_mask)
    fill_val_dataset = MeanFillDataset(X_val, y_val, missing_rate=args.missing_rate, missing_mask=val_missing_mask)
    fill_test_dataset = MeanFillDataset(X_test, y_test, missing_rate=args.missing_rate, missing_mask=test_missing_mask)
    
    train_loader = DataLoader(fill_train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(fill_val_dataset, batch_size=32)
    test_loader = DataLoader(fill_test_dataset, batch_size=32)
    
    # 均值填充方法只有原始特征，没有缺失指示器
    model_fill = SOHNetwork(input_size=16).to(device)
    fill_path = f'best_model_fill_{missing_rate_str}_{timestamp}.pth'
    model_fill, fill_train_losses, fill_val_losses = train_model(model_fill, train_loader, val_loader, args.epochs, device, fill_path)
    
    # 评估均值填充方法
    fill_preds = evaluate_model(model_fill, test_loader, device)
    fill_mae = np.mean(np.abs(fill_preds - y_test))
    fill_rmse = np.sqrt(np.mean((fill_preds - y_test) ** 2))
    denominator = np.sum((y_test - np.mean(y_test)) ** 2)
    fill_r2 = 1 - (np.sum((y_test - fill_preds) ** 2) / denominator) if denominator > 1e-8 else 0.0
    
    print(f"Mean filling method - MAE: {fill_mae:.4f}, RMSE: {fill_rmse:.4f}, R²: {fill_r2:.4f}")
    
    # 可视化与结果输出
    plt.figure(figsize=(12, 10))
    
    # 损失曲线 - 展示所有实验的训练过程
    plt.subplot(2, 1, 1)
    plt.plot(indicator_train_losses, 'r-', label='Indicator Train Loss')
    plt.plot(indicator_val_losses, 'r--', label='Indicator Val Loss')
    plt.plot(fill_train_losses, 'g-', label='Mean Fill Train Loss')
    plt.plot(fill_val_losses, 'g--', label='Mean Fill Val Loss')
    plt.plot(baseline_train_losses, 'b-', label='Baseline Train Loss')
    plt.plot(baseline_val_losses, 'b--', label='Baseline Val Loss')
    plt.title(f'Training and Validation Loss (Missing Rate: {args.missing_rate})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 预测结果对比 - 使用统一的真实值y_test
    plt.subplot(2, 1, 2)
    plt.plot(y_test, 'k-', label='True SOH', linewidth=2)
    plt.plot(indicator_preds, 'r--', label='Prediction with Missing Indicators', alpha=0.8)
    plt.plot(fill_preds, 'g-.', label='Prediction with Mean Filling', alpha=0.8)
    plt.plot(baseline_preds, 'b:', label='Baseline (Complete Data)', alpha=0.8)
    plt.title('SOH Estimation Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('SOH')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    result_path = f'results/comparison_{missing_rate_str}_{timestamp}.png'
    plt.savefig(result_path)
    print(f"Results saved to {result_path}")
    
    # 结果总结
    print("\n" + "="*50)
    print(f"=== 实验结果 (缺失率: {args.missing_rate}) ===")
    print(f"完整数据基线: MAE={baseline_mae:.4f}, RMSE={baseline_rmse:.4f}, R²={baseline_r2:.4f}")
    print(f"本方法 (指示器): MAE={indicator_mae:.4f}, RMSE={indicator_rmse:.4f}, R²={indicator_r2:.4f}")
    print(f"均值填充方法: MAE={fill_mae:.4f}, RMSE={fill_rmse:.4f}, R²={fill_r2:.4f}")
    
    # 计算改进百分比
    improvement = (fill_mae - indicator_mae) / fill_mae * 100
    print(f"优势: 本方法比填充方法误差降低{improvement:.1f}%")
    print("="*50)

if __name__ == "__main__":
    main()