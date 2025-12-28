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
# 1. 参数
# =========================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--missing_rates', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                   help='Missing rates to test (0.1-0.9)')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# =========================
# 2. 数据清洗
# =========================
def clean_data(df, feature_cols, target_col):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    out_idx = []
    for col in feature_cols + [target_col]:
        mu, sigma = df[col].mean(), df[col].std()
        if sigma > 0:
            outliers = df[(df[col] < mu - 3 * sigma) | (df[col] > mu + 3 * sigma)].index
            out_idx.extend(outliers.tolist())

    if len(out_idx) > 0:
        df = df.drop(list(set(out_idx))).reset_index(drop=True)

    return df

# =========================
# 3. Dataset
# =========================
class BatteryDatasetDynamicMissing(Dataset):
    def __init__(self, X, y, max_missing_rate=0.9):
        self.X = X
        self.y = y
        self.max_missing_rate = max_missing_rate

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        mr = np.random.uniform(0.0, self.max_missing_rate)
        mask = np.random.binomial(1, 1 - mr, size=x.shape)

        x_miss = np.where(mask == 1, x, 0.0)
        indicator = 1 - mask

        x = torch.tensor(x_miss, dtype=torch.float32)
        m = torch.tensor(indicator, dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        return torch.cat([x, m]), y

class BatteryDatasetFixedMissing(Dataset):
    def __init__(self, X, y, missing_rate):
        self.y = y
        self.mask = np.random.binomial(1, 1 - missing_rate, size=X.shape)
        self.X_miss = np.where(self.mask == 1, X, 0.0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X_miss[idx], dtype=torch.float32)
        m = torch.tensor(1 - self.mask[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return torch.cat([x, m]), y

# =========================
# 均值填充Dataset
# =========================
class MeanFillDataset(Dataset):
    def __init__(self, X, y, missing_rate=0.0):
        self.X = X.copy()
        self.y = y
        self.missing_rate = missing_rate
        
        # 生成随机缺失掩码
        missing_mask = np.random.binomial(1, 1-missing_rate, size=X.shape)
        
        # 应用缺失
        features_with_missing = np.where(missing_mask == 1, self.X, np.nan)
        
        # 均值填充
        feature_means = np.nanmean(features_with_missing, axis=0)
        self.features_filled = np.where(np.isnan(features_with_missing), feature_means, features_with_missing)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features_filled[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# =========================
# 4. 模型
# =========================
class SOHNetwork(nn.Module):
    def __init__(self, input_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()

# =========================
# 5. 训练（添加早停机制）
# =========================
def train_model(model, train_loader, val_loader, epochs, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val = float('inf')
    patience = 10
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * x.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(f"[Epoch {epoch+1:03d}] Train={train_loss:.6f}  Val={val_loss:.6f}")

    model.load_state_dict(torch.load('best_model.pth'))
    return model, train_losses, val_losses

# =========================
# 6. 评估
# =========================
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds.append(model(x).cpu().numpy())
            trues.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    r2 = 1 - np.sum((preds - trues) ** 2) / np.sum((trues - trues.mean()) ** 2)

    return mae, rmse, r2, preds, trues

# =========================
# 7. 主程序
# =========================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    print(f"Testing missing rates: {args.missing_rates}")

    feature_cols = [
        'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
        'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
        'current mean', 'current std', 'current kurtosis', 'current skewness',
        'CV Q', 'CV charge time', 'current slope', 'current entropy'
    ]
    target_col = 'capacity'

    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "data" / "XJTU data" / "2C_battery-1.csv"
    print(f"Loading data from: {DATA_PATH}")

    df = clean_data(pd.read_csv(DATA_PATH), feature_cols, target_col)

    X = StandardScaler().fit_transform(df[feature_cols].values)
    cap = df[target_col].values
    y = cap / cap[0]

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=args.seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=args.seed)

    print(f"Dataset split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # 存储所有结果
    results = {
        'missing_rates': args.missing_rates,
        'baseline_mae': [], 'baseline_rmse': [], 'baseline_r2': [],
        'indicator_mae': [], 'indicator_rmse': [], 'indicator_r2': [],
        'fill_mae': [], 'fill_rmse': [], 'fill_r2': [],
        'improvement': []
    }

    # ===== 完整数据基线（只运行一次） =====
    print("\n" + "="*60)
    print("=== Training baseline model (complete data) ===")
    print("="*60)
    
    full_train_dataset = BatteryDatasetFixedMissing(X_train, y_train, missing_rate=0.0)
    full_val_dataset = BatteryDatasetFixedMissing(X_val, y_val, missing_rate=0.0)
    full_test_dataset = BatteryDatasetFixedMissing(X_test, y_test, missing_rate=0.0)
    
    train_loader = DataLoader(full_train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(full_val_dataset, batch_size=32)
    test_loader = DataLoader(full_test_dataset, batch_size=32)
    
    model_baseline = SOHNetwork(input_size=32).to(device)
    model_baseline, _, _ = train_model(model_baseline, train_loader, val_loader, args.epochs, device)
    baseline_mae, baseline_rmse, baseline_r2, _, _ = evaluate(model_baseline, test_loader, device)
    
    # 为所有缺失率保存相同的基线结果
    for _ in args.missing_rates:
        results['baseline_mae'].append(baseline_mae)
        results['baseline_rmse'].append(baseline_rmse)
        results['baseline_r2'].append(baseline_r2)
    
    print(f"Baseline (complete data) - MAE: {baseline_mae:.4f}, RMSE: {baseline_rmse:.4f}, R²: {baseline_r2:.4f}")

    # ===== 对不同缺失率进行测试 =====
    for i, missing_rate in enumerate(args.missing_rates):
        print(f"\n" + "="*60)
        print(f"=== Testing Missing Rate: {missing_rate*100:.0f}% ===")
        print("="*60)
        
        # 设置当前缺失率的随机种子（确保可重复性）
        torch.manual_seed(args.seed + i)
        np.random.seed(args.seed + i)

        # ===== 缺失指示器方法 =====
        print(f"\n--- Missing Indicators Method ---")
        train_loader = DataLoader(
            BatteryDatasetDynamicMissing(X_train, y_train, max_missing_rate=missing_rate), 
            batch_size=32, shuffle=True
        )
        val_loader = DataLoader(
            BatteryDatasetFixedMissing(X_val, y_val, missing_rate), 
            batch_size=32
        )

        model_indicator = SOHNetwork(input_size=32).to(device)
        model_indicator, train_losses, val_losses = train_model(
            model_indicator, train_loader, val_loader, args.epochs, device
        )

        test_loader = DataLoader(
            BatteryDatasetFixedMissing(X_test, y_test, missing_rate), 
            batch_size=32
        )
        indicator_mae, indicator_rmse, indicator_r2, indicator_preds, indicator_targets = evaluate(
            model_indicator, test_loader, device
        )
        
        results['indicator_mae'].append(indicator_mae)
        results['indicator_rmse'].append(indicator_rmse)
        results['indicator_r2'].append(indicator_r2)
        
        print(f"MAE: {indicator_mae:.4f}, RMSE: {indicator_rmse:.4f}, R²: {indicator_r2:.4f}")

        # ===== 均值填充方法 =====
        print(f"\n--- Mean Filling Method ---")
        fill_train_dataset = MeanFillDataset(X_train, y_train, missing_rate=missing_rate)
        fill_val_dataset = MeanFillDataset(X_val, y_val, missing_rate=missing_rate)
        fill_test_dataset = MeanFillDataset(X_test, y_test, missing_rate=missing_rate)
        
        train_loader = DataLoader(fill_train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(fill_val_dataset, batch_size=32)
        test_loader = DataLoader(fill_test_dataset, batch_size=32)
        
        model_fill = SOHNetwork(input_size=16).to(device)
        model_fill, _, _ = train_model(model_fill, train_loader, val_loader, args.epochs, device)
        fill_mae, fill_rmse, fill_r2, fill_preds, fill_targets = evaluate(model_fill, test_loader, device)
        
        results['fill_mae'].append(fill_mae)
        results['fill_rmse'].append(fill_rmse)
        results['fill_r2'].append(fill_r2)
        
        # 计算改进百分比
        improvement = (fill_mae - indicator_mae) / fill_mae * 100
        results['improvement'].append(improvement)
        
        print(f"MAE: {fill_mae:.4f}, RMSE: {fill_rmse:.4f}, R²: {fill_r2:.4f}")
        print(f"Improvement: {improvement:.1f}%")

        # ===== 为每个缺失率保存详细结果图 =====
        plt.figure(figsize=(15, 10))
        
        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Training Curves (Missing Rate: {missing_rate*100:.0f}%)')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        
        # 预测结果对比
        plt.subplot(2, 2, 2)
        plt.plot(indicator_targets, 'b-', label='True SOH', linewidth=2)
        plt.plot(indicator_preds, 'r--', label='Prediction with Missing Indicators', alpha=0.8)
        plt.plot(fill_preds, 'g-.', label='Prediction with Mean Filling', alpha=0.8)
        plt.title(f'SOH Estimation Comparison (MR: {missing_rate*100:.0f}%)')
        plt.xlabel('Sample Index')
        plt.ylabel('SOH')
        plt.legend()
        plt.grid(True)
        
        # 误差分布
        plt.subplot(2, 2, 3)
        indicator_errors = np.abs(indicator_preds - indicator_targets)
        fill_errors = np.abs(fill_preds - indicator_targets)
        
        plt.boxplot([indicator_errors, fill_errors], labels=['Missing Indicators', 'Mean Filling'])
        plt.title('Absolute Error Distribution')
        plt.ylabel('Absolute Error')
        plt.grid(True)
        
        # 性能对比柱状图
        plt.subplot(2, 2, 4)
        methods = ['Baseline', 'Indicators', 'Mean Fill']
        mae_values = [baseline_mae, indicator_mae, fill_mae]
        
        bars = plt.bar(methods, mae_values, color=['blue', 'red', 'green'], alpha=0.7)
        plt.title('MAE Comparison')
        plt.ylabel('MAE')
        
        # 在柱子上添加数值
        for bar, value in zip(bars, mae_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results_detailed_mr_{missing_rate*100:.0f}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Detailed results saved to results_detailed_mr_{missing_rate*100:.0f}.png")

    # ===== 生成综合比较图 =====
    print("\n" + "="*60)
    print("Generating comprehensive comparison plots...")
    print("="*60)
    
    plt.figure(figsize=(15, 10))
    
    # MAE随缺失率变化趋势
    plt.subplot(2, 2, 1)
    plt.plot(results['missing_rates'], results['baseline_mae'], 'k-', label='Baseline (Complete Data)', linewidth=3)
    plt.plot(results['missing_rates'], results['indicator_mae'], 'r-o', label='Missing Indicators', linewidth=2)
    plt.plot(results['missing_rates'], results['fill_mae'], 'g--s', label='Mean Filling', linewidth=2)
    plt.xlabel('Missing Rate')
    plt.ylabel('MAE')
    plt.title('MAE vs Missing Rate')
    plt.legend()
    plt.grid(True)
    
    # RMSE随缺失率变化趋势
    plt.subplot(2, 2, 2)
    plt.plot(results['missing_rates'], results['baseline_rmse'], 'k-', label='Baseline (Complete Data)', linewidth=3)
    plt.plot(results['missing_rates'], results['indicator_rmse'], 'r-o', label='Missing Indicators', linewidth=2)
    plt.plot(results['missing_rates'], results['fill_rmse'], 'g--s', label='Mean Filling', linewidth=2)
    plt.xlabel('Missing Rate')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Missing Rate')
    plt.legend()
    plt.grid(True)
    
    # R²随缺失率变化趋势
    plt.subplot(2, 2, 3)
    plt.plot(results['missing_rates'], results['baseline_r2'], 'k-', label='Baseline (Complete Data)', linewidth=3)
    plt.plot(results['missing_rates'], results['indicator_r2'], 'r-o', label='Missing Indicators', linewidth=2)
    plt.plot(results['missing_rates'], results['fill_r2'], 'g--s', label='Mean Filling', linewidth=2)
    plt.xlabel('Missing Rate')
    plt.ylabel('R² Score')
    plt.title('R² Score vs Missing Rate')
    plt.legend()
    plt.grid(True)
    
    # 改进百分比
    plt.subplot(2, 2, 4)
    plt.bar(range(len(results['missing_rates'])), results['improvement'], 
            color='orange', alpha=0.7, tick_label=[f'{mr*100:.0f}%' for mr in results['missing_rates']])
    plt.xlabel('Missing Rate')
    plt.ylabel('Improvement (%)')
    plt.title('Improvement of Missing Indicators over Mean Filling')
    plt.grid(True, alpha=0.3)
    
    # 在柱子上添加数值
    for i, improvement in enumerate(results['improvement']):
        plt.text(i, improvement + (0.5 if improvement >= 0 else -1), 
                f'{improvement:.1f}%', ha='center', va='bottom' if improvement >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('results_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== 输出详细结果表格 =====
    print("\n" + "="*80)
    print("SUMMARY RESULTS TABLE")
    print("="*80)
    print(f"{'Missing Rate':<12} {'Method':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'Improvement':<12}")
    print("-"*80)
    
    for i, mr in enumerate(results['missing_rates']):
        print(f"{mr*100:>10.0f}%  {'Baseline':<20} {results['baseline_mae'][i]:<10.4f} {results['baseline_rmse'][i]:<10.4f} {results['baseline_r2'][i]:<10.4f} {'-':<12}")
        print(f"{'':<12} {'Missing Indicators':<20} {results['indicator_mae'][i]:<10.4f} {results['indicator_rmse'][i]:<10.4f} {results['indicator_r2'][i]:<10.4f} {'-':<12}")
        print(f"{'':<12} {'Mean Filling':<20} {results['fill_mae'][i]:<10.4f} {results['fill_rmse'][i]:<10.4f} {results['fill_r2'][i]:<10.4f} {results['improvement'][i]:<10.1f}%")
        print("-"*80)
    
    # ===== 保存结果到CSV文件 =====
    results_df = pd.DataFrame({
        'Missing_Rate': results['missing_rates'],
        'Baseline_MAE': results['baseline_mae'],
        'Baseline_RMSE': results['baseline_rmse'],
        'Baseline_R2': results['baseline_r2'],
        'Indicator_MAE': results['indicator_mae'],
        'Indicator_RMSE': results['indicator_rmse'],
        'Indicator_R2': results['indicator_r2'],
        'Fill_MAE': results['fill_mae'],
        'Fill_RMSE': results['fill_rmse'],
        'Fill_R2': results['fill_r2'],
        'Improvement_Percent': results['improvement']
    })
    results_df.to_csv('experiment_results.csv', index=False)
    print(f"\nDetailed results saved to experiment_results.csv")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("Generated files:")
    print("1. results_detailed_mr_XX.png - Detailed results for each missing rate")
    print("2. results_comprehensive_comparison.png - Overall performance trends")
    print("3. experiment_results.csv - Complete results in CSV format")
    print("="*80)

if __name__ == '__main__':
    main()