# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""
瀵规瘮绮楃矑搴﹀拰缁嗙矑搴︽悳绱㈢粨鏋?

浣跨敤绀轰緥:
    python compare_search_results.py --coarse results_coarse.csv --fine results_fine.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def compare_results(coarse_path: str, fine_path: str, model_type: str):
    """瀵规瘮绮楃矑搴﹀拰缁嗙矑搴︽悳绱㈢粨鏋?""
    
    # 璇诲彇缁撴灉
    df_coarse = pd.read_csv(coarse_path)
    df_fine = pd.read_csv(fine_path)
    
    # 绛涢€夋寚瀹氭ā鍨?
    df_coarse = df_coarse[df_coarse['model_type'] == model_type]
    df_fine = df_fine[df_fine['model_type'] == model_type]
    
    print("=" * 70)
    print(f"{model_type.upper()} 鎼滅储缁撴灉瀵规瘮")
    print("=" * 70)
    
    # 绮楃矑搴︽渶浣?
    if len(df_coarse) > 0:
        best_coarse = df_coarse.loc[df_coarse['mae_mean'].idxmin()]
        print(f"\n銆愮矖绮掑害鏈€浣炽€?)
        print(f"  閰嶇疆: {best_coarse['config_name']}")
        print(f"  MAE: {best_coarse['mae_mean']:.4f}")
        print(f"  鍙傛暟閲? {best_coarse['param_count']:,}")
        print(f"  閰嶇疆璇︽儏: {best_coarse['config']}")
    
    # 缁嗙矑搴︽渶浣?
    if len(df_fine) > 0:
        best_fine = df_fine.loc[df_fine['mae_mean'].idxmin()]
        print(f"\n銆愮粏绮掑害鏈€浣炽€?)
        print(f"  閰嶇疆: {best_fine['config_name']}")
        print(f"  MAE: {best_fine['mae_mean']:.4f}")
        print(f"  鍙傛暟閲? {best_fine['param_count']:,}")
        print(f"  閰嶇疆璇︽儏: {best_fine['config']}")
        
        # 鏀硅繘骞呭害
        if len(df_coarse) > 0:
            improvement = (best_coarse['mae_mean'] - best_fine['mae_mean']) / best_coarse['mae_mean'] * 100
            print(f"\n銆愭敼杩涘箙搴︺€?)
            print(f"  MAE闄嶄綆: {improvement:.2f}%")
            print(f"  缁濆鏀硅繘: {best_coarse['mae_mean'] - best_fine['mae_mean']:.4f}")
    
    # 缁樺埗瀵规瘮鍥?
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 鏁ｇ偣鍥惧姣?
    ax = axes[0]
    ax.scatter(df_coarse['param_count'], df_coarse['mae_mean'], 
              alpha=0.6, s=100, label='Coarse Search', color='blue')
    ax.scatter(df_fine['param_count'], df_fine['mae_mean'], 
              alpha=0.6, s=50, label='Fine Search', color='red')
    
    if len(df_coarse) > 0:
        ax.scatter(best_coarse['param_count'], best_coarse['mae_mean'], 
                  s=200, marker='*', color='blue', edgecolors='black', 
                  label='Coarse Best', zorder=5)
    if len(df_fine) > 0:
        ax.scatter(best_fine['param_count'], best_fine['mae_mean'], 
                  s=200, marker='*', color='red', edgecolors='black', 
                  label='Fine Best', zorder=5)
    
    ax.set_xlabel('Parameter Count')
    ax.set_ylabel('MAE')
    ax.set_title(f'{model_type.upper()}: Coarse vs Fine Search')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MAE鍒嗗竷鐩存柟鍥?
    ax = axes[1]
    if len(df_coarse) > 0 and len(df_fine) > 0:
        ax.hist(df_coarse['mae_mean'], bins=15, alpha=0.5, label='Coarse', color='blue')
        ax.hist(df_fine['mae_mean'], bins=20, alpha=0.5, label='Fine', color='red')
        ax.axvline(best_coarse['mae_mean'], color='blue', linestyle='--', 
                  label=f'Coarse Best: {best_coarse["mae_mean"]:.4f}')
        ax.axvline(best_fine['mae_mean'], color='red', linestyle='--',
                  label=f'Fine Best: {best_fine["mae_mean"]:.4f}')
        ax.set_xlabel('MAE')
        ax.set_ylabel('Count')
        ax.set_title('MAE Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(fine_path).parent / f'comparison_{model_type}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n瀵规瘮鍥惧凡淇濆瓨: {output_path}")
    
    # 鏄剧ず缁嗙矑搴op 10
    if len(df_fine) > 0:
        print("\n" + "=" * 70)
        print(f"缁嗙矑搴︽悳绱?Top 10 ({model_type.upper()})")
        print("=" * 70)
        top10 = df_fine.nsmallest(10, 'mae_mean')[['config_name', 'mae_mean', 'param_count']]
        print(top10.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='瀵规瘮绮楃矑搴﹀拰缁嗙矑搴︽悳绱㈢粨鏋?)
    parser.add_argument('--coarse', required=True, help='绮楃矑搴︾粨鏋淐SV璺緞')
    parser.add_argument('--fine', required=True, help='缁嗙矑搴︾粨鏋淐SV璺緞')
    parser.add_argument('--model', default='all',
                       choices=['mlp', 'lstm', 'gru', 'cnn1d', 'xgboost', 'all'],
                       help='瑕佸姣旂殑妯″瀷')
    
    args = parser.parse_args()
    
    if args.model == 'all':
        for model in ['mlp', 'lstm', 'gru', 'cnn1d', 'xgboost']:
            compare_results(args.coarse, args.fine, model)
            print("\n" + "=" * 70 + "\n")
    else:
        compare_results(args.coarse, args.fine, args.model)


if __name__ == '__main__':
    main()

