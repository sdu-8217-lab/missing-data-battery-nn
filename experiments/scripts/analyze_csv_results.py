# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""
鍒嗘瀽CSV鏋舵瀯鎼滅储缁撴灉
姹囨€诲悇妯″瀷鐨勫笗绱墭鍓嶆部
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def analyze_model_results(csv_path: str, model_type: str):
    """鍒嗘瀽鍗曚釜妯″瀷鐨勭粨鏋?""
    df = pd.read_csv(csv_path)
    
    # 鍙垎鏋愬凡瀹屾垚鐨?
    df_done = df[df['status'] == 'completed'].copy()
    
    if len(df_done) == 0:
        print(f"{model_type.upper()}: 鏃犲畬鎴愬疄楠?)
        return None
    
    # 杞崲鏁板€煎垪
    for col in ['actual_params', 'mae', 'rmse', 'r2', 'training_time']:
        if col in df_done.columns:
            df_done[col] = pd.to_numeric(df_done[col], errors='coerce')
    
    print(f"\n{'='*70}")
    print(f"{model_type.upper()} 鍒嗘瀽缁撴灉")
    print(f"{'='*70}")
    print(f"瀹屾垚閰嶇疆鏁? {len(df_done)} / {len(df)}")
    
    # 鎸夊眰鍒嗘瀽
    for layer in sorted(df_done['layer_type'].unique()):
        layer_df = df_done[df_done['layer_type'] == layer]
        if len(layer_df) > 0:
            best = layer_df.loc[layer_df['mae'].idxmin()]
            print(f"\n灞倇layer}: {len(layer_df)}涓厤缃?)
            print(f"  鏈€浣矼AE: {best['mae']:.4f}")
            print(f"  閰嶇疆: {best['config_id']}")
            print(f"  鍙傛暟閲? {best['actual_params']:,.0f}")
    
    # 鍏ㄥ眬鏈€浣?
    best_global = df_done.loc[df_done['mae'].idxmin()]
    print(f"\n{'='*70}")
    print(f"鍏ㄥ眬鏈€浣?")
    print(f"  閰嶇疆: {best_global['config_id']}")
    print(f"  MAE: {best_global['mae']:.4f}")
    print(f"  RMSE: {best_global['rmse']:.4f}")
    print(f"  R2: {best_global['r2']:.4f}")
    print(f"  鍙傛暟閲? {best_global['actual_params']:,.0f}")
    print(f"  閰嶇疆璇︽儏: {best_global['hidden_config']}")
    print(f"{'='*70}")
    
    return df_done


def plot_pareto_comparison(all_results: dict, output_dir: Path):
    """缁樺埗甯曠疮鎵樺墠娌垮姣?""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = {'mlp': 'blue', 'lstm': 'orange', 'gru': 'green', 'cnn1d': 'red'}
    
    # 1. 甯曠疮鎵樺墠娌?- 鎵€鏈夋ā鍨?
    ax = axes[0, 0]
    for model_type, df in all_results.items():
        if df is not None and len(df) > 0:
            ax.scatter(df['actual_params'], df['mae'], 
                      alpha=0.5, s=30, label=model_type.upper(),
                      color=colors.get(model_type, 'gray'))
            
            # 鏍囪鏈€浣?
            best = df.loc[df['mae'].idxmin()]
            ax.scatter(best['actual_params'], best['mae'],
                      s=200, marker='*', color=colors.get(model_type, 'gray'),
                      edgecolors='black', zorder=5)
    
    ax.set_xlabel('Parameters')
    ax.set_ylabel('MAE')
    ax.set_title('Pareto Frontier: All Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100000)
    
    # 2. 鍚勫眰鏈€浣冲姣?
    ax = axes[0, 1]
    layer_best = {}
    for model_type, df in all_results.items():
        if df is not None:
            for layer in [1, 2, 3, 4]:
                layer_df = df[df['layer_type'] == layer]
                if len(layer_df) > 0:
                    best = layer_df.loc[layer_df['mae'].idxmin()]
                    key = f"{model_type}_l{layer}"
                    layer_best[key] = best['mae']
    
    if layer_best:
        items = sorted(layer_best.items(), key=lambda x: x[1])
        names = [x[0] for x in items[:15]]  # 鍙樉绀哄墠15
        values = [x[1] for x in items[:15]]
        ax.barh(range(len(names)), values, color='steelblue')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('MAE')
        ax.set_title('Best MAE by Model & Layer')
        ax.grid(True, alpha=0.3, axis='x')
    
    # 3. MAE鍒嗗竷
    ax = axes[1, 0]
    for model_type, df in all_results.items():
        if df is not None and len(df) > 0:
            ax.hist(df['mae'], bins=20, alpha=0.4, label=model_type.upper(),
                   color=colors.get(model_type, 'gray'))
    ax.set_xlabel('MAE')
    ax.set_ylabel('Count')
    ax.set_title('MAE Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 璁粌鏃堕棿 vs 鎬ц兘
    ax = axes[1, 1]
    for model_type, df in all_results.items():
        if df is not None and len(df) > 0:
            scatter = ax.scatter(df['training_time'], df['mae'],
                               alpha=0.5, s=30, label=model_type.upper(),
                               color=colors.get(model_type, 'gray'))
    ax.set_xlabel('Training Time (s)')
    ax.set_ylabel('MAE')
    ax.set_title('Training Time vs Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'pareto_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n鍥捐〃淇濆瓨: {output_path}")


def generate_summary_csv(all_results: dict, output_dir: Path):
    """鐢熸垚姹囨€籆SV"""
    summary = []
    
    for model_type, df in all_results.items():
        if df is None or len(df) == 0:
            continue
        
        # 鍏ㄥ眬鏈€浣?
        best = df.loc[df['mae'].idxmin()]
        summary.append({
            'model': model_type,
            'layer_type': 'best',
            'config_id': best['config_id'],
            'mae': best['mae'],
            'rmse': best['rmse'],
            'r2': best['r2'],
            'params': best['actual_params'],
            'training_time': best['training_time'],
            'config': best['hidden_config']
        })
        
        # 姣忓眰鏈€浣?
        for layer in sorted(df['layer_type'].unique()):
            layer_df = df[df['layer_type'] == layer]
            if len(layer_df) > 0:
                best_layer = layer_df.loc[layer_df['mae'].idxmin()]
                summary.append({
                    'model': model_type,
                    'layer_type': layer,
                    'config_id': best_layer['config_id'],
                    'mae': best_layer['mae'],
                    'rmse': best_layer['rmse'],
                    'r2': best_layer['r2'],
                    'params': best_layer['actual_params'],
                    'training_time': best_layer['training_time'],
                    'config': best_layer['hidden_config']
                })
    
    summary_df = pd.DataFrame(summary)
    summary_path = output_dir / 'summary_best_configs.csv'
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n姹囨€昏〃淇濆瓨: {summary_path}")
    print("\n鏈€浣抽厤缃眹鎬?")
    print(summary_df[summary_df['layer_type'] == 'best'].to_string(index=False))
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='鍒嗘瀽CSV鎼滅储缁撴灉')
    parser.add_argument('--models', nargs='+', 
                       default=['mlp', 'lstm', 'gru', 'cnn1d'],
                       help='瑕佸垎鏋愮殑妯″瀷鍒楄〃')
    parser.add_argument('--output', default='./analysis_results',
                       help='杈撳嚭鐩綍')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # 鍒嗘瀽鍚勬ā鍨?
    all_results = {}
    for model in args.models:
        csv_path = f'configs/{model}_configs.csv'
        if Path(csv_path).exists():
            results = analyze_model_results(csv_path, model)
            all_results[model] = results
        else:
            print(f"鏈壘鍒? {csv_path}")
    
    # 鐢熸垚鍥捐〃
    if all_results:
        plot_pareto_comparison(all_results, output_dir)
        generate_summary_csv(all_results, output_dir)
    
    print("\n" + "="*70)
    print("鍒嗘瀽瀹屾垚!")
    print(f"杈撳嚭鐩綍: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()

