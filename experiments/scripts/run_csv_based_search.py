# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""
鍩轰簬CSV閰嶇疆鏂囦欢鐨勬灦鏋勬悳绱㈣繍琛屽櫒
- 浠嶤SV璇诲彇閰嶇疆
- 杩愯瀹為獙锛堟棤缂哄け鏁版嵁锛?
- 灏嗙粨鏋滃啓鍥濩SV
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime

from loguru import logger
from src.config.pydantic_config import ExperimentConfig
from src.utils.logger_v2 import setup_logger
from src.data.dataset_loader import XJTUDatasetLoader
from src.data.datasets import BatteryDataset, SequenceDataset
from src.models.model_factory import ModelFactory
from src.trainers.lightning_trainer import LightningTrainer, SOHLightningModule
from src.evaluators.model_evaluator import ModelEvaluator
from torch.utils.data import DataLoader
import torch


class CSVBasedSearchRunner:
    """鍩轰簬CSV閰嶇疆鐨勬悳绱㈣繍琛屽櫒"""
    
    def __init__(self, model_type: str, csv_path: str, batch_name: str = '3C', seed: int = 42):
        """
        Args:
            model_type: 妯″瀷绫诲瀷 (mlp/lstm/gru/cnn1d/xgboost)
            csv_path: 閰嶇疆鏂囦欢CSV璺緞
            batch_name: 鏁版嵁鎵规
            seed: 闅忔満绉嶅瓙
        """
        self.model_type = model_type.lower()
        self.csv_path = Path(csv_path)
        self.batch_name = batch_name
        self.seed = seed
        
        # 纭繚CSV瀛樺湪
        if not self.csv_path.exists():
            raise FileNotFoundError(f"閰嶇疆鏂囦欢涓嶅瓨鍦? {csv_path}")
        
        # 璇诲彇閰嶇疆
        self.df = pd.read_csv(csv_path)
        
        # 瀹為獙鐩綍
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(f"./experiments_v2/csv_search_{model_type}_{batch_name}_{timestamp}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 璁剧疆鏃ュ織
        self.logger = setup_logger(
            name=f"csv_search_{model_type}",
            log_file=str(self.exp_dir / "search.log"),
            level="INFO"
        )
        
        # 鍔犺浇鏁版嵁锛堜竴娆″姞杞斤紝閲嶅浣跨敤锛?
        self._load_data()
    
    def _load_data(self):
        """鍔犺浇鏁版嵁闆嗭紙鏃犵己澶憋級"""
        self.logger.info("鍔犺浇鏁版嵁...")
        loader = XJTUDatasetLoader(
            data_dir='./data/XJTU data',
            batch=self.batch_name
        )
        
        self.data = loader.prepare_data(
            feature_cols=ExperimentConfig().feature_cols,
            target_col='capacity',
            test_size=0.25,
            val_size=0.25,
            random_seed=self.seed
        )
        
        self.logger.info(f"鏁版嵁鍔犺浇瀹屾垚:")
        self.logger.info(f"  璁粌闆? {self.data['X_train'].shape}")
        self.logger.info(f"  楠岃瘉闆? {self.data['X_val'].shape}")
        self.logger.info(f"  娴嬭瘯闆? {self.data['X_test'].shape}")
    
    def parse_config(self, row: pd.Series) -> dict:
        """浠嶤SV琛岃В鏋愰厤缃?""
        if self.model_type == 'mlp':
            hidden_str = row['hidden_config'].strip('[]')
            hidden_layers = [int(x) for x in hidden_str.split(',')]
            return {
                'hidden_layers': hidden_layers,
                'dropout': float(row['dropout'])
            }
        
        elif self.model_type in ['lstm', 'gru']:
            config_str = row['hidden_config']  # e.g., "h=64,l=2"
            parts = config_str.split(',')
            hidden = int(parts[0].split('=')[1])
            layers = int(parts[1].split('=')[1])
            return {
                'hidden_size': hidden,
                'num_layers': layers,
                'dropout': float(row['dropout'])
            }
        
        elif self.model_type == 'cnn1d':
            channels_str = row['hidden_config'].strip('[]')
            channels = [int(x) for x in channels_str.split(',')]
            return {
                'channels': channels,
                'kernel_size': int(row['kernel']),
                'dropout': float(row['dropout'])
            }
        
        elif self.model_type == 'xgboost':
            config_str = row['hidden_config']  # e.g., "n=200,d=5,lr=0.1"
            parts = config_str.split(',')
            n_est = int(parts[0].split('=')[1])
            depth = int(parts[1].split('=')[1])
            lr = float(parts[2].split('=')[1])
            return {
                'n_estimators': n_est,
                'max_depth': depth,
                'learning_rate': lr
            }
        
        return {}
    
    def run_single_config(self, idx: int) -> dict:
        """杩愯鍗曚釜閰嶇疆"""
        row = self.df.iloc[idx]
        config_id = row['config_id']
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"杩愯閰嶇疆: {config_id}")
        self.logger.info(f"灞傜被鍨? {row['layer_type']}")
        self.logger.info(f"閰嶇疆: {row['hidden_config']}")
        self.logger.info(f"浼扮畻鍙傛暟閲? {row['estimated_params']:,}")
        
        # 瑙ｆ瀽閰嶇疆
        config = self.parse_config(row)
        
        try:
            # 鍒涘缓妯″瀷
            model = ModelFactory.create_model(
                model_type=self.model_type,
                input_dim=16,
                use_mim=False,  # 鏃犵己澶辨暟鎹?
                device='cpu',
                **config
            )
            
            # 瀹為檯鍙傛暟閲?
            actual_params = ModelFactory.count_parameters(model)
            self.logger.info(f"瀹為檯鍙傛暟閲? {actual_params:,}")
            
            # 鍑嗗鏁版嵁锛堟棤缂哄け锛?
            if self.model_type in ['lstm', 'gru', 'cnn1d']:
                train_dataset = SequenceDataset(
                    self.data['X_train'], self.data['y_train'],
                    seq_len=5,
                    missing_rate=0.0,
                    use_mim=False,
                    seed=self.seed
                )
                val_dataset = SequenceDataset(
                    self.data['X_val'], self.data['y_val'],
                    seq_len=5,
                    missing_rate=0.0,
                    use_mim=False,
                    seed=self.seed
                )
                test_dataset = SequenceDataset(
                    self.data['X_test'], self.data['y_test'],
                    seq_len=5,
                    missing_rate=0.0,
                    use_mim=False,
                    seed=self.seed
                )
            else:
                train_dataset = BatteryDataset(
                    self.data['X_train'], self.data['y_train'],
                    missing_rate=0.0,
                    use_mim=False,
                    seed=self.seed
                )
                val_dataset = BatteryDataset(
                    self.data['X_val'], self.data['y_val'],
                    missing_rate=0.0,
                    use_mim=False,
                    seed=self.seed
                )
                test_dataset = BatteryDataset(
                    self.data['X_test'], self.data['y_test'],
                    missing_rate=0.0,
                    use_mim=False,
                    seed=self.seed
                )
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32)
            
            # 璁粌
            start_time = time.time()
            
            if self.model_type == 'xgboost':
                # XGBoost鐗规畩澶勭悊
                X_train = train_dataset.X_missing
                y_train = train_dataset.y
                X_val = val_dataset.X_missing
                y_val = val_dataset.y
                X_test = test_dataset.X_missing
                y_test = test_dataset.y
                
                model.fit(X_train, y_train, X_val, y_val)
            else:
                pl_module = SOHLightningModule(
                    model.model if hasattr(model, 'model') else model,
                    learning_rate=0.001
                )
                trainer = LightningTrainer(
                    max_epochs=50,
                    patience=10,
                    device='cpu'
                )
                trainer.train(pl_module, train_loader, val_loader)
            
            training_time = time.time() - start_time
            
            # 璇勪及
            evaluator = ModelEvaluator('cpu')
            
            if self.model_type == 'xgboost':
                metrics, _, _ = evaluator.evaluate_xgboost(
                    model.model if hasattr(model, 'model') else model,
                    X_test, y_test
                )
            else:
                metrics, _, _ = evaluator.evaluate(model, test_loader)
            
            results = {
                'actual_params': actual_params,
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'training_time': training_time,
                'status': 'completed'
            }
            
            self.logger.info(f"缁撴灉: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"閰嶇疆 {config_id} 杩愯澶辫触: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                'actual_params': '',
                'mae': '',
                'rmse': '',
                'r2': '',
                'training_time': '',
                'status': 'failed'
            }
    
    def run(self, start_idx: int = 0, end_idx: int = None):
        """
        杩愯鎼滅储
        
        Args:
            start_idx: 璧峰绱㈠紩
            end_idx: 缁撴潫绱㈠紩锛圢one琛ㄧず鍏ㄩ儴锛?
        """
        if end_idx is None:
            end_idx = len(self.df)
        
        self.logger.info("=" * 70)
        self.logger.info(f"CSV鏋舵瀯鎼滅储 - {self.model_type.upper()}")
        self.logger.info("=" * 70)
        self.logger.info(f"鎬婚厤缃暟: {len(self.df)}")
        self.logger.info(f"杩愯鑼冨洿: {start_idx} - {end_idx}")
        self.logger.info(f"鏁版嵁鎵规: {self.batch_name}")
        self.logger.info("=" * 70)
        
        completed = 0
        failed = 0
        
        for idx in range(start_idx, end_idx):
            # 妫€鏌ユ槸鍚﹀凡瀹屾垚
            if self.df.iloc[idx]['status'] == 'completed':
                self.logger.info(f"璺宠繃宸插畬鎴愰厤缃? {self.df.iloc[idx]['config_id']}")
                continue
            
            # 鏇存柊鐘舵€佷负杩愯涓?
            self.df.at[idx, 'status'] = 'running'
            self._save_csv()
            
            # 杩愯瀹為獙
            results = self.run_single_config(idx)
            
            # 鏇存柊缁撴灉
            for key, value in results.items():
                self.df.at[idx, key] = value
            
            # 淇濆瓨CSV
            self._save_csv()
            
            if results['status'] == 'completed':
                completed += 1
            else:
                failed += 1
            
            # 姣?涓繚瀛樹竴娆′腑闂寸粨鏋?
            if idx % 5 == 0:
                self._save_intermediate_results()
        
        self.logger.info("=" * 70)
        self.logger.info("鎼滅储瀹屾垚!")
        self.logger.info(f"瀹屾垚: {completed}, 澶辫触: {failed}")
        self.logger.info(f"缁撴灉淇濆瓨: {self.csv_path}")
        self.logger.info("=" * 70)
    
    def _save_csv(self):
        """淇濆瓨CSV"""
        self.df.to_csv(self.csv_path, index=False)
    
    def _save_intermediate_results(self):
        """淇濆瓨涓棿缁撴灉鍓湰"""
        backup_path = self.exp_dir / f"results_backup_{datetime.now().strftime('%H%M%S')}.csv"
        self.df.to_csv(backup_path, index=False)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='鍩轰簬CSV鐨勬灦鏋勬悳绱?)
    parser.add_argument('--model', choices=['mlp', 'lstm', 'gru', 'cnn1d'],
                       help='妯″瀷绫诲瀷')
    parser.add_argument('--config', type=str, default=None,
                       help='閰嶇疆鏂囦欢CSV璺緞锛堝彲閫夛紝榛樿浣跨敤configs/{model}_configs.csv锛?)
    parser.add_argument('--batch', default='3C', help='鏁版嵁鎵规')
    parser.add_argument('--seed', type=int, default=42, help='闅忔満绉嶅瓙')
    parser.add_argument('--start', type=int, default=0, help='璧峰绱㈠紩')
    parser.add_argument('--end', type=int, default=None, help='缁撴潫绱㈠紩')
    
    args = parser.parse_args()
    
    # 浼樺厛浣跨敤--config鎸囧畾鐨勮矾寰勶紝鍚﹀垯浣跨敤榛樿璺緞
    if args.config:
        csv_path = args.config
    else:
        csv_path = f'configs/{args.model}_configs.csv'
    
    runner = CSVBasedSearchRunner(
        model_type=args.model,
        csv_path=csv_path,
        batch_name=args.batch,
        seed=args.seed
    )
    
    runner.run(start_idx=args.start, end_idx=args.end)


if __name__ == '__main__':
    main()

