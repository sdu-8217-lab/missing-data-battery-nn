"""Single experiment runner - executes one training job."""

import os
import subprocess
import time
import torch
from pathlib import Path
from typing import Dict, Optional, Any
import json
import re

from .database import ExperimentRecord, ExperimentStatus


class ExperimentRunner:
    """Runs a single experiment with proper resource management."""
    
    def __init__(self, 
                 output_dir: str = "experiments/logs",
                 use_gpu: bool = True,
                 gpu_id: int = 0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 注意：在 spawn 模式下，子进程可能没有 CUDA 上下文
        # 延迟 CUDA 检查，只通过参数存储配置
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        
        # 只在确认有 CUDA 上下文时才检查 GPU
        if use_gpu:
            try:
                if torch.cuda.is_available():
                    print(f"GPU available: {torch.cuda.get_device_name(self.gpu_id)}")
                    print(f"   Memory: {torch.cuda.get_device_properties(self.gpu_id).total_memory / 1e9:.1f} GB")
                else:
                    print("GPU not available in subprocess, using CPU")
                    self.use_gpu = False
                    self.gpu_id = None
            except RuntimeError:
                print("GPU context not available in subprocess, using CPU")
                self.use_gpu = False
                self.gpu_id = None
        else:
            print("Using CPU")
    
    def run(self, record: ExperimentRecord) -> Dict[str, Any]:
        """Run a single experiment. Returns result dict."""
        start_time = time.time()
        
        # Prepare log file
        log_file = self.output_dir / f"{record.exp_id}.log"
        
        # Build command
        cmd = self._build_command(record)
        
        # Set device
        env = os.environ.copy()
        if self.use_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        
        # Update record
        record.device = f"cuda:{self.gpu_id}" if self.use_gpu else "cpu"
        
        try:
            # Run subprocess
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"# Experiment: {record.exp_id}\n")
                f.write(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Command: {' '.join(cmd)}\n")
                f.write("=" * 50 + "\n\n")
                f.flush()
                
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=3600  # 1 hour timeout
                )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                # Extract metrics from log
                metrics = self._extract_metrics(log_file)
                return {
                    "success": True,
                    "metrics": metrics,
                    "duration": duration,
                    "log_file": str(log_file)
                }
            else:
                return {
                    "success": False,
                    "error": f"Process exited with code {result.returncode}",
                    "duration": duration,
                    "log_file": str(log_file)
                }
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                "success": False,
                "error": "Timeout after 1 hour",
                "duration": duration,
                "log_file": str(log_file)
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "duration": duration,
                "log_file": str(log_file)
            }
    
    def _build_command(self, record: ExperimentRecord) -> list:
        """Build command line for the experiment.
        
        Loop order: seed(outer) → model(middle) → method(inner)
        """
        # Map model names to config names
        model_config_map = {
            "mlp": "paper_mlp",
            "lstm": "paper_lstm",
            "gru": "paper_gru",
            "cnn1d": "paper_cnn1d"
        }
        
        model_config = model_config_map.get(record.model, f"paper_{record.model}")
        
        # 获取缺失率列表（20档：0.0-0.95，步长0.05）
        eval_mrs = record.eval_missing_rates if record.eval_missing_rates else "0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95"
        mim_train_mrs = record.mim_train_missing_rates if record.mim_train_missing_rates else "0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95"
        
        # 使用默认 config.yaml 作为基础，通过覆盖参数来配置
        cmd = [
            "python", "src/main.py",
            f"model={model_config}",           # 模型架构
            f"method={record.method}",          # 方法（最内层循环）
            f"experiment.seeds=[{record.seed}]", # 随机种子（最外层循环）
            f"+experiment.run_name={record.exp_id}",
            f"+missing.missing_rates_eval=[{eval_mrs}]",       # 10档测试MR
            f"+missing.missing_rates_train=[{mim_train_mrs}]", # MIM训练MR（合并）
        ]
        
        return cmd
    
    def _extract_metrics(self, log_file: Path) -> Dict[str, Any]:
        """Extract all MR results from log file by parsing the result lines."""
        metrics = {}
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for all result lines: "MR=X.X: MAE=X.XXXX"
            pattern = r"MR=([\d.]+):\s*MAE=([\d.]+)"
            matches = re.findall(pattern, content)
            
            if matches:
                # Store all MR results as a list
                mr_results = []
                for mr_str, mae_str in matches:
                    mr_results.append({
                        'missing_rate': float(mr_str),
                        'test_mae': float(mae_str)
                    })
                
                # Store the full list
                metrics['mr_results'] = mr_results
                
                # Also store summary statistics
                mae_values = [r['test_mae'] for r in mr_results]
                metrics['test_mae_mean'] = sum(mae_values) / len(mae_values)
                metrics['test_mae_max'] = max(mae_values)
                metrics['test_mae_min'] = min(mae_values)
                
                # Store the last MR result as primary metric for backward compatibility
                metrics['test_mae'] = mr_results[-1]['test_mae']
                metrics['missing_rate'] = mr_results[-1]['missing_rate']
            
        except Exception as e:
            print(f"Warning: Could not extract metrics from {log_file}: {e}")
        
        return metrics


class CPUExperimentRunner(ExperimentRunner):
    """CPU-only runner for parallel CPU execution."""
    
    def __init__(self, output_dir: str = "experiments/logs"):
        super().__init__(output_dir=output_dir, use_gpu=False)


class GPUExperimentRunner(ExperimentRunner):
    """GPU runner with memory monitoring."""
    
    def __init__(self, gpu_id: int = 0, output_dir: str = "experiments/logs"):
        super().__init__(output_dir=output_dir, use_gpu=True, gpu_id=gpu_id)
    
    def get_gpu_memory(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not self.use_gpu:
            return {"allocated": 0, "reserved": 0, "total": 0}
        
        torch.cuda.set_device(self.gpu_id)
        return {
            "allocated": torch.cuda.memory_allocated(self.gpu_id) / 1e9,
            "reserved": torch.cuda.memory_reserved(self.gpu_id) / 1e9,
            "total": torch.cuda.get_device_properties(self.gpu_id).total_memory / 1e9
        }
    
    def is_memory_available(self, required_gb: float = 2.0) -> bool:
        """Check if enough GPU memory is available."""
        if not self.use_gpu:
            return True
        
        mem = self.get_gpu_memory()
        available = mem["total"] - mem["allocated"]
        return available >= required_gb
