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
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.gpu_id = gpu_id if self.use_gpu else None
        
        # Check GPU
        if self.use_gpu:
            print(f"GPU available: {torch.cuda.get_device_name(self.gpu_id)}")
            print(f"   Memory: {torch.cuda.get_device_properties(self.gpu_id).total_memory / 1e9:.1f} GB")
        else:
            print("GPU not available, using CPU")
    
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
        """Build command line for the experiment."""
        # Map model names to config names
        model_config_map = {
            "mlp": "paper_mlp",
            "lstm": "paper_lstm",
            "gru": "paper_gru",
            "cnn1d": "paper_cnn1d"
        }
        
        model_config = model_config_map.get(record.model, f"paper_{record.model}")
        
        # Use generic run config based on method
        run_config = f"paper/run_{record.method}"
        
        cmd = [
            "python", "src/main.py",
            f"--config-name={run_config}",  # Use run config as base
            f"model={model_config}",  # Override model
            f"experiment.seeds=[{record.seed}]",  # Override seeds to single value
            f"experiment.run_name={record.exp_id}",
            f"missing_rates=[{record.mr}]",  # Override missing rates to single value
        ]
        
        # Note: trainer config is not exposed in main.py
        # GPU is auto-detected in main.py
        
        return cmd
    
    def _extract_metrics(self, log_file: Path) -> Dict[str, float]:
        """Extract final metrics from log file by parsing the result line."""
        metrics = {}
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for the result line: "MR=X.X: MAE=X.XXXX"
            pattern = r"MR=([\d.]+):\s*MAE=([\d.]+)"
            matches = re.findall(pattern, content)
            
            if matches:
                # Get the last match (should be the test result)
                mr_str, mae_str = matches[-1]
                metrics['test_mae'] = float(mae_str)
                metrics['missing_rate'] = float(mr_str)
            
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
