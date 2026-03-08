"""Experiment database - CSV-based tracking for paper reproduction."""

import csv
import time
import threading
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum


class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentRecord:
    """Single experiment run record.
    
    循环层级（从内到外重要性递增）:
    - 最内层（核心）: method [baseline/mim]
    - 中间层: model [mlp/lstm/gru/cnn1d]  
    - 最外层: seed [42-141]
    
    一个run = (seed, model, method)，测试所有eval_missing_rates
    metrics 字段存储所有测试结果
    """
    exp_id: str
    seed: int           # outer loop
    model: str          # middle loop  
    method: str         # inner loop (core)
    status: str = "pending"
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metrics: str = ""  # JSON string: {mr_results: [{mr: 0.0, mae: 0.05}, ...], summary: {...}}
    log_file: Optional[str] = None
    error_message: Optional[str] = None
    device: Optional[str] = None
    duration_seconds: Optional[float] = None
    eval_missing_rates: str = ""      # 测试缺失率列表 "0.0,0.1,...,0.9"
    mim_train_missing_rates: str = "" # MIM训练缺失率列表（合并训练）
    timestamp: str = ""               # 实验时间戳 YYYYMMDD_HHMMSS
    run_dir: str = ""                 # 实验运行目录
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentRecord':
        # Handle missing fields for backward compatibility
        defaults = {
            'status': 'pending',
            'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'started_at': None,
            'completed_at': None,
            'metrics': '',
            'log_file': None,
            'error_message': None,
            'device': None,
            'duration_seconds': None,
        }
        defaults.update(data)
        return cls(**defaults)


class ExperimentDatabase:
    """CSV-based experiment database with thread-safe operations."""
    
    CSV_FIELDS = [
        'exp_id', 'seed', 'model', 'method', 'status',
        'created_at', 'started_at', 'completed_at',
        'metrics', 'log_file', 'error_message',
        'device', 'duration_seconds',
        'eval_missing_rates', 'mim_train_missing_rates',
        'timestamp', 'run_dir'
    ]
    
    def __init__(self, db_path: str = "experiments/experiment_db.csv"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
        # Initialize CSV with headers if not exists
        if not self.db_path.exists():
            self._write_all([])
    
    def _read_all(self) -> List[Dict[str, Any]]:
        """Read all records from CSV."""
        if not self.db_path.exists():
            return []
        
        with open(self.db_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    def _write_all(self, records: List[Dict[str, Any]]):
        """Write all records to CSV atomically."""
        temp_path = self.db_path.with_suffix('.tmp')
        with open(temp_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            writer.writerows(records)
        temp_path.replace(self.db_path)
    
    def add_experiment(self, record: ExperimentRecord) -> bool:
        """Add a new experiment. Returns False if already exists."""
        with self._lock:
            records = self._read_all()
            
            # Check if exists
            if any(r['exp_id'] == record.exp_id for r in records):
                return False
            
            records.append(record.to_dict())
            self._write_all(records)
            return True
    
    def get_experiment(self, exp_id: str) -> Optional[ExperimentRecord]:
        """Get experiment by ID."""
        records = self._read_all()
        for r in records:
            if r['exp_id'] == exp_id:
                return ExperimentRecord.from_dict(r)
        return None
    
    def update_experiment(self, exp_id: str, updates: Dict[str, Any]) -> bool:
        """Update experiment fields."""
        with self._lock:
            records = self._read_all()
            
            for i, r in enumerate(records):
                if r['exp_id'] == exp_id:
                    records[i].update(updates)
                    self._write_all(records)
                    return True
            return False
    
    def get_by_status(self, status: ExperimentStatus) -> List[ExperimentRecord]:
        """Get all experiments with given status."""
        records = self._read_all()
        return [
            ExperimentRecord.from_dict(r)
            for r in records
            if r.get('status') == status.value
        ]
    
    def get_pending_batch(self, batch_size: int = 10) -> List[ExperimentRecord]:
        """Get a batch of pending experiments (atomically marks as running)."""
        with self._lock:
            records = self._read_all()
            
            # Find pending records
            pending_indices = [
                i for i, r in enumerate(records)
                if r.get('status') == ExperimentStatus.PENDING.value
            ]
            
            batch_indices = pending_indices[:batch_size]
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            
            result = []
            for i in batch_indices:
                records[i]['status'] = ExperimentStatus.RUNNING.value
                records[i]['started_at'] = now
                result.append(ExperimentRecord.from_dict(records[i]))
            
            if batch_indices:
                self._write_all(records)
            
            return result
    
    def mark_completed(self, exp_id: str, metrics: Dict[str, Any], 
                       duration: float, log_file: str):
        """Mark experiment as completed with results.
        
        Args:
            metrics: Dictionary containing 'mr_results' list and summary stats
        """
        import json
        self.update_experiment(exp_id, {
            "status": ExperimentStatus.COMPLETED.value,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": json.dumps(metrics),  # Includes mr_results list
            "duration_seconds": duration,
            "log_file": log_file
        })
    
    def mark_failed(self, exp_id: str, error: str, log_file: str):
        """Mark experiment as failed."""
        self.update_experiment(exp_id, {
            "status": ExperimentStatus.FAILED.value,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error_message": error,
            "log_file": log_file
        })
    
    def reset_running(self) -> int:
        """Reset all running experiments to pending (for recovery). Returns count."""
        with self._lock:
            records = self._read_all()
            count = 0
            
            for r in records:
                if r.get('status') == ExperimentStatus.RUNNING.value:
                    r['status'] = ExperimentStatus.PENDING.value
                    r['started_at'] = None
                    count += 1
            
            if count:
                self._write_all(records)
            return count
    
    def get_statistics(self) -> Dict[str, int]:
        """Get count of experiments by status."""
        records = self._read_all()
        stats = {status.value: 0 for status in ExperimentStatus}
        
        for r in records:
            status = r.get('status', 'unknown')
            stats[status] = stats.get(status, 0) + 1
        
        stats['total'] = len(records)
        return stats
    
    def get_all_experiments(self) -> List[ExperimentRecord]:
        """Get all experiments."""
        records = self._read_all()
        return [ExperimentRecord.from_dict(r) for r in records]
    
    def generate_experiment_id(self, seed: int, model: str, method: str) -> str:
        """Generate standardized experiment ID reflecting loop nesting.
        
        Loop order: seed(outer) → model(middle) → method(inner)
        """
        return f"seed{seed}_{model}_{method}"
