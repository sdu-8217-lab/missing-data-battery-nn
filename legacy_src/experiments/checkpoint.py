"""检查点管理模块 - 支持实验中断恢复"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class CheckpointManager:
    """管理实验检查点，支持断点续跑"""
    
    def __init__(self, experiment_dir: Path, total_seeds: int):
        self.experiment_dir = Path(experiment_dir)
        self.checkpoint_file = self.experiment_dir / "checkpoint.json"
        self.total_seeds = total_seeds
        
        # 确保目录存在
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
    def load_checkpoint(self) -> Optional[Dict]:
        """加载检查点，如果不存在返回None"""
        if not self.checkpoint_file.exists():
            return None
        
        with open(self.checkpoint_file, 'r') as f:
            return json.load(f)
    
    def create_checkpoint(self, batch_name: str, seeds: List[int]) -> Dict:
        """创建新的检查点"""
        checkpoint = {
            "batch_name": batch_name,
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
            "total_seeds": len(seeds),
            "seeds": seeds,
            "completed_seeds": [],
            "failed_seeds": [],
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "status": "running"
        }
        self._save_checkpoint(checkpoint)
        return checkpoint
    
    def update_checkpoint(self, seed: int, status: str = "completed"):
        """更新检查点状态"""
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return
        
        if status == "completed":
            if seed not in checkpoint["completed_seeds"]:
                checkpoint["completed_seeds"].append(seed)
        elif status == "failed":
            if seed not in checkpoint["failed_seeds"]:
                checkpoint["failed_seeds"].append(seed)
        
        checkpoint["last_update"] = datetime.now().isoformat()
        self._save_checkpoint(checkpoint)
    
    def finalize_checkpoint(self):
        """标记实验完成"""
        checkpoint = self.load_checkpoint()
        if checkpoint:
            checkpoint["status"] = "completed"
            checkpoint["last_update"] = datetime.now().isoformat()
            self._save_checkpoint(checkpoint)
    
    def get_remaining_seeds(self) -> List[int]:
        """获取未完成的种子列表"""
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return []
        
        all_seeds = set(checkpoint["seeds"])
        completed = set(checkpoint["completed_seeds"])
        failed = set(checkpoint["failed_seeds"])
        
        return sorted(list(all_seeds - completed))
    
    def get_completed_seeds(self) -> List[int]:
        """获取已完成的种子列表"""
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return []
        return checkpoint.get("completed_seeds", [])
    
    def get_progress(self) -> Dict:
        """获取进度信息"""
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return {"completed": 0, "total": self.total_seeds, "percentage": 0}
        
        completed = len(checkpoint.get("completed_seeds", []))
        total = checkpoint.get("total_seeds", self.total_seeds)
        
        return {
            "completed": completed,
            "total": total,
            "percentage": (completed / total * 100) if total > 0 else 0,
            "status": checkpoint.get("status", "unknown")
        }
    
    def _save_checkpoint(self, checkpoint: Dict):
        """保存检查点到文件"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def seed_dir_exists(self, seed: int) -> bool:
        """检查特定种子的结果目录是否存在"""
        seed_dir = self.experiment_dir / f"seed_{seed}"
        result_file = seed_dir / "results.csv"
        return result_file.exists()
