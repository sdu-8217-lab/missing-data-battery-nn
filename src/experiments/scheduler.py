"""Experiment scheduler - manages 7200 experiments with parallel execution."""

import time
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import torch

from .database import ExperimentDatabase, ExperimentRecord, ExperimentStatus
from .runner import ExperimentRunner, GPUExperimentRunner, CPUExperimentRunner


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""
    # Parallel execution
    gpu_workers: int = 2  # Number of GPU processes (limited by VRAM)
    cpu_workers: int = 2  # Number of CPU processes
    
    # GPU settings
    use_gpu: bool = True
    
    # Retry settings - auto retry failed experiments
    max_retries: int = 3
    retry_delay_seconds: int = 5
    
    # Progress reporting
    report_interval_seconds: int = 60
    
    # Paths
    db_path: str = "experiments/experiment_db.csv"
    log_dir: str = "experiments/logs"
    
    @property
    def total_workers(self) -> int:
        return self.gpu_workers + self.cpu_workers


def _run_experiment_worker(record_dict: Dict, worker_id: int, 
                           gpu_workers: int, use_gpu: bool, log_dir: str) -> Dict[str, Any]:
    """
    Worker function to run a single experiment.
    This runs in a separate process - no database access here!
    """
    record = ExperimentRecord(**record_dict)
    
    # Determine if this worker should use GPU
    has_gpu = use_gpu and torch.cuda.is_available()
    use_gpu_for_this = worker_id < gpu_workers and has_gpu
    gpu_id = worker_id if use_gpu_for_this else None
    
    # Create runner
    if use_gpu_for_this:
        runner = GPUExperimentRunner(gpu_id=gpu_id, output_dir=log_dir)
    else:
        runner = CPUExperimentRunner(output_dir=log_dir)
    
    device_str = 'GPU' if use_gpu_for_this else 'CPU'
    print(f"  [{worker_id}] Running {record.exp_id} on {device_str}")
    
    # Run experiment
    result = runner.run(record)
    
    # Return result (database update happens in main process)
    return {
        "exp_id": record.exp_id,
        "success": result["success"],
        "metrics": result.get("metrics", {}),
        "duration": result["duration"],
        "log_file": result["log_file"],
        "error": result.get("error", ""),
    }


class ExperimentScheduler:
    """Manages and executes batches of experiments."""
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self.db = ExperimentDatabase(self.config.db_path)
        self._shutdown_requested = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Check GPU availability
        if self.config.use_gpu and torch.cuda.is_available():
            self.gpu_available = True
            self.num_gpus = torch.cuda.device_count()
            print(f"Found {self.num_gpus} GPU(s)")
            for i in range(self.num_gpus):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        else:
            self.gpu_available = False
            self.config.gpu_workers = 0
            print("GPU not available, using CPU only")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self._shutdown_requested = True
    
    def initialize_experiments(self, 
                              models: List[str],
                              methods: List[str],
                              missing_rates: List[float],
                              seeds: List[int],
                              clear_existing: bool = False) -> int:
        """Initialize experiment database with all combinations."""
        if clear_existing and Path(self.config.db_path).exists():
            print(f"Clearing existing database: {self.config.db_path}")
            Path(self.config.db_path).unlink()
            self.db = ExperimentDatabase(self.config.db_path)
        
        count = 0
        for model in models:
            for method in methods:
                for mr in missing_rates:
                    for seed in seeds:
                        exp_id = self.db.generate_experiment_id(model, method, mr, seed)
                        record = ExperimentRecord(
                            exp_id=exp_id,
                            model=model,
                            method=method,
                            mr=mr,
                            seed=seed
                        )
                        if self.db.add_experiment(record):
                            count += 1
        
        print(f"Initialized {count} experiments")
        return count
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status summary."""
        stats = self.db.get_statistics()
        pending = stats.get("pending", 0)
        running = stats.get("running", 0)
        completed = stats.get("completed", 0)
        failed = stats.get("failed", 0)
        total = stats.get("total", 0)
        
        progress = (completed / total * 100) if total > 0 else 0
        
        return {
            "total": total,
            "pending": pending,
            "running": running,
            "completed": completed,
            "failed": failed,
            "progress_percent": progress,
            "estimated_remaining": pending + running
        }
    
    def print_status(self):
        """Print current status."""
        status = self.get_status()
        print(f"\nStatus: {status['completed']}/{status['total']} "
              f"({status['progress_percent']:.1f}%) | "
              f"Pending: {status['pending']} | "
              f"Running: {status['running']} | "
              f"Failed: {status['failed']}")
    
    def recover(self) -> int:
        """Reset stuck running experiments to pending."""
        count = self.db.reset_running()
        if count > 0:
            print(f"Recovered {count} stuck experiments")
        return count
    
    def _update_experiment_result(self, result: Dict[str, Any]):
        """Update experiment status based on worker result (main process only)."""
        if result["success"]:
            self.db.mark_completed(
                result["exp_id"],
                result["metrics"],
                result["duration"],
                result["log_file"]
            )
            print(f"  [OK] {result['exp_id']} completed ({result['duration']:.1f}s)")
        else:
            self.db.mark_failed(result["exp_id"], result["error"], result["log_file"])
            print(f"  [FAIL] {result['exp_id']}: {result['error']}")
    
    def run_batch(self, batch_size: Optional[int] = None):
        """Run a batch of pending experiments with auto-retry."""
        if batch_size is None:
            batch_size = self.config.total_workers
        
        # Get pending experiments
        pending = self.db.get_by_status(ExperimentStatus.PENDING)
        if not pending:
            print("No pending experiments")
            return 0
        
        batch = pending[:batch_size]
        print(f"\nRunning batch of {len(batch)} experiments "
              f"({self.config.gpu_workers} GPU + {self.config.cpu_workers} CPU)")
        
        # Track retry counts
        retry_counts = {record.exp_id: 0 for record in batch}
        failed_in_batch = set()
        
        # Run with auto-retry
        for attempt in range(self.config.max_retries + 1):
            if self._shutdown_requested:
                print("Shutdown requested, stopping...")
                break
            
            # Get experiments to run in this attempt
            if attempt == 0:
                to_run = batch
            else:
                # Retry failed experiments
                to_run = [self.db.get_experiment(exp_id) for exp_id in failed_in_batch]
                to_run = [r for r in to_run if r is not None]
                if not to_run:
                    break
                print(f"\nRetry attempt {attempt}/{self.config.max_retries}: "
                      f"{len(to_run)} experiments")
                time.sleep(self.config.retry_delay_seconds)
            
            failed_in_batch = set()
            
            # Run in parallel
            with ProcessPoolExecutor(max_workers=self.config.total_workers) as executor:
                futures = {}
                for i, record in enumerate(to_run):
                    worker_id = i % self.config.total_workers
                    future = executor.submit(
                        _run_experiment_worker,
                        record.to_dict(),
                        worker_id,
                        self.config.gpu_workers,
                        self.config.use_gpu,
                        self.config.log_dir
                    )
                    futures[future] = record
                
                for future in as_completed(futures):
                    if self._shutdown_requested:
                        break
                    
                    record = futures[future]
                    try:
                        result = future.result()
                        self._update_experiment_result(result)
                        
                        if not result["success"]:
                            retry_counts[record.exp_id] += 1
                            if retry_counts[record.exp_id] < self.config.max_retries:
                                failed_in_batch.add(record.exp_id)
                            
                    except Exception as e:
                        print(f"  [EXCEPTION] {record.exp_id}: {e}")
                        self.db.mark_failed(record.exp_id, str(e), "")
                        retry_counts[record.exp_id] += 1
                        if retry_counts[record.exp_id] < self.config.max_retries:
                            failed_in_batch.add(record.exp_id)
        
        return len(batch) - len(failed_in_batch)
    
    def run_all(self, continuous: bool = True):
        """Run all pending experiments."""
        print(f"\n{'='*50}")
        print(f"Starting Experiment Scheduler")
        print(f"   GPU workers: {self.config.gpu_workers}")
        print(f"   CPU workers: {self.config.cpu_workers}")
        print(f"   Max retries: {self.config.max_retries}")
        print(f"{'='*50}\n")
        
        self.print_status()
        
        last_report = time.time()
        total_completed = 0
        
        while True:
            if self._shutdown_requested:
                print("\nShutdown requested, exiting...")
                break
            
            # Check if there are pending experiments
            pending_count = len(self.db.get_by_status(ExperimentStatus.PENDING))
            if pending_count == 0:
                print("\nAll experiments completed!")
                break
            
            # Run a batch
            completed = self.run_batch()
            total_completed += completed
            
            # Report progress
            current_time = time.time()
            if current_time - last_report > self.config.report_interval_seconds:
                self.print_status()
                last_report = current_time
            
            # If not continuous, stop after one batch
            if not continuous:
                break
            
            # Small delay between batches
            time.sleep(1)
        
        # Final status
        self.print_status()
        print(f"\nTotal experiments completed this session: {total_completed}")
        
        return total_completed
    
    def retry_failed(self) -> int:
        """Reset failed experiments to pending for manual retry."""
        failed = self.db.get_by_status(ExperimentStatus.FAILED)
        count = 0
        for record in failed:
            self.db.update_experiment(record.exp_id, {
                "status": ExperimentStatus.PENDING.value,
                "error_message": None
            })
            count += 1
        
        print(f"Reset {count} failed experiments to pending")
        return count
