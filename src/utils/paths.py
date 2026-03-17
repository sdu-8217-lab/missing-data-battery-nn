"""
Path management for experiment results.

Provides centralized path management for batch-specific experiments.
Directory structure: results/{timestamp}_{batch}/
"""

from pathlib import Path
from datetime import datetime
from typing import Optional


def get_timestamp() -> str:
    """Generate timestamp string: YYYYMMDD_HHMMSS"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_results_dir(batch_id: str, timestamp: Optional[str] = None, 
                   base_dir: str = "results") -> Path:
    """
    Get results directory for a specific batch.
    
    Args:
        batch_id: Batch identifier (e.g., "2C", "3C", "R2.5", "R3", "RW", "Sim_satellite")
        timestamp: Optional timestamp (default: current time)
        base_dir: Base results directory
    
    Returns:
        Path to results directory: results/{timestamp}_{batch_id}/
    """
    if timestamp is None:
        timestamp = get_timestamp()
    
    # Normalize batch_id for filesystem (replace special chars)
    safe_batch = batch_id.replace(".", "_").replace("-", "_")
    dir_name = f"{timestamp}_{safe_batch}"
    
    return Path(base_dir) / dir_name


def ensure_results_structure(results_dir: Path) -> dict:
    """
    Ensure results directory structure exists.
    
    Creates:
        - results_dir/
        - results_dir/figures/
    
    Returns:
        Dict with paths
    """
    figures_dir = results_dir / "figures"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "results_dir": results_dir,
        "figures_dir": figures_dir,
        "csv_path": results_dir / "results.csv",
    }


def find_latest_results(batch_id: str, base_dir: str = "results") -> Optional[Path]:
    """
    Find the latest results directory for a batch.
    
    Args:
        batch_id: Batch identifier
        base_dir: Base results directory
    
    Returns:
        Path to latest results directory or None
    """
    safe_batch = batch_id.replace(".", "_").replace("-", "_")
    base = Path(base_dir)
    
    if not base.exists():
        return None
    
    # Find all directories matching pattern *_{batch_id}
    matching_dirs = [
        d for d in base.iterdir() 
        if d.is_dir() and d.name.endswith(f"_{safe_batch}")
    ]
    
    if not matching_dirs:
        return None
    
    # Sort by timestamp (directory name) and return latest
    return sorted(matching_dirs)[-1]


def list_all_batches(base_dir: str = "results") -> list:
    """
    List all batch result directories.
    
    Returns:
        List of (timestamp, batch_id, path) tuples
    """
    base = Path(base_dir)
    
    if not base.exists():
        return []
    
    batches = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        
        # Parse directory name: {timestamp}_{batch_id}
        parts = d.name.rsplit("_", 1)
        if len(parts) == 2:
            timestamp, batch_id = parts
            batches.append((timestamp, batch_id, d))
    
    return sorted(batches)
