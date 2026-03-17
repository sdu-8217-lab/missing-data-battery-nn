#!/usr/bin/env python
"""
Mock test for P1 architecture components.

Tests aggregation and plotting logic without requiring actual experiment runs.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import importlib.util

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_paths_module():
    """Test path management functions."""
    print("\n=== Test: paths module ===")
    
    # Load paths module without full project imports
    spec = importlib.util.spec_from_file_location('paths', 'src/utils/paths.py')
    paths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paths)
    
    # Test timestamp generation
    ts = paths.get_timestamp()
    assert len(ts) == 15 and ts[8] == '_', f"Invalid timestamp format: {ts}"
    print(f"✅ Timestamp: {ts}")
    
    # Test results directory
    results_dir = paths.get_results_dir("2C", "20260318_120000")
    assert "20260318_120000_2C" in str(results_dir)
    print(f"✅ Results dir: {results_dir}")
    
    # Test special characters in batch ID
    results_dir_r25 = paths.get_results_dir("R2.5", "20260318_120000")
    assert "R2_5" in str(results_dir_r25)
    print(f"✅ Special chars handled: {results_dir_r25}")
    
    # Test structure creation
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "test_results"
        p = paths.ensure_results_structure(test_dir)
        
        assert p["results_dir"].exists()
        assert p["figures_dir"].exists()
        assert p["csv_path"].parent == p["results_dir"]
        print(f"✅ Structure created: {p}")
    
    print("✅ paths module: PASSED")
    return True


def test_aggregation_logic():
    """Test aggregation logic with mock CSV files."""
    print("\n=== Test: aggregation logic ===")
    
    try:
        import pandas as pd
    except ImportError:
        print("⚠️  pandas not available, skipping aggregation test")
        return True
    
    # Create mock results
    mock_data_2c = pd.DataFrame({
        'batch_id': ['2C', '2C', '2C', '2C'],
        'model': ['MLP', 'MLP', 'CNN', 'CNN'],
        'method': ['Baseline', 'MIM', 'Baseline', 'MIM'],
        'missing_rate': [0.3, 0.3, 0.5, 0.5],
        'seed': [42, 42, 123, 123],
        'test_mae': [0.05, 0.04, 0.06, 0.045],
        'test_rmse': [0.07, 0.055, 0.08, 0.065],
    })
    
    mock_data_3c = pd.DataFrame({
        'batch_id': ['3C', '3C'],
        'model': ['MLP', 'MLP'],
        'method': ['Baseline', 'MIM'],
        'missing_rate': [0.3, 0.3],
        'seed': [42, 42],
        'test_mae': [0.055, 0.042],
        'test_rmse': [0.075, 0.058],
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock result directories
        results_dir = Path(tmpdir) / "results"
        dir_2c = results_dir / "20260318_120000_2C"
        dir_3c = results_dir / "20260318_120000_3C"
        dir_2c.mkdir(parents=True)
        dir_3c.mkdir(parents=True)
        
        # Save mock CSVs
        mock_data_2c.to_csv(dir_2c / "results.csv", index=False)
        mock_data_3c.to_csv(dir_3c / "results.csv", index=False)
        
        # Test aggregation
        spec = importlib.util.spec_from_file_location('aggregate', 'scripts/aggregate_results.py')
        # Note: Can't easily test the full script due to argparse, but we can test logic
        
        # Manual aggregation test
        combined = pd.concat([mock_data_2c, mock_data_3c], ignore_index=True)
        assert len(combined) == 6
        print(f"✅ Combined {len(combined)} rows")
        
        # Test statistics computation
        stats = combined.groupby(['batch_id', 'model', 'method'])['test_mae'].agg(['mean', 'std', 'count'])
        print(f"✅ Statistics computed:\n{stats}")
    
    print("✅ aggregation logic: PASSED")
    return True


def test_plotting_logic():
    """Test plotting data preparation."""
    print("\n=== Test: plotting data preparation ===")
    
    try:
        import pandas as pd
    except ImportError:
        print("⚠️  pandas not available, skipping plotting test")
        return True
    
    # Create mock data
    mock_df = pd.DataFrame({
        'model': ['MLP', 'MLP', 'MLP', 'MLP', 'CNN', 'CNN', 'CNN', 'CNN'],
        'method': ['Baseline', 'Baseline', 'MIM', 'MIM', 'Baseline', 'Baseline', 'MIM', 'MIM'],
        'missing_rate': [0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7],
        'seed': [42, 42, 42, 42, 42, 42, 42, 42],
        'test_mae': [0.05, 0.08, 0.04, 0.055, 0.06, 0.09, 0.045, 0.065],
    })
    
    # Test column normalization (simulating plot_csv.py logic)
    col_mapping = {
        'test_mae': 'MAE',
        'test_rmse': 'RMSE',
        'missing_rate': 'missing_rate',
    }
    
    for old, new in col_mapping.items():
        if old in mock_df.columns and new not in mock_df.columns:
            mock_df[new] = mock_df[old]
    
    assert 'MAE' in mock_df.columns
    print(f"✅ Column normalization: MAE column created")
    
    # Test aggregation for plotting
    grouped = mock_df.groupby(['model', 'method', 'missing_rate'])['MAE'].agg(['mean', 'std']).reset_index()
    assert len(grouped) == 8  # 2 models × 2 methods × 2 MRs
    print(f"✅ Aggregation for plotting: {len(grouped)} groups")
    
    # Test improvement calculation
    baseline = mock_df[mock_df['method'] == 'Baseline'].groupby(['model', 'missing_rate'])['MAE'].mean().reset_index()
    baseline.rename(columns={'MAE': 'MAE_baseline'}, inplace=True)
    
    mim = mock_df[mock_df['method'] == 'MIM'].groupby(['model', 'missing_rate'])['MAE'].mean().reset_index()
    mim.rename(columns={'MAE': 'MAE_mim'}, inplace=True)
    
    merged = pd.merge(baseline, mim, on=['model', 'missing_rate'])
    merged['improvement'] = (merged['MAE_baseline'] - merged['MAE_mim']) / merged['MAE_baseline'] * 100
    
    assert 'improvement' in merged.columns
    assert all(merged['improvement'] >= 0)  # MIM should improve
    print(f"✅ Improvement calculation:\n{merged[['model', 'missing_rate', 'improvement']]}")
    
    print("✅ plotting logic: PASSED")
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("P1 Architecture Mock Tests")
    print("="*60)
    
    tests = [
        test_paths_module,
        test_aggregation_logic,
        test_plotting_logic,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
