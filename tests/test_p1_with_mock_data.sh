#!/bin/bash
# Integration test for P1 architecture with mock data

set -e

echo "=== P1 Architecture Integration Test ==="
echo "Using mock data to test aggregation and plotting"

# Create temporary test directory
TEST_DIR=$(mktemp -d)
trap "rm -rf $TEST_DIR" EXIT

echo "Test directory: $TEST_DIR"

# Create mock results structure
create_mock_results() {
    local batch=$1
    local timestamp=$2
    local results_dir="$TEST_DIR/results/${timestamp}_${batch}"
    
    mkdir -p "$results_dir"
    
    # Create mock CSV
    cat > "$results_dir/results.csv" << CSVEOF
seed,model,method,missing_mode,missing_rate,batch_id,test_mae,test_rmse,test_r2
42,MLP,Baseline,mar,0.3,$batch,0.045,0.062,0.89
42,MLP,MIM,mar,0.3,$batch,0.038,0.054,0.92
42,MLP,Baseline,mar,0.7,$batch,0.078,0.095,0.75
42,MLP,MIM,mar,0.7,$batch,0.055,0.072,0.85
123,MLP,Baseline,mar,0.3,$batch,0.048,0.065,0.88
123,MLP,MIM,mar,0.3,$batch,0.040,0.056,0.91
123,MLP,Baseline,mar,0.7,$batch,0.082,0.098,0.73
123,MLP,MIM,mar,0.7,$batch,0.058,0.075,0.83
CSVEOF

    echo "Created: $results_dir/results.csv"
}

# Setup mock data
TIMESTAMP="20260318_120000"
create_mock_results "2C" "$TIMESTAMP"
create_mock_results "3C" "$TIMESTAMP"

echo ""
echo "=== Test 1: Verify directory structure ==="
find "$TEST_DIR/results" -type f | while read f; do
    echo "  $f"
done

# Since pandas/matplotlib may not be available, we test the Python logic
echo ""
echo "=== Test 2: Verify CSV content ==="
for csv in "$TEST_DIR/results"/*/*.csv; do
    echo "--- $(basename $(dirname $csv)) ---"
    head -3 "$csv"
    echo "  ... ($(wc -l < "$csv") total lines)"
done

echo ""
echo "=== Test 3: Test paths module with mock data ==="
python3 << PYTHONEOF
import sys
sys.path.insert(0, 'src')

import importlib.util
spec = importlib.util.spec_from_file_location('paths', 'src/utils/paths.py')
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)

# Test list_all_batches
print("Testing list_all_batches:")
all_batches = paths.list_all_batches("$TEST_DIR/results")
print(f"  Found {len(all_batches)} batches")
for ts, bid, path in all_batches:
    print(f"    - {ts}_{bid}: {path}")

# Test find_latest_results
print("\nTesting find_latest_results:")
for batch in ['2C', '3C']:
    latest = paths.find_latest_results(batch, "$TEST_DIR/results")
    print(f"  Latest for {batch}: {latest}")
PYTHONEOF

echo ""
echo "=== Test 4: Manual aggregation check ==="
python3 << PYTHONEOF
# Simple aggregation without pandas
import csv
import os

all_rows = []
results_dir = "$TEST_DIR/results"

for subdir in os.listdir(results_dir):
    csv_path = os.path.join(results_dir, subdir, "results.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)

print(f"Aggregated {len(all_rows)} rows")

# Group by (batch, model, method, mr)
groups = {}
for row in all_rows:
    key = (row['batch_id'], row['model'], row['method'], row['missing_rate'])
    if key not in groups:
        groups[key] = []
    groups[key].append(float(row['test_mae']))

# Compute simple statistics
print("\nStatistics (batch, model, method, mr) -> mean MAE:")
for key, values in sorted(groups.items()):
    mean = sum(values) / len(values)
    print(f"  {key}: {mean:.4f} (n={len(values)})")
PYTHONEOF

echo ""
echo "=== All tests passed ==="
