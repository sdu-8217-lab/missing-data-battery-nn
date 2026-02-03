import sys
sys.path.insert(0, 'src')
from data.dataset_loader import XJTUDatasetLoader

loader = XJTUDatasetLoader('./data/XJTU data', '3C')

try:
    data = loader.prepare_data(
        ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
         'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
         'current mean', 'current std', 'current kurtosis', 'current skewness',
         'CV Q', 'CV charge time', 'current slope', 'current entropy'],
        'capacity',
        test_size=0.25,
        val_size=0.25,
        random_seed=42
    )
    print('Success!')
    print(f"Train: {data['X_train'].shape}")
    print(f"Val: {data['X_val'].shape}")
    print(f"Test: {data['X_test'].shape}")
    print(f"Battery info: {data['battery_info']}")
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
