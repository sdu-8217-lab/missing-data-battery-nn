"""Prototype richer NASA feature extraction from .mat raw curves."""
import sys
from pathlib import Path
import numpy as np
import scipy.io
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DATA_DIR = Path('data/NASA data')


def integrate(y, t):
    """梯形积分 ∫ y dt"""
    if len(y) < 2:
        return 0.0
    dt = np.diff(t)
    y_mid = (y[:-1] + y[1:]) / 2.0
    return float(np.sum(y_mid * dt))


def extract_charge_features(cycle):
    """从 charge 循环提取特征"""
    d = cycle.data
    t = np.asarray(d.Time).ravel()
    v = np.asarray(d.Voltage_measured).ravel()
    i = np.asarray(d.Current_measured).ravel()
    temp = np.asarray(d.Temperature_measured).ravel()
    v_charge = np.asarray(d.Voltage_charge).ravel()
    i_charge = np.asarray(d.Current_charge).ravel()

    # 只取电流为正（充电）的区段做主要特征
    pos_mask = i > 0.05
    if pos_mask.sum() < 5:
        return None

    t0 = t[pos_mask][0]
    t1 = t[pos_mask][-1]
    duration = float(t1 - t0)

    # 充电电量（正电流积分）
    q_charge = integrate(np.maximum(i, 0), t)

    # CC/CV 划分：基于参考电流/电压
    v_max_ref = np.percentile(v_charge, 99)
    i_cc_mode = np.percentile(i_charge[i_charge > 0.05], 90) if np.any(i_charge > 0.05) else 1.0
    if i_cc_mode <= 0:
        i_cc_mode = 1.0

    cc_mask = (i_charge > 0.7 * i_cc_mode) & (v_charge < 0.98 * v_max_ref)
    cv_mask = (v_charge > 0.95 * v_max_ref) & (i_charge < 0.5 * i_cc_mode)

    cc_time = float(np.sum(cc_mask) * np.median(np.diff(t))) if cc_mask.sum() else 0.0
    cv_time = float(np.sum(cv_mask) * np.median(np.diff(t))) if cv_mask.sum() else 0.0

    return {
        'charge_duration': duration,
        'charge_Q': q_charge,
        'charge_voltage_mean': float(np.mean(v[pos_mask])),
        'charge_voltage_max': float(np.max(v[pos_mask])),
        'charge_current_mean': float(np.mean(i[pos_mask])),
        'charge_current_std': float(np.std(i[pos_mask])),
        'charge_temp_mean': float(np.mean(temp[pos_mask])),
        'charge_temp_max': float(np.max(temp[pos_mask])),
        'charge_cc_time': cc_time,
        'charge_cv_time': cv_time,
    }


def extract_discharge_features(cycle):
    """从 discharge 循环提取目标与特征"""
    d = cycle.data
    t = np.asarray(d.Time).ravel()
    v = np.asarray(d.Voltage_measured).ravel()
    i = np.asarray(d.Current_measured).ravel()
    temp = np.asarray(d.Temperature_measured).ravel()

    # 只取明显放电区段（电流负且电压下降）
    dis_mask = i < -0.05
    if dis_mask.sum() < 5:
        return None

    capacity = -integrate(np.minimum(i, 0), t) / 3600.0  # 放电容量 (Ah)
    duration = float(t[dis_mask][-1] - t[dis_mask][0])

    return {
        'capacity': capacity,
        'discharge_duration': duration,
        'discharge_voltage_min': float(np.min(v[dis_mask])),
        'discharge_voltage_mean': float(np.mean(v[dis_mask])),
        'discharge_voltage_std': float(np.std(v[dis_mask])),
        'discharge_current_mean': float(np.mean(i[dis_mask])),
        'discharge_temp_mean': float(np.mean(temp[dis_mask])),
        'discharge_temp_max': float(np.max(temp[dis_mask])),
    }


def extract_battery(file_path):
    mat = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    key = [k for k in mat.keys() if not k.startswith('__')][0]
    batt = mat[key]
    cycles = batt.cycle

    records = []
    last_charge_features = None
    last_capacity = None
    cycle_num = 0

    for idx in range(len(cycles)):
        c = cycles[idx]
        if c.type == 'charge':
            cf = extract_charge_features(c)
            if cf is not None:
                last_charge_features = cf
        elif c.type == 'discharge':
            df = extract_discharge_features(c)
            if df is None:
                continue
            if last_charge_features is None:
                continue
            rec = {
                'cycle_num': cycle_num,
                'cycle_index': idx,
                'prev_capacity': last_capacity if last_capacity is not None else df['capacity'],
            }
            rec.update(last_charge_features)
            rec.update(df)
            records.append(rec)
            last_capacity = df['capacity']
            cycle_num += 1
        # impedance ignored

    if not records:
        return None
    return pd.DataFrame(records)


def main():
    files = sorted(DATA_DIR.glob('B*.mat'))
    stats = []
    for f in files:
        df = extract_battery(f)
        if df is None or len(df) == 0:
            print(f"{f.name}: no valid discharge cycles")
            continue
        # 基本过滤：容量必须在合理范围
        cap = df['capacity']
        valid = (cap > 0.5) & (cap < 5.0)
        n_valid = valid.sum()
        stats.append({
            'file': f.name,
            'n_discharge': len(df),
            'capacity_min': cap.min(),
            'capacity_max': cap.max(),
            'capacity_mean': cap.mean(),
            'n_valid': n_valid,
        })
        print(f"{f.name}: {len(df)} discharge, cap [{cap.min():.3f}, {cap.max():.3f}], valid {n_valid}")

    stats_df = pd.DataFrame(stats)
    print('\nSummary:')
    print(stats_df.to_string(index=False))
    print(f"\nTotal valid discharge cycles: {stats_df['n_valid'].sum()}")


if __name__ == '__main__':
    main()
