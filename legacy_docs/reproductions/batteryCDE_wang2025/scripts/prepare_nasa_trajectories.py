"""从 NASA .mat 提取 BatteryCDE 所需的充电轨迹。

每个电池输出一个 `.npz` 文件，包含：
- trajectories: [n_charge_cycles, seq_len, 3]  （电压、电流、温度）
- capacities: [n_charge_cycles]  对应放电容量
- cycle_indices: [n_charge_cycles]  原始循环编号

输出路径：reproductions/batteryCDE_wang2025/data/nasa_trajectories/
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d


def resample_trajectory(
    t: np.ndarray,
    v: np.ndarray,
    i: np.ndarray,
    temp: np.ndarray,
    target_len: int = 100,
) -> np.ndarray:
    """将变长充电轨迹重采样为固定长度 [target_len, 3]。"""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    i = np.asarray(i, dtype=float)
    temp = np.asarray(temp, dtype=float)

    # 按时间排序并去重
    order = np.argsort(t)
    t, v, i, temp = t[order], v[order], i[order], temp[order]
    uniq = np.concatenate(([True], np.diff(t) > 1e-9))
    t, v, i, temp = t[uniq], v[uniq], i[uniq], temp[uniq]

    if len(t) < 2:
        return None

    # 归一化到 [0, 1] 的时间轴
    t_norm = (t - t[0]) / (t[-1] - t[0] + 1e-9)
    t_new = np.linspace(0, 1, target_len)

    try:
        v_new = interp1d(t_norm, v, kind="linear", fill_value="extrapolate")(t_new)
        i_new = interp1d(t_norm, i, kind="linear", fill_value="extrapolate")(t_new)
        temp_new = interp1d(t_norm, temp, kind="linear", fill_value="extrapolate")(t_new)
    except ValueError:
        return None

    return np.stack([v_new, i_new, temp_new], axis=1).astype(np.float32)


def extract_battery_trajectories(
    mat_path: str,
    target_len: int = 100,
    min_charge_points: int = 10,
) -> dict:
    """从单个 .mat 文件提取充电轨迹与对应容量。"""
    mat = sio.loadmat(mat_path)

    # 顶层键为电池名，如 B0005
    top_keys = [k for k in mat.keys() if not k.startswith("__")]
    if len(top_keys) != 1:
        raise ValueError(f"Expected one battery key in {mat_path}, got {top_keys}")
    batt_key = top_keys[0]
    batt = mat[batt_key][0, 0]
    cycles = batt["cycle"]

    trajectories = []
    capacities = []
    cycle_indices = []

    for idx in range(cycles.size):
        c = cycles[0, idx]
        ctype = str(c["type"][0])
        if ctype != "charge":
            continue

        data = c["data"][0, 0]
        names = set(data.dtype.names or [])
        if not {"Voltage_measured", "Current_measured", "Temperature_measured", "Time"}.issubset(names):
            continue

        v = data["Voltage_measured"][0, :]
        i = data["Current_measured"][0, :]
        temp = data["Temperature_measured"][0, :]
        t = data["Time"][0, :]

        if len(v) < min_charge_points:
            continue

        traj = resample_trajectory(t, v, i, temp, target_len=target_len)
        if traj is None:
            continue

        # 找下一个 discharge cycle 的容量
        capacity = np.nan
        for j in range(idx + 1, cycles.size):
            c_next = cycles[0, j]
            if str(c_next["type"][0]) == "discharge":
                data_next = c_next["data"][0, 0]
                if "Capacity" in data_next.dtype.names:
                    cap_arr = data_next["Capacity"]
                    if cap_arr.size == 1:
                        capacity = float(cap_arr[0, 0])
                    elif cap_arr.size > 0:
                        capacity = float(cap_arr.flat[0])
                break

        if not np.isfinite(capacity):
            continue

        trajectories.append(traj)
        capacities.append(capacity)
        cycle_indices.append(idx)

    if len(trajectories) == 0:
        return None

    return {
        "trajectories": np.stack(trajectories, axis=0),
        "capacities": np.array(capacities, dtype=np.float32),
        "cycle_indices": np.array(cycle_indices, dtype=np.int32),
    }


def main():
    parser = argparse.ArgumentParser(description="提取 NASA 充电轨迹用于 BatteryCDE")
    parser.add_argument("--data_dir", type=str, default="../../data/NASA data",
                        help="NASA .mat 文件目录")
    parser.add_argument("--output_dir", type=str, default="../data/nasa_trajectories",
                        help="输出目录")
    parser.add_argument("--target_len", type=int, default=100,
                        help="每条充电轨迹重采样后的长度")
    parser.add_argument("--batteries", type=str, default=None,
                        help="逗号分隔的电池编号，如 5,6,7,18；默认处理所有 B*.mat")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batteries:
        battery_ids = [b.strip() for b in args.batteries.split(",")]
        mat_files = []
        for bid in battery_ids:
            candidates = list(data_dir.glob(f"B{bid.zfill(4)}.mat")) + list(data_dir.glob(f"B{bid}.mat"))
            mat_files.extend(candidates)
        mat_files = sorted(set(mat_files))
    else:
        mat_files = sorted(data_dir.glob("B*.mat"))

    summary = []
    for mat_path in mat_files:
        print(f"Processing {mat_path.name} ...")
        try:
            result = extract_battery_trajectories(str(mat_path), target_len=args.target_len)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        if result is None:
            print(f"  No valid charge cycles found.")
            continue

        out_path = output_dir / mat_path.with_suffix(".npz").name
        np.savez_compressed(out_path, **result)
        print(f"  Saved {out_path}: {result['trajectories'].shape}")
        summary.append({
            "battery": mat_path.stem,
            "n_cycles": result["trajectories"].shape[0],
            "capacity_mean": float(result["capacities"].mean()),
            "capacity_std": float(result["capacities"].std()),
        })

    if summary:
        print("\nSummary:")
        for row in summary:
            print(f"  {row['battery']}: n_cycles={row['n_cycles']}, "
                  f"capacity={row['capacity_mean']:.3f}±{row['capacity_std']:.3f}")


if __name__ == "__main__":
    main()
