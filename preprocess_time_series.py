import pandas as pd
import numpy as np
import os
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from torch.utils.data import TensorDataset
import joblib

def is_valid_csv(fpath, expected_columns):
    try:
        df = pd.read_csv(fpath, nrows=1)
        return all(col in df.columns for col in expected_columns)
    except:
        return False

def load_pt_dataset(pt_path, batch_size):
    data = torch.load(pt_path)
    print(f"✅ Loaded {data.tensors[0].shape[0]} samples from {pt_path}")
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

def sliding_window_sequences(data, window_size=128, stride=64):
    sequences = []
    for start in range(0, len(data) - window_size + 1, stride):
        window = data[start:start + window_size]
        sequences.append(window)
    return np.stack(sequences)  # [N, T, D]

def load_all_valid_csv_tensors(folder_path, feature_cols, batch_size=8, save_split_path=None, split_ratio=0.8, window_size=128, stride=64):
    scaler = MinMaxScaler()
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    total_files = len(files)
    valid_files = 0
    all_windows = []
    for fname in tqdm(files, desc="Processing CSV files"):
        fpath = os.path.join(folder_path, fname)
        if not is_valid_csv(fpath, feature_cols):
            continue
        try:
            df = pd.read_csv(fpath)
            data = df[feature_cols].values.astype(np.float32)
            data = scaler.fit_transform(data)
            windows = sliding_window_sequences(data, window_size, stride)  # [N, T, D]
            all_windows.append(torch.tensor(windows, dtype=torch.float32))
            valid_files += 1
        except Exception as e:
            print(f"[WARN] Failed to process {fname}: {e}")

    print(f"✅ Valid CSV files used: {valid_files} / {total_files}")

    if not all_windows:
        print("[ERROR] No valid data windows found.")
        return

    full_tensor = torch.cat(all_windows, dim=0)  # [Total_N, T, D]

    if save_split_path:
        N = full_tensor.shape[0]
        train_len = int(N * split_ratio)
        train_data = TensorDataset(full_tensor[:train_len])
        test_data = TensorDataset(full_tensor[train_len:])
        torch.save(train_data, os.path.join(save_split_path, 'train_data.pt'))
        torch.save(test_data, os.path.join(save_split_path, 'test_data.pt'))
        print(f"✅ Saved train_data.pt ({train_len} samples), test_data.pt ({N - train_len} samples) to {save_split_path}")

        scaler_path = os.path.join(save_split_path, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"✅ Saved scaler to {scaler_path}")

if __name__ == '__main__':
    feature_cols = [
        'Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time'
    ]

    load_all_valid_csv_tensors(
        folder_path="./cleaned_dataset/data",
        feature_cols=feature_cols,
        batch_size=8,
        save_split_path="./preprocessed_data",
        split_ratio=0.8
    )