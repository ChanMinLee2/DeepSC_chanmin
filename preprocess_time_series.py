import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# CSV 파일이 예상한 컬럼들을 모두 포함하고 있는지 확인
def is_valid_csv(fpath, expected_columns):
    try:
        df = pd.read_csv(fpath, nrows=1)
        return all(col in df.columns for col in expected_columns)
    except:
        return False

# 저장된 pt 파일에서 TensorDataset 불러오기
def load_pt_dataset(pt_path, batch_size):
    data = torch.load(pt_path)
    print(f"✅ Loaded {data.shape[0]} samples from {pt_path}")
    return DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=False)

# 폴더 내 모든 유효한 CSV 파일을 로드하고 패딩 후 텐서로 변환 (모든 시퀀스를 max 길이에 맞춤)
def load_all_valid_csv_tensors(folder_path, feature_cols, batch_size=8, save_split_path=None, split_ratio=0.8):
    tensors = []
    lengths = []
    scaler = StandardScaler()
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    total_files = len(files)
    valid_files = 0
    for fname in tqdm(files, desc="Processing CSV files"):
        fpath = os.path.join(folder_path, fname)
        if not is_valid_csv(fpath, feature_cols):
            continue
        try:
            df = pd.read_csv(fpath)
            data = df[feature_cols].values.astype(np.float32)
            data = scaler.fit_transform(data)
            tensor = torch.tensor(data, dtype=torch.float32)  # [T, D]
            tensors.append(tensor)
            lengths.append(tensor.shape[0])
            valid_files += 1
        except Exception as e:
            print(f"[WARN] Failed to process {fname}: {e}")

    print(f"✅ Valid CSV files used: {valid_files} / {total_files}")

    # 유효한 텐서가 하나도 없으면 None 반환
    if not tensors:
        return None

    # max 길이에 맞춰 오른쪽 padding 적용
    max_len = max(lengths)
    padded_tensors = []
    for t in tensors:
        pad_len = max_len - t.shape[0]
        if pad_len > 0:
            pad = (0, 0, 0, pad_len)  # pad last dimension
            t = F.pad(t, pad, mode='constant', value=0)
        padded_tensors.append(t.unsqueeze(0))  # [1, T, D]

    full_tensor = torch.cat(padded_tensors, dim=0)  # [N, T, D]

    # 학습/테스트 분할 및 저장 (.pt 파일로)
    if save_split_path:
        N = full_tensor.shape[0]
        train_len = int(N * split_ratio)
        test_len = N - train_len
        train_data = full_tensor[:train_len]
        test_data = full_tensor[train_len:]
        torch.save(train_data, os.path.join(save_split_path, 'train_data.pt'))
        torch.save(test_data, os.path.join(save_split_path, 'test_data.pt'))
        print(f"✅ Saved train_data.pt ({train_len} samples), test_data.pt ({test_len} samples) to {save_split_path}")

    dataset = TensorDataset(full_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

# 단일 DataFrame에서 지정된 피처들을 정규화하고 DeepSC 입력 형태의 텐서로 변환
def prepare_tensor_for_deepsc(data, feature_cols):
    features = data[feature_cols].values.astype(np.float32)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # [1, T, D]
    return tensor

# 슬라이딩 윈도우를 기반으로 시계열 데이터를 [B, T, D] 형태의 시퀀스로 분할
def normalize_and_window(data, feature_cols, seq_len=30, stride=10):
    features = data[feature_cols].values.astype(np.float32)
    features = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0) + 1e-8)

    sequences = []
    for i in range(0, len(features) - seq_len + 1, stride):
        seq = features[i:i+seq_len]
        sequences.append(seq)

    return np.stack(sequences)  # [B, T, D]

# DeepSC 모델에 배치 입력을 넣어 인코더-디코더를 통과시켜 출력 반환
def run_deepsc_forward(deepsc_model, dataloader, device):
    deepsc_model.eval()
    outputs = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)  # [B, T, D]
            B, T, D = x.shape
            x_input = x.view(B * T, D)  # 시간 차원 펼치기
            encoded = deepsc_model.channel_encoder(x_input)  # [B*T, encoded_dim]
            decoded = deepsc_model.channel_decoder(encoded)  # [B*T, D_model]
            decoded = decoded.view(B, T, -1)  # 원래 [B, T, D] 형태로 복원
            outputs.append(decoded)
    return outputs



feature_cols = [
    'Voltage_measured', 'Current_measured', 'Temperature_measured',
    'Current_load', 'Voltage_load', 'Time'
]

load_all_valid_csv_tensors(
    folder_path="./cleaned_dataset/data",
    feature_cols=feature_cols,
    batch_size=8,
    save_split_path="./preprocessed_data",  # 이 경로에 pkl 저장됨
    split_ratio=0.8
)

# train_loader = load_pt_dataset('./preprocessed_data/train_data.pt', batch_size=8)
# test_loader = load_pt_dataset('./preprocessed_data/test_data.pt', batch_size=8)

# for batch in train_loader:
    # print(batch[0].shape)
    # break
