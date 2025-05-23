import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm

# CSV 파일 로드
# df1 = pd.read_csv("./data/UM_Internal_0620_-_Form_-_1.001.csv")
# df2 = pd.read_csv("./data/UM_Internal_0620_-_MicroForm_-_21.022.csv")
df1 = pd.read_csv("./cleaned_dataset/data/00001.csv")
df2 = pd.read_csv("./cleaned_dataset/data/00003.csv")

# 데이터 샘플 확인
print(df1.head())
print(df2.head())

# 데이터 정보 및 결측치 확인
print(df1.info(), df2.info())
print(df1.isnull().sum(), df2.isnull().sum())

# 예제: 특정 피처 선택
selected_features = ["Voltage_measured","Current_measured","Temperature_measured","Current_load","Voltage_load","Time"]  # 데이터셋에 맞춰 수정 필요
df_selected = df1["Current (A)"]

# 딥러닝 모델 입력 형태로 변환
input_tensor = torch.tensor(df_selected.values, dtype=torch.float32)
print(input_tensor.shape)  # (샘플 수, 피처 수)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_selected)

# PyTorch Tensor 변환
input_tensor = torch.tensor(scaled_data, dtype=torch.float32)


# 예제: 모델 인스턴스 생성
deepsc = DeepSC(num_layers=4, src_vocab_size=100, trg_vocab_size=100,
                 src_max_len=50, trg_max_len=50, d_model=128, num_heads=8, dff=512)

# 데이터 형태 맞추기 (배치 차원 추가)
input_tensor = input_tensor.unsqueeze(0)  # (1, 시퀀스 길이, d_model)

# 모델에 입력
output = deepsc(input_tensor)

print(output.shape)  # (1, 시퀀스 길이, trg_vocab_size)
