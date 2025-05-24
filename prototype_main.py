import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: [B, T, D]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class DeepSCForTimeSeries(nn.Module):
    def __init__(self, input_dim, d_model=128, num_heads=4, num_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.channel_encoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )

        self.channel_decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )

        self.output_fc = nn.Linear(d_model, input_dim)

    def forward(self, x):  # x: [B, T, D]
        x = self.input_fc(x)           # [B, T, d_model]
        x = self.pos_encoder(x)        # [B, T, d_model]
        x = self.encoder(x)            # [B, T, d_model]
        x = self.channel_encoder(x)    # [B, T, d_model]
        x = self.channel_decoder(x)    # [B, T, d_model]
        x = self.output_fc(x)          # [B, T, D]
        return x

# ✅ 학습 및 복원 예시 루프
if __name__ == '__main__':
    import os
    import pandas as pd
    from preprocess_time_series import load_pt_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSCForTimeSeries(input_dim=6, d_model=128).to(device)

    train_loader = load_pt_dataset('./preprocessed_data/train_data.pt', batch_size=8)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader, desc="Reconstructing CSV batches")):
            x = batch[0].to(device)  # [B, T, D]
            output = model(x)        # 복원 결과

            # ✅ 복원된 배치 첫 샘플을 CSV로 저장
            restored = output[0].cpu().numpy()  # [T, D]
            df = pd.DataFrame(restored, columns=[f'feature_{j}' for j in range(restored.shape[1])])
            os.makedirs('./reconstructed', exist_ok=True)
            df.to_csv(f'./reconstructed/restored_batch{i}_sample0.csv', index=False)

            if i >= 2: break  # 예시로 3개만 저장
