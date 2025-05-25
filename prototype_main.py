import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def visualize_reconstruction(original, restored, sample_idx):
    """original/restored: [T, D] np.array"""
    plt.figure(figsize=(12, 6))
    num_features = original.shape[1]
    for i in range(num_features):
        plt.subplot(num_features, 1, i + 1)
        plt.plot(original[:, i], label='Original', linestyle='--')
        plt.plot(restored[:, i], label='Restored', alpha=0.7)
        plt.title(f'Feature {i}')
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig(f'./reconstructed/compare_sample{sample_idx}.png')
    plt.close()

def add_gaussian_noise(tensor, std=0.05):
    noise = torch.randn_like(tensor) * std
    return tensor + noise

def randomly_mask_segments(tensor, drop_ratio=0.1):
    B, T, D = tensor.shape
    mask = torch.ones_like(tensor)
    for b in range(B):
        start = torch.randint(0, T - int(T * drop_ratio), (1,))
        end = start + int(T * drop_ratio)
        mask[b, start:end, :] = 0
    return tensor * mask

if __name__ == '__main__':
    import os
    import pandas as pd
    from preprocess_time_series import load_pt_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSCForTimeSeries(input_dim=6, d_model=128).to(device)

    train_loader = load_pt_dataset('./preprocessed_data/train_data.pt', batch_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # ✅ 학습 루프
    model.train()
    num_epochs = 5
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x = batch[0].to(device)

            # 🔹 노이즈 삽입 (가우시안 또는 마스킹 중 택 1)
            x_noised = add_gaussian_noise(x)
            # x_noised = randomly_mask_segments(x)  # 필요시 교체 가능

            optimizer.zero_grad()
            output = model(x_noised)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    # ✅ 복원 결과 저장 및 시각화
    model.eval()
    with torch.no_grad():
        os.makedirs('./reconstructed', exist_ok=True)
        count = 0
        for batch in train_loader:
            x = batch[0].to(device)
            output = model(x)
            for b in range(min(3 - count, x.size(0))):
                original = x[b].cpu().numpy()   # [T, D]
                restored = output[b].cpu().numpy()

                df = pd.DataFrame(restored, columns=[f'feature_{j}' for j in range(restored.shape[1])])
                df.to_csv(f'./reconstructed/restored_sample{count}.csv', index=False)

                visualize_reconstruction(original, restored, count)

                count += 1
                if count >= 3:
                    break
            if count >= 3:
                break
        print("✅ 복원된 샘플 3개 저장 및 시각화 완료")