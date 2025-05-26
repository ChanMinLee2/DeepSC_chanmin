import torch
import torch.nn as nn
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from preprocess_time_series import load_pt_dataset
import joblib

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class DeepSCForTimeSeries(nn.Module):
    def __init__(self, input_dim, compressed_dim=16, d_model=8, num_heads=4, num_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_ff, 
            dropout=dropout, 
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.channel_encoder = nn.Sequential(
            nn.Linear(d_model, 64), 
            nn.ReLU(), 
            nn.Linear(64, compressed_dim)
        )
        self.channel_decoder = nn.Sequential(
            nn.Linear(compressed_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, d_model)
        )
        self.output_fc = nn.Linear(d_model, input_dim)
        self.compressed_dim = compressed_dim
        self.d_model = d_model

    def forward(self, x):
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.channel_encoder(x)
        x = self.simulate_channel(x)
        x = self.channel_decoder(x)
        x = self.output_fc(x)
        return x

    def simulate_channel(self, x):
        if self.channel_mode == 'awgn':
            return x + torch.randn_like(x) * self.noise_std
        elif self.channel_mode == 'rayleigh':
            fading = torch.randn_like(x) / math.sqrt(2)
            return x * fading + torch.randn_like(x) * self.noise_std
        else:
            raise ValueError(f"Unsupported channel mode: {self.channel_mode}")

    def set_channel_mode(self, mode='awgn', noise_std=0.05):
        self.channel_mode = mode
        self.noise_std = noise_std

def visualize_reconstruction(original, restored, sample_idx):
    plt.figure(figsize=(12, 6))
    num_features = original.shape[1]
    for i in range(num_features):
        plt.subplot(num_features, 1, i + 1)
        plt.plot(original[:, i], label='Original', linestyle='--')
        plt.plot(restored[:, i], label='Restored', alpha=0.7)
        if i == 0:
            plt.legend()
        plt.title(f'Feature {i}')
    plt.tight_layout()
    plt.savefig(f'./reconstructed/compare_sample{sample_idx}.png')
    plt.close()

def train(model, dataloader, device, num_epochs=5):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x = batch[0].to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    scaler = joblib.load('./preprocessed_data/scaler.pkl')
    os.makedirs('./reconstructed', exist_ok=True)
    count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            x = batch[0].to(device)
            output = model(x)
            loss = criterion(output, x)
            total_loss += loss.item()
            for b in range(x.size(0)):
                if count >= 3:
                    break
                restored = output[b].cpu().numpy()
                restored_original = scaler.inverse_transform(restored)
                df = pd.DataFrame(restored_original, columns=[f'feature_{j}' for j in range(restored.shape[1])])
                df.to_csv(f'./reconstructed/restored_sample{count}.csv', index=False)
                original = x[b].cpu().numpy()
                visualize_reconstruction(original, restored, count)
                count += 1
    print(f"✅ Evaluation Complete | 🔁 MSE: {total_loss:.4f} | 📉 Compression Ratio: {model.compressed_dim/model.d_model:.2f}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSCForTimeSeries(input_dim=6, compressed_dim=16, d_model=8).to(device)
    model.set_channel_mode(mode='awgn', noise_std=0.05)
    train_loader = load_pt_dataset('./preprocessed_data/train_data.pt', batch_size=8)
    test_loader = load_pt_dataset('./preprocessed_data/test_data.pt', batch_size=8)
    train(model, train_loader, device, num_epochs=30)
    evaluate(model, test_loader, device)
