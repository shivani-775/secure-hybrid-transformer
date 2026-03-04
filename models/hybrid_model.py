import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.cnn = nn.Conv1d(input_dim, 32, kernel_size=3)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=32,
                nhead=4,
                batch_first=True
            ),
            num_layers=2
        )

        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (batch, features, seq)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)   # (batch, seq, features)
        x = self.transformer(x)
        x = x[:, -1, :]          # last time step
        x = self.fc(x)

        return x   # 🚨 NO SIGMOID