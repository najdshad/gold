import torch
import torch.nn as nn
import torch.optim as optim

class NTT(nn.Module):
    def __init__(self, input_dim=4, d_model=64, nhead=4, num_layers=4, dropout=0.1):
        super(NTT, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model))

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 classes: long, short, no trade
        )

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # Take the last time step's output
        return self.classifier(x)

# Hyperparameters
input_dim = 4  # OHLC
sequence_length = 50
d_model = 64
nhead = 4
num_layers = 4
dropout = 0.1

# Model
model = NTT(input_dim, d_model, nhead, num_layers, dropout)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)
