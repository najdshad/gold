import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

class OHLCDataset(Dataset):
    def __init__(self, data, sequence_length=50):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.sequence_length, :-1]  # OHLC features
        y = self.data[idx+self.sequence_length, -1]       # Target order_type
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

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

# Load data (example CSV)
df = pd.read_csv('Processed_Data/1h_3r_10usd_new.csv')
data = df[['open', 'high', 'low', 'close', 'order_type']].values

# Split data into train, validation, and test sets
train_data, temp_data = train_test_split(data, test_size=0.3, shuffle=False)
val_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

# Create datasets and dataloaders
train_dataset = OHLCDataset(train_data)
val_dataset = OHLCDataset(val_data)
test_dataset = OHLCDataset(test_data)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop
train_losses = []
val_losses = []

for epoch in range(10):
    model.train()
    for x_batch, y_batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    train_losses.append(loss.item())

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_val, y_val in val_dataloader:
            val_outputs = model(x_val)
            val_loss += criterion(val_outputs, y_val).item()

    val_loss /= len(val_dataloader)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'ntt_model.pth')
print("Model saved to ntt_model.pth")

# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
