import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

df = pd.read_csv('data.csv', parse_dates=['datetime'])

cutoff_date = pd.Timestamp('2004-03-01 00:15:00')
df_before = df[df['datetime'] < cutoff_date]
df_after = df[df['datetime'] >= cutoff_date]

df_used = df  

groups = df_used.groupby('moteid')

def create_sequences(data, seq_length, overlap):
    """
    Create sequences with given length and overlap.
    Each sequence's label is determined by the first non-zero 'type'
    within that window. If all 'type' values are zero, the label is 0.
    """
    sequences = []
    labels = []
    step = seq_length - overlap  # window moves by this step
    for i in range(0, len(data) - seq_length + 1, step):
        seq = data.iloc[i:i+seq_length]
        sequences.append(seq['temperature'].values)
        types_in_seq = seq['type'].values
        non_zero = types_in_seq[types_in_seq != 0]
        label = non_zero[0] if len(non_zero) > 0 else 0
        labels.append(label)
    return np.array(sequences), np.array(labels)

sequence_length = 5   
overlap = 2          

all_sequences = []
all_labels = []

for mote, group in groups:
    group = group.sort_values('datetime')  # ensure chronological order
    seq, lbl = create_sequences(group, sequence_length, overlap)
    if len(seq) > 0:
        all_sequences.append(seq)
        all_labels.append(lbl)

X = np.concatenate(all_sequences, axis=0)  
y = np.concatenate(all_labels, axis=0) 

X = X.reshape((X.shape[0], X.shape[1], 1))

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TimeSeriesDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

input_dim = 1
hidden_dim = 64
num_layers = 1
num_epochs = 10
learning_rate = 0.001
num_classes = 6

model = LSTMClassifier(input_dim, hidden_dim, num_layers, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)  
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_X.size(0)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    epoch_loss /= total
    accuracy = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
