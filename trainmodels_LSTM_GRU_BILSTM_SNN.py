# ============================================================
# FAST FLUX DETECTION USING RNN & SNN (END-TO-END PIPELINE)
# ============================================================

# ----------------------------
# IMPORTS
# ----------------------------
import os, re, math
import pandas as pd
import numpy as np
from collections import Counter
from statistics import mean, stdev

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import snntorch as snn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Fast Flux"],
        yticklabels=["Benign", "Fast Flux"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ----------------------------
# DATASET PATHS (CHANGE IF NEEDED)
# ----------------------------
LEGIT_FILE = "C:/Users/09shi/Desktop/UIA LEARNING MATERIAL/FYP/newdataset/archive2/Fastflux Attack Dataset/benign"
FASTFLUX_FILE = "C:/Users/09shi/Desktop/UIA LEARNING MATERIAL/FYP/newdataset/archive2/Fastflux Attack Dataset/ff"

LABEL_FILES = [
    (LEGIT_FILE, 0),      # benign
    (FASTFLUX_FILE, 1)    # fast flux
]

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def shannon_entropy(items):
    counts = Counter(items)
    total = sum(counts.values())
    return -sum((c/total) * math.log2(c/total) for c in counts.values()) if total else 0

def extract_features(dig_text):
    a_records = re.findall(r'IN\s+A\s+(\d+\.\d+\.\d+\.\d+)', dig_text)
    ttl_vals = list(map(int, re.findall(r'(\d+)\s+IN\s+A\s+\d+\.\d+\.\d+\.\d+', dig_text)))
    cname_records = re.findall(r'IN\s+CNAME\s+(\S+)', dig_text)
    ns_records = re.findall(r'IN\s+NS\s+(\S+)', dig_text)

    subnets = {'.'.join(ip.split('.')[:3]) for ip in a_records}

    return {
        "num_A_records": len(a_records),
        "ttl_min": min(ttl_vals) if ttl_vals else 0,
        "ttl_max": max(ttl_vals) if ttl_vals else 0,
        "ttl_avg": mean(ttl_vals) if ttl_vals else 0,
        "ttl_stddev": stdev(ttl_vals) if len(ttl_vals) > 1 else 0,
        "num_CNAME_records": len(cname_records),
        "num_NS_records": len(ns_records),
        "ip_entropy": shannon_entropy(a_records),
        "num_unique_subnets": len(subnets)
    }

# ============================================================
# LOAD FAST FLUX DATASET
# ============================================================
records = []

for folder, label in LABEL_FILES:
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), encoding="utf-8", errors="ignore") as f:
                content = f.read()
                entries = re.split(r'(?=; <<>> DiG)', content)
                for entry in entries:
                    if entry.strip():
                        feats = extract_features(entry)
                        feats["label"] = label
                        records.append(feats)

df = pd.DataFrame(records)
print("Dataset loaded:", df.shape)
print(df["label"].value_counts())

X = df.drop("label", axis=1)
y = df["label"]

# ============================================================
# PREPARE DATA FOR RNN
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

SEQ_LEN = 5

def make_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return torch.stack(xs), torch.tensor(ys)

X_seq, y_seq = make_sequences(X_tensor, y_tensor, SEQ_LEN)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_seq, y_seq, test_size=0.3, stratify=y_seq, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# ============================================================
# EARLY STOPPING
# ============================================================
class EarlyStopping:
    def __init__(self, patience=5):
        self.best = float("inf")
        self.counter = 0
        self.patience = patience
        self.stop = False

    def step(self, val_loss):
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

# ============================================================
# RNN MODELS
# ============================================================
class LSTM(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True)
        self.fc = nn.Linear(h, 2)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

class GRU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.gru = nn.GRU(d, h, batch_first=True)
        self.fc = nn.Linear(h, 2)
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

class BiLSTM(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(h*2, 2)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(torch.cat((h[-2], h[-1]), dim=1))

# ============================================================
# RNN TRAINING
# ============================================================
def train_rnn(model, epochs=50):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    early = EarlyStopping()

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += loss_fn(model(xb.to(device)), yb.to(device)).item()
        val_loss /= len(val_loader)

        print(f"Epoch {ep+1}: Val Loss={val_loss:.4f}")
        early.step(val_loss)
        if early.stop:
            break

    return model

# ============================================================
# TRAIN & TEST RNNs
# ============================================================
models = {
    "LSTM": LSTM(X.shape[1], 64),
    "GRU": GRU(X.shape[1], 64),
    "BiLSTM": BiLSTM(X.shape[1], 64)
}

for name, model in models.items():
    print(f"\nTraining {name}")
    model = train_rnn(model)
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(device)).argmax(dim=1).cpu()
            y_true.extend(yb.numpy())
            y_pred.extend(preds.numpy())
    print(name)
    print(classification_report(y_true, y_pred))
    plot_confusion_matrix(
        y_true,
        y_pred,
        title=f"{name} Confusion Matrix"
    )

# ============================================================
# SNN
# ============================================================
num_steps = 20
X_spikes = X_tensor.unsqueeze(1).repeat(1, num_steps, 1)

class SNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 100)
        self.lif1 = snn.Leaky(beta=0.95)
        self.fc2 = nn.Linear(100, 2)
        self.lif2 = snn.Leaky(beta=0.95)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        for t in range(x.size(1)):
            spk1, mem1 = self.lif1(self.fc1(x[:, t]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
        return mem2

print("\nTraining SNN")
snn_model = SNN(X.shape[1]).to(device)
opt = torch.optim.Adam(snn_model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
early = EarlyStopping()

for ep in range(50):
    opt.zero_grad()
    loss = loss_fn(snn_model(X_spikes.to(device)), y_tensor.to(device))
    loss.backward()
    opt.step()
    print(f"SNN Epoch {ep+1}: Loss={loss.item():.4f}")
    early.step(loss.item())
    if early.stop:
        break

with torch.no_grad():
    snn_preds = snn_model(X_spikes.to(device)).argmax(dim=1).cpu()

print("SNN Classification Report")
print(classification_report(y_tensor.numpy(), snn_preds.numpy()))

# 🔥 Confusion Matrix
plot_confusion_matrix(
    y_tensor.numpy(),
    snn_preds.numpy(),
    title="SNN Confusion Matrix"
)


