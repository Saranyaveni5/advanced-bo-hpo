# multiclass_cnn_baseline.py
# Simple CNN baseline (placeholder) - creates small synthetic dataset and trains for 1 epoch
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, pandas as pd, os
os.makedirs('bo_outputs', exist_ok=True)
# tiny synthetic data
num_classes = 10; N = 500
X = np.random.randn(N, 3, 32, 32).astype('float32')
y = np.random.randint(0, num_classes, size=(N,))
split = int(0.8 * N)
train_ds = TensorDataset(torch.tensor(X[:split]), torch.tensor(y[:split]))
val_ds = TensorDataset(torch.tensor(X[split:]), torch.tensor(y[split:]))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*8*8, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleCNN(num_classes).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3); loss_fn = nn.CrossEntropyLoss()
# one epoch for placeholder
model.train()
for xb, yb in train_loader:
    xb, yb = xb.to(device), yb.to(device)
    opt.zero_grad(); out = model(xb); loss = loss_fn(out, yb); loss.backward(); opt.step()
# validation
model.eval(); correct=0; total=0
with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).argmax(dim=1)
        correct += (preds==yb).sum().item(); total += yb.size(0)
acc = correct/total
pd.DataFrame({'val_accuracy':[acc]}).to_csv('bo_outputs/validation_scores.csv', index=False)
torch.save(model.state_dict(), 'bo_outputs/best_model_baseline.pth')
print('Baseline done, val_acc=', acc)
