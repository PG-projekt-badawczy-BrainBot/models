import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from MSHCNN import MSHCNN
from torch.utils.data import Dataset, DataLoader
from loader.loader import load_patient_dataset
from sklearn.metrics import confusion_matrix


default_batch_size = 32

class EEGDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, T, C)
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (N, C, T)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train():
    file_path = os.path.join("../../", 'processed_eeg_data', 'S001')
    print('Loading dataset')
    X_train, X_test, y_train, y_test = load_patient_dataset(
        file_path, window_offset=128, window_size=256, even_classes=True)


    classes, counts = np.unique(y_train, return_counts=True)
    print(f"Classes: {classes}")
    print(f"Counts: {counts}")

    selected = [4, 3, 2, 1, 5, 6]  # Fc1, Fc2, Fc3, Fc4, Af7, Af8 po indeksach
    X_train = X_train[:, :, selected]
    X_test  = X_test[:, :, selected]

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test  = le.transform(y_test)

    batch_size = default_batch_size
    train_ds = EEGDataset(X_train, y_train)
    val_ds   = EEGDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MSHCNN(
        n_channels=X_train.shape[2],
        time_points=X_train.shape[1],
        symmetric_pairs=[(0,1),(2,3),(4,5)],
        n_classes=len(le.classes_),
        n_filters=5,
        dropout=0.5
    ).to(device)
    class_counts = np.bincount(y_train)
    prior = class_counts / class_counts.sum()
    model.fc2.bias.data = torch.log(torch.from_numpy(prior + 1e-6).float().to(device))


    # Kryterium strat z wagami odwrotnymi do częstości klas
    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    epochs = 100
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-2,
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )

    for epoch in range(1, epochs+1):
        model.train()
        y_true, y_pred = [], []
        total_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        train_loss = total_loss / total
        train_acc  = correct / total * 100
        cm = confusion_matrix(y_true, y_pred)
        print("cm train")
        print(cm)
        # Walidacja
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)
                val_loss += loss.item() * X.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += X.size(0)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_loss /= val_total
        val_acc  = val_correct / val_total * 100

        cm = confusion_matrix(y_true, y_pred)
        print("cm test")

        print(cm)
        report = classification_report(
            y_true, y_pred,
            target_names=le.classes_,
            digits=4,
            zero_division=0
        )
        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | "
              f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%")
        print("Classification Report:\n", report)

    torch.save(model.state_dict(), 'mshcnn_final.pth')

if __name__ == '__main__':
    train()