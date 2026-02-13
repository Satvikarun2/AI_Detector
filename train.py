import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from models import TextureContrastClassifier

# --- Dataset Handler with Enhanced Pickling Safety ---
class MultiModalDataset(Dataset):
    def __init__(self, h5_dir):
        self.h5_dir = h5_dir
        # Store only the filenames to avoid pickling open file handles
        self.h5_files = sorted([f for f in os.listdir(h5_dir) if f.endswith('.h5')])
        self.lengths = []
        
        # Briefly open files to calculate total dataset size and check balance
        self.total_pos = 0
        self.total_neg = 0
        print("🔍 Scanning dataset for size and class balance...")
        for f_name in self.h5_files:
            with h5py.File(os.path.join(h5_dir, f_name), 'r') as f:
                labels = f['labels'][:]
                self.lengths.append(len(labels))
                self.total_pos += np.sum(labels == 1)
                self.total_neg += np.sum(labels == 0)
            
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.total_size = self.cumulative_lengths[-1]
        
        # Display class distribution for thesis documentation
        print(f"📊 Dataset Summary: {self.total_size:,} total images")
        print(f"   - AI Generated (Positive): {self.total_pos:,} ({100*self.total_pos/self.total_size:.1f}%)")
        print(f"   - Real Photographs (Negative): {self.total_neg:,} ({100*self.total_neg/self.total_size:.1f}%)")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Find which H5 file the index belongs to
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        local_idx = idx if file_idx == 0 else idx - self.cumulative_lengths[file_idx - 1]
        
        f_path = os.path.join(self.h5_dir, self.h5_files[file_idx])
        
        # OPEN FILE HERE: Every worker process will create its own local handle
        # using 'with' ensures it is closed after reading, avoiding process locks.
        with h5py.File(f_path, 'r', libver='latest', swmr=True) as f:
            return (
                torch.tensor(f['rich'][local_idx], dtype=torch.float32),
                torch.tensor(f['poor'][local_idx], dtype=torch.float32),
                torch.tensor(f['ela'][local_idx], dtype=torch.float32),
                torch.tensor(f['noise'][local_idx], dtype=torch.float32),
                torch.tensor(f['labels'][local_idx], dtype=torch.float32)
            )

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for r, p, e, n, labels in loader:
            r, p, e, n, labels = r.to(device), p.to(device), e.to(device), n.to(device), labels.to(device)
            outputs = model(r, p, e, n).squeeze()
            val_loss += criterion(outputs, labels).item()
            
            y_true.extend(labels.cpu().numpy())
            # Since model output is raw logits, apply sigmoid for the binary threshold
            y_pred.extend((torch.sigmoid(outputs) > 0.5).float().cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate False Positive Rate (FPR) for Master's requirements
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return val_loss / len(loader), acc, prec, rec, fpr, cm

if __name__ == '__main__':
    H5_DIR = './h5_storage'  
    SAVE_DIR = './checkpoints'
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("📦 Initializing Dataset...")
    full_dataset = MultiModalDataset(H5_DIR)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # Settings tuned for RTX 3050 and Windows stability
    train_loader = DataLoader(
        train_ds, 
        batch_size=256, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=2, 
        persistent_workers=True
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, pin_memory=True)

    device = torch.device('cuda')
    model = TextureContrastClassifier().to(device)

    # pos_weight=0.8 encourages the model to be more cautious about flagging AI,
    # helping reach your goal of a low False Positive Rate.
    pos_weight = torch.tensor([0.8]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    
    # Reduced learning rate for smoother attention weight convergence
    optimizer = optim.Adam(model.parameters(), lr=5e-5) 

    best_val_acc = 0.0

    print(f"🚀 Starting training on {device}...")
    for epoch in range(50):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for r, p, e, n, labels in progress_bar:
            r, p, e, n, labels = r.to(device), p.to(device), e.to(device), n.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(r, p, e, n).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        v_loss, v_acc, v_prec, v_rec, v_fpr, v_cm = validate(model, val_loader, criterion, device)
        
        print(f"\n--- Epoch {epoch+1} Results ---")
        print(f"Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {v_acc:.4f}")
        print(f"Precision: {v_prec:.4f} | Recall: {v_rec:.4f} | FPR: {v_fpr:.4f}")
        print(f"Confusion Matrix:\n{v_cm}")
        
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")
            print("⭐ Best model saved!")

        torch.save(model.state_dict(), f"{SAVE_DIR}/last_checkpoint.pth")