import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm

# Import your custom classes from your original files
from train import MultiModalDataset, validate
from models import TextureContrastClassifier

def resume_training():
    # --- Configuration ---
    H5_DIR = './h5_storage'  
    SAVE_DIR = './checkpoints'
    CHECKPOINT_FILE = os.path.join(SAVE_DIR, 'last_checkpoint.pth')
    
    START_EPOCH = 50  # We finished 50, starting with 51
    TOTAL_EPOCHS = 80 
    BATCH_SIZE = 256
    LEARNING_RATE = 5e-5 # Stable rate for fine-tuning
    
    # Best Accuracy from your previous Epoch 50 check
    best_val_acc = 0.7428 

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📦 Initializing Dataset for Fine-Tuning...")
    
    full_dataset = MultiModalDataset(H5_DIR)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    # Use manual_seed to ensure the train/val split is identical to the first run
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], 
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              pin_memory=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # --- Model & Checkpoint Loading ---
    model = TextureContrastClassifier().to(device)
    
    if os.path.exists(CHECKPOINT_FILE):
        print(f"🔄 Loading checkpoint: {CHECKPOINT_FILE}")
        model.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=device))
        model.train() # Set to training mode for dropout/batchnorm
    else:
        raise FileNotFoundError(f"Could not find {CHECKPOINT_FILE}. Ensure the path is correct.")

    # --- Optimizer & Loss ---
    # Encourages low False Positive Rate
    pos_weight = torch.tensor([0.8]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"🚀 Continuing training from Epoch {START_EPOCH + 1} to {TOTAL_EPOCHS}...")

    # --- Training Loop ---
    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
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

        # Validation phase
        v_loss, v_acc, v_prec, v_rec, v_fpr, v_cm = validate(model, val_loader, criterion, device)
        
        print(f"\n--- Epoch {epoch+1} Results ---")
        print(f"Loss: {v_loss:.4f} | Acc: {v_acc:.4f} | Precision: {v_prec:.4f} | Recall: {v_rec:.4f} | FPR: {v_fpr:.4f}")
        print(f"Confusion Matrix:\n{v_cm}")
        
        # Save best model separately
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f"⭐ New best accuracy ({v_acc:.4f})! Model saved.")

        # Always save last checkpoint for safety
        torch.save(model.state_dict(), CHECKPOINT_FILE)

if __name__ == '__main__':
    resume_training()