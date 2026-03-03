import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import os
from tqdm import tqdm

# Import your custom classes from your original files
from train import MultiModalDataset, validate
from models import TextureContrastClassifier

def resume_training():
    # --- Configuration ---
    H5_DIR = './h5_storage'  
    SAVE_DIR = './checkpoints'
    
    # Path to the best model achieved at Epoch 77
    BEST_MODEL_FILE = os.path.join(SAVE_DIR, 'best_model.pth')
    CHECKPOINT_FILE = os.path.join(SAVE_DIR, 'last_checkpoint.pth')
    
    START_EPOCH = 77  # Resuming from the best state
    TOTAL_EPOCHS = 110 
    BATCH_SIZE = 256
    
    # Lower Learning Rate for stable fine-tuning with new data
    LEARNING_RATE = 1e-5 
    
    # Existing best accuracy record
    best_val_acc = 0.7868 

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📦 Initializing Dataset and calculating importance weights...")
    
    full_dataset = MultiModalDataset(H5_DIR)
    
    # --- Importance Weighting logic (Fixes AttributeError) ---
    # We use full_dataset.h5_files and full_dataset.lengths directly
    sample_weights = []
    for h5_file, length in zip(full_dataset.h5_files, full_dataset.lengths):
        # Identify the new data files from your recent ai4/real4 run
        if "processed_data_1" in h5_file: 
            # Give new data 3x the weight so the model learns it faster
            sample_weights.extend([2.0] * length)  
        else:
            # Keep original data at baseline importance
            sample_weights.extend([1.0] * length)  

    # --- Dataset Split ---
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], 
                                    generator=torch.Generator().manual_seed(42))

    # --- Weighted Sampler for Training ---
    # We must extract the weights corresponding only to the training subset
    train_indices = train_ds.indices
    train_weights = [sample_weights[i] for i in train_indices]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

    # Use the sampler in the loader (shuffle must be False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, 
                                 pin_memory=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # --- Model Loading ---
    model = TextureContrastClassifier().to(device)
    
    if os.path.exists(BEST_MODEL_FILE):
        print(f"🔄 Resuming from Best Model (Epoch 77): {BEST_MODEL_FILE}")
        model.load_state_dict(torch.load(BEST_MODEL_FILE, map_location=device))
        model.train() 
    else:
        raise FileNotFoundError(f"Could not find {BEST_MODEL_FILE}. Ensure Epoch 77 weights are present.")

    # --- Optimizer & Loss ---
    pos_weight = torch.tensor([0.8]).to(device) # Maintain Low FPR priority
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"🚀 Fine-Tuning from Epoch {START_EPOCH + 1} to {TOTAL_EPOCHS}...")

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
        
        # Save new best model separately if accuracy improves
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), BEST_MODEL_FILE)
            print(f"⭐ New best accuracy ({v_acc:.4f})! Best Model updated.")

        # Always save last checkpoint for safety
        torch.save(model.state_dict(), CHECKPOINT_FILE)

if __name__ == '__main__':
    resume_training()