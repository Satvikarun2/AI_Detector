import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Import components from original train.py
from train import MultiModalDataset, validate
from models import TextureContrastClassifier

def test_on_unseen():
    # --- Configuration ---
    # Path to the NEW H5 folder created by preprocess2.py
    TEST_H5_DIR = './h5_unseen'  
    MODEL_PATH = './checkpoints/best_model.pth' # Using Epoch 77 best model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Dataset
    if not os.path.exists(TEST_H5_DIR) or not os.listdir(TEST_H5_DIR):
        print(f"❌ Error: {TEST_H5_DIR} is empty. Run preprocess2.py first!")
        return

    print(f"📂 Loading unseen test data from {TEST_H5_DIR}...")
    test_dataset = MultiModalDataset(TEST_H5_DIR)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 2. Load the Best Model
    model = TextureContrastClassifier().to(device)
    if os.path.exists(MODEL_PATH):
        print(f"✅ Loading Best Model weights (Epoch 77) from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval() 
    else:
        print(f"❌ Error: {MODEL_PATH} not found!")
        return

    # 3. Evaluation Logic
    print(f"🚀 Testing on {len(test_dataset)} unseen images...")
    criterion = nn.BCEWithLogitsLoss()
    
    # Use standard validate function for consistency
    t_loss, t_acc, t_prec, t_rec, t_fpr, t_cm = validate(model, test_loader, criterion, device)

    # 4. Final Thesis Metrics
    print("\n" + "="*40)
    print("EXTERNAL VALIDATION")
    print("="*40)
    print(f"Unseen Accuracy:  {t_acc:.4f}")
    print(f"Precision:        {t_prec:.4f}")
    print(f"Recall (Catch):   {t_rec:.4f}")
    print(f"FPR (False Alarm): {t_fpr:.4f}")
    print("-" * 40)
    print(f"Confusion Matrix:\n{t_cm}")
    print("="*40)

if __name__ == '__main__':
    test_on_unseen()