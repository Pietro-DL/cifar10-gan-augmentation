import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. SETUP & PATHS
# ==========================================
# Point strictly to your local absolute path
BASE_DIR = r"C:\Users\pietr\OneDrive\Desktop\deep_learning\project_assignement"
DATA_DIR = os.path.join(BASE_DIR, "data")
WEIGHTS_DIR = os.path.join(BASE_DIR, "model_weights")

# Paths to the specific models
baseline_model_path = os.path.join(WEIGHTS_DIR, "cifar_model_cnn.pth")
augmented_model_path = os.path.join(WEIGHTS_DIR, "augmented_cifar_cnn.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Starting Final Evaluation on: {device} ---")

# ==========================================
# 2. DEFINING THE CLASSIFIER ARCHITECTURE
# ==========================================
# We must redefine the architecture so PyTorch knows where to put the loaded weights!
class CIFARClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_1a = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn_1a = nn.BatchNorm2d(32)
        self.relu_1a = nn.ReLU() 
        self.conv_1b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn_1b = nn.BatchNorm2d(32)
        self.relu_1b = nn.ReLU() 
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv_2a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_2a = nn.BatchNorm2d(64)
        self.relu_2a = nn.ReLU()
        self.conv_2b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_2b = nn.BatchNorm2d(64)
        self.relu_2b = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv_3a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn_3a = nn.BatchNorm2d(128)
        self.relu_3a = nn.ReLU()
        self.conv_3b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn_3b = nn.BatchNorm2d(128)
        self.relu_3b = nn.ReLU()
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(in_features=128 * 4 * 4, out_features=128) 
        self.relu_4 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=128, out_features=64)
        self.relu_5 = nn.ReLU()
        self.fc_3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.conv_1a(x)
        x = self.bn_1a(x)
        x = self.relu_1a(x)
        x = self.conv_1b(x)
        x = self.bn_1b(x)
        x = self.relu_1b(x)
        x = self.pool_1(x)

        x = self.conv_2a(x)
        x = self.bn_2a(x)
        x = self.relu_2a(x)
        x = self.conv_2b(x)
        x = self.bn_2b(x)
        x = self.relu_2b(x)
        x = self.pool_2(x)

        x = self.conv_3a(x)
        x = self.bn_3a(x)
        x = self.relu_3a(x)
        x = self.conv_3b(x)
        x = self.bn_3b(x)
        x = self.relu_3b(x)
        x = self.pool_3(x)

        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.relu_4(x)
        x = self.fc_2(x)
        x = self.relu_5(x)
        x = self.fc_3(x) 
        return x

# ==========================================
# 3. PREPARING THE STRICT TEST SET
# ==========================================
# NO Augmentations! Just Tensor conversion and matching the [-1, 1] normalization.
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

# ==========================================
# 4. EVALUATION FUNCTION
# ==========================================


def evaluate_model(model_path, model_name):
    print(f"\nEvaluating: {model_name}...")
    
    model = CIFARClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
            
    # Calculate Global Metrics
    acc = accuracy_score(all_true, all_preds) * 100
    precision = precision_score(all_true, all_preds, average='macro', zero_division=0) * 100
    recall = recall_score(all_true, all_preds, average='macro', zero_division=0) * 100
    f1 = f1_score(all_true, all_preds, average='macro') * 100
    cm = confusion_matrix(all_true, all_preds)

    # --- NEW: Extract Per-Class Metrics ---
    # Setting average=None returns an array of 10 values (one for each class)
    prec_per_class = precision_score(all_true, all_preds, average=None, zero_division=0) * 100
    rec_per_class = recall_score(all_true, all_preds, average=None, zero_division=0) * 100
    
    print(f"[{model_name}] Accuracy:  {acc:.2f}%")
    print(f"[{model_name}] Precision: {precision:.2f}%")
    print(f"[{model_name}] Recall:    {recall:.2f}%")
    print(f"[{model_name}] F1-Score:  {f1:.2f}%")
    
    # We now return the global metrics AND the per-class arrays!
    return cm, acc, f1, precision, recall, prec_per_class, rec_per_class

# ==========================================
# 5. EXECUTE AND PLOT (UPDATED)
# ==========================================
# 1. Get ALL the matrices, global scores, and per-class arrays
cm_base, acc_base, f1_base, mac_prec_base, mac_rec_base, prec_arr_base, rec_arr_base = evaluate_model(baseline_model_path, "Baseline Model")
cm_aug, acc_aug, f1_aug, mac_prec_aug, mac_rec_aug, prec_arr_aug, rec_arr_aug = evaluate_model(augmented_model_path, "Augmented Model")

cifar_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# ------------------------------------------
# Plot A: The Confusion Matrices
# ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.heatmap(cm_base, annot=True, fmt='d', cmap='Reds', ax=axes[0], xticklabels=cifar_classes, yticklabels=cifar_classes)
axes[0].set_title(f'Baseline Model\nAccuracy: {acc_base:.2f}% | F1: {f1_base:.2f}%', fontsize=14)
axes[0].set_xlabel('Predicted Class')
axes[0].set_ylabel('True Class')

sns.heatmap(cm_aug, annot=True, fmt='d', cmap='Blues', ax=axes[1], xticklabels=cifar_classes, yticklabels=cifar_classes)
axes[1].set_title(f'Augmented Model (WGAN-GP)\nAccuracy: {acc_aug:.2f}% | F1: {f1_aug:.2f}%', fontsize=14)
axes[1].set_xlabel('Predicted Class')
axes[1].set_ylabel('True Class')

plt.tight_layout()
save_cm_path = os.path.join(BASE_DIR, "images", "confusion_matrix_comparison.png")
plt.savefig(save_cm_path, dpi=300)
print(f"\n--- Confusion Matrices saved to: {save_cm_path} ---")
plt.show()

# ------------------------------------------
# Plot B: The Deviation Bar Charts
# ------------------------------------------
# Calculate how far each class deviates from the model's global average
dev_prec_base = prec_arr_base - mac_prec_base
dev_rec_base = rec_arr_base - mac_rec_base

dev_prec_aug = prec_arr_aug - mac_prec_aug
dev_rec_aug = rec_arr_aug - mac_rec_aug

x = np.arange(len(cifar_classes))
width = 0.35  # the width of the grouped bars

fig2, axes2 = plt.subplots(2, 1, figsize=(14, 12))

# --- Precision Deviation Subplot ---
axes2[0].bar(x - width/2, dev_prec_base, width, label='Baseline Model', color='#e74c3c', edgecolor='black')
axes2[0].bar(x + width/2, dev_prec_aug, width, label='Augmented Model', color='#3498db', edgecolor='black')
axes2[0].set_ylabel('Deviation from Macro Precision (%)', fontsize=12)
axes2[0].set_title('Per-Class PRECISION Deviation from Mean', fontsize=14, fontweight='bold')
axes2[0].set_xticks(x)
axes2[0].set_xticklabels(cifar_classes, fontsize=12)
axes2[0].legend()
axes2[0].axhline(0, color='black', linewidth=1.5, linestyle='--') # The Zero-Mean Line
axes2[0].grid(axis='y', linestyle='--', alpha=0.7)

# --- Recall Deviation Subplot ---
axes2[1].bar(x - width/2, dev_rec_base, width, label='Baseline Model', color='#e74c3c', edgecolor='black')
axes2[1].bar(x + width/2, dev_rec_aug, width, label='Augmented Model', color='#3498db', edgecolor='black')
axes2[1].set_ylabel('Deviation from Macro Recall (%)', fontsize=12)
axes2[1].set_title('Per-Class RECALL Deviation from Mean', fontsize=14, fontweight='bold')
axes2[1].set_xticks(x)
axes2[1].set_xticklabels(cifar_classes, fontsize=12)
axes2[1].legend()
axes2[1].axhline(0, color='black', linewidth=1.5, linestyle='--') # The Zero-Mean Line
axes2[1].grid(axis='y', linestyle='--', alpha=0.7)

# THE FIX: Use fig2.tight_layout with explicit height padding instead of the generic plt.tight_layout()
fig2.tight_layout(h_pad=4.0) 

save_dev_path = os.path.join(BASE_DIR, "images", "deviation_plot.png")
# bbox_inches='tight' ensures the outer edges aren't cut off when saving
fig2.savefig(save_dev_path, dpi=300, bbox_inches='tight') 
print(f"--- Deviation Plot saved to: {save_dev_path} ---")
plt.show()