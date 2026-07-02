# ==========================================
# 1. SETUP & DYNAMIC PATH MANAGEMENT
# ==========================================
"""
To ensure cross-platform compatibility, this script dynamically resolves its own 
directory path. It assumes the following repository structure:
/github
  ├── /model_weights
  ├── /images
  ├── /data
  └── Comparison_script.py
"""
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Dynamically set BASE_DIR to the folder containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
WEIGHTS_DIR = os.path.join(BASE_DIR, "model_weights")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# Ensure the images directory exists for saving plots
os.makedirs(IMAGES_DIR, exist_ok=True)

# Paths to the specific evaluation models
baseline_model_path = os.path.join(WEIGHTS_DIR, "cnn_100real_0fake.pth")
augmented_model_path = os.path.join(WEIGHTS_DIR, "augmented_cifar_cnn.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Starting Final Evaluation on: {device} ---")
print(f"--- Working Directory: {BASE_DIR} ---")

# ==========================================
# 2. DEFINING THE CLASSIFIER ARCHITECTURE
# ==========================================
"""
The identical VGG-style Convolutional Neural Network used during training.
PyTorch requires the structural blueprint of the network to be defined in memory 
before it can successfully map the loaded .pth weight dictionaries.

--- Convolutional Feature Extractor ---
Block 1: [Batch, 3, 32, 32]   --> [Batch, 32, 16, 16]
Block 2: [Batch, 32, 16, 16]  --> [Batch, 64, 8, 8]
Block 3: [Batch, 64, 8, 8]    --> [Batch, 128, 4, 4]

--- Classification Head ---
Flatten: [Batch, 128, 4, 4]   --> [Batch, 2048]
Linear:  [Batch, 2048]        --> [Batch, 10] (via intermediate 128 and 64 layers)
"""
class CIFARClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1
        self.conv_1a = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn_1a = nn.BatchNorm2d(32)
        self.relu_1a = nn.ReLU()
        self.conv_1b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn_1b = nn.BatchNorm2d(32)
        self.relu_1b = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Block 2
        self.conv_2a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_2a = nn.BatchNorm2d(64)
        self.relu_2a = nn.ReLU()
        self.conv_2b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_2b = nn.BatchNorm2d(64)
        self.relu_2b = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Block 3
        self.conv_3a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn_3a = nn.BatchNorm2d(128)
        self.relu_3a = nn.ReLU()
        self.conv_3b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn_3b = nn.BatchNorm2d(128)
        self.relu_3b = nn.ReLU()
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Classification Head
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(in_features=128 * 4 * 4, out_features=128)
        self.relu_4 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=128, out_features=64)
        self.relu_5 = nn.ReLU()
        self.fc_3 = nn.Linear(in_features=64, out_features=10) 

    def forward(self, x):
        # Block 1
        x = self.pool_1(self.relu_1b(self.bn_1b(self.conv_1b(self.relu_1a(self.bn_1a(self.conv_1a(x)))))))
        # Block 2
        x = self.pool_2(self.relu_2b(self.bn_2b(self.conv_2b(self.relu_2a(self.bn_2a(self.conv_2a(x)))))))
        # Block 3
        x = self.pool_3(self.relu_3b(self.bn_3b(self.conv_3b(self.relu_3a(self.bn_3a(self.conv_3a(x)))))))
        
        # Classification Head
        x = self.flatten(x)
        # Final output leverages raw logits; CrossEntropyLoss applies the Softmax internally
        return self.fc_3(self.relu_5(self.fc_2(self.relu_4(self.fc_1(x)))))

# ==========================================
# 3. PREPARING THE STRICT TEST SET
# ==========================================
"""
To ensure a strictly objective benchmark, the test dataloader completely disables 
random crops and horizontal flips. The images are only tensorized and normalized 
to the [-1, 1] range to match the mathematical scale of the training distributions.
"""
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

# ==========================================
# 4. EVALUATION FUNCTION
# ==========================================
"""
This function loads a pre-trained model and evaluates it against the strict test set.
It utilizes torch.no_grad() to disable the computational graph, saving memory and 
preventing accidental weight updates. It calculates both Macro-averaged global metrics 
and localized per-class metrics to identify specific categorical degradations.
"""
def evaluate_model(model_path, model_name):
    print(f"\nEvaluating: {model_name}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
        
    model = CIFARClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device) # Load images into the device
            outputs = model(images) # Forward Pass
            _, preds = torch.max(outputs, 1) # Get the predicted class
            
            all_preds.extend(preds.cpu().numpy()) # Append the predicted class to the list
            all_true.extend(labels.cpu().numpy()) # Append the true class to the list
            
    # Calculate Global Metrics
    acc = accuracy_score(all_true, all_preds) * 100
    precision = precision_score(all_true, all_preds, average='macro', zero_division=0) * 100
    recall = recall_score(all_true, all_preds, average='macro', zero_division=0) * 100
    f1 = f1_score(all_true, all_preds, average='macro') * 100
    cm = confusion_matrix(all_true, all_preds)

    # Extract Per-Class Metrics (average=None returns arrays of length 10)
    prec_per_class = precision_score(all_true, all_preds, average=None, zero_division=0) * 100
    rec_per_class = recall_score(all_true, all_preds, average=None, zero_division=0) * 100
    
    print(f"[{model_name}] Accuracy:  {acc:.2f}%")
    print(f"[{model_name}] Precision: {precision:.2f}%")
    print(f"[{model_name}] Recall:    {recall:.2f}%")
    print(f"[{model_name}] F1-Score:  {f1:.2f}%")
    
    return cm, acc, f1, precision, recall, prec_per_class, rec_per_class

# ==========================================
# 5. EXECUTE AND PLOT
# ==========================================
"""
Executes the evaluation pipeline for both the Baseline and Augmented models, 
extracting their respective confusion matrices and localized deviations. 
Generates and saves comparative visual analytics.
"""
# get all the scores and metrics from the evaluate_model function for both models
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