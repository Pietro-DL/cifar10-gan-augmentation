# ==========================================
# 1. SETUP & DYNAMIC PATH MANAGEMENT
# ==========================================
"""
Dynamically resolves the working directory to ensure cross-platform compatibility.
This script expects the pre-trained StyleGAN2 ablation weights to be located 
in the /model_weights directory.
"""
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd 

# Dynamically set BASE_DIR to the folder containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
WEIGHTS_DIR = os.path.join(BASE_DIR, "model_weights")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# Ensure output directory exists
os.makedirs(IMAGES_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Starting Ablation Evaluation on: {device} ---")
print(f"--- Working Directory: {BASE_DIR} ---")

# Define the models sequentially to track the ablation trend.
# We are specifically evaluating the _v2 weights (StyleGAN2-ADA generations).
experiments = [
    {"label": "0% Fake (Baseline)", "fake_pct": 0, "file": "cnn_100real_0fake.pth"},
    {"label": "30% Fake (StyleGAN2)", "fake_pct": 30, "file": "cnn_70real_30fake.pth"},
    {"label": "50% Fake (StyleGAN2)", "fake_pct": 50, "file": "cnn_50real_50fake.pth"},
    {"label": "70% Fake (StyleGAN2)", "fake_pct": 70, "file": "cnn_30real_70fake.pth"},
    {"label": "100% Fake (Zero-Shot)", "fake_pct": 100, "file": "cnn_0real_100fake.pth"}
]

# ==========================================
# 2. DEFINING THE CLASSIFIER ARCHITECTURE
# ==========================================
"""
The standard VGG-style Convolutional Neural Network used across all experiments. 
Reinstantiated here to map the structural dictionaries of the loaded .pth weights.
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
Loads the unmodified CIFAR-10 test set. No geometric augmentations are applied, 
ensuring an objective, purely inferential benchmark.
"""
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)
testloader = DataLoader(testset, batch_size=256, shuffle=False)

# ==========================================
# 4. BATCH EVALUATION PIPELINE
# ==========================================
"""
Iterates through every ablation model, running inference on the complete test set.
Extracts both global (Macro) metrics and granular per-class arrays, appending them 
to a central results dictionary for consolidated visual/tabular processing.
"""
results = {
    "fake_pct": [], "accuracy": [], "f1_score": [],
    "precision": [], "recall": [], 
    "prec_per_class": [], "rec_per_class": [], "cm": []              
}

print("\n" + "="*50)
print(f"{'Model Mix':<20} | {'Acc':<6} | {'F1':<6} | {'Prec':<6} | {'Rec':<6}")
print("="*50)

for exp in experiments:
    model_path = os.path.join(WEIGHTS_DIR, exp["file"]) # loads .pth weights
    
    if not os.path.exists(model_path):
        print(f"Warning: {exp['file']} not found. Skipping.")
        continue
        
    model = CIFARClassifier().to(device) # instantiate the model
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
    f1 = f1_score(all_true, all_preds, average='macro') * 100
    prec = precision_score(all_true, all_preds, average='macro', zero_division=0) * 100
    rec = recall_score(all_true, all_preds, average='macro', zero_division=0) * 100
    
    # Calculate Per-Class & Matrix Metrics
    cm = confusion_matrix(all_true, all_preds)
    prec_per_class = precision_score(all_true, all_preds, average=None, zero_division=0) * 100
    rec_per_class = recall_score(all_true, all_preds, average=None, zero_division=0) * 100
    
    # Store results sequentially
    results["fake_pct"].append(exp["fake_pct"])
    results["accuracy"].append(acc)
    results["f1_score"].append(f1)
    results["precision"].append(prec)
    results["recall"].append(rec)
    results["prec_per_class"].append(prec_per_class)
    results["rec_per_class"].append(rec_per_class)
    results["cm"].append(cm)
    
    print(f"{exp['label']:<20} | {acc:.2f}% | {f1:.2f}% | {prec:.2f}% | {rec:.2f}%")

print("="*50)

# ==========================================
# 5. GENERATE 5-MODEL DEVIATION PLOT
# ==========================================
"""
Generates a side-by-side grouped bar chart comparing the class-wise deviations 
of all 5 models simultaneously. Uses mathematical offsets to align the 5 bars 
neatly over each class label on the X-axis.
"""
cifar_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if len(results["fake_pct"]) > 0:
    fig, axes = plt.subplots(2, 1, figsize=(18, 14))
    
    x = np.arange(len(cifar_classes))
    width = 0.15 # Thinner bars to fit 5 side-by-side per class
    
    # Define horizontal offsets to center the 5 bars over the x-tick
    offsets = [-2*width, -width, 0, width, 2*width]
    colors = ['#2c3e50', '#e74c3c', '#f39c12', '#3498db', '#9b59b6']
    
    for i in range(len(results["fake_pct"])):
        label = f"{results['fake_pct'][i]}% Fake"
        
        # Calculate deviations from this specific model's global macro average
        dev_prec = results["prec_per_class"][i] - results["precision"][i]
        dev_rec = results["rec_per_class"][i] - results["recall"][i]
        
        axes[0].bar(x + offsets[i], dev_prec, width, label=label, color=colors[i], edgecolor='black')
        axes[1].bar(x + offsets[i], dev_rec, width, label=label, color=colors[i], edgecolor='black')

    # [Keep all your axes[0] and axes[1] formatting lines exactly the same here...]

    fig.tight_layout(h_pad=4.0)
    save_dev_path = os.path.join(IMAGES_DIR, "5_model_deviation_plot_stylegan2.png")
    fig.savefig(save_dev_path, dpi=300, bbox_inches='tight')
    print(f"\n--- Deviation Plot saved to: {save_dev_path} ---")