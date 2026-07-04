import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import time

# ==========================================
# UNIVERSAL PATH MANAGEMENT & DRIVE MOUNTING
# ==========================================

# 1. Attempt to mount Google Drive if we are in Colab
# 2. Set the paths based on the environment
if is_colab and os.path.exists('/content/drive/MyDrive/'):
    print("Environment Detected: Google Colab")
    fake_data_path = '/content/drive/MyDrive/fake_cifar10_tensor.pt'
    real_data_root = '/content/data' 
    save_path = '/content/drive/MyDrive/augmented_cifar_cnn.pth' # Colab Save Path
else:
    print("Environment Detected: Local Machine")
    fake_data_path = './fake_cifar10_tensor.pt'
    real_data_root = './data'
    
    # Ensure the local weights folder exists!
    os.makedirs('./model_weights', exist_ok=True)
    save_path = './model_weights/augmented_cifar_cnn.pth' # Local Save Path

# ==========================================
# 1. HARDWARE & DATASETS (THE HYBRID PIPELINE)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------
# A. TRAINING DATA (REAL + FAKE)
# ------------------------------------------
"""
This section constructs the augmented training pipeline. 
Standard geometric augmentations (crops, flips, rotations) are applied to the real CIFAR-10 images. 
These are then concatenated with the pre-generated synthetic tensors.

CRITICAL FIX: Standard CIFAR-10 labels are loaded as native Python integers. The synthetic labels 
are stored as PyTorch tensors. The `target_transform` forces the real labels into `torch.long` tensors, 
ensuring mathematical compatibility during the ConcatDataset merge.
"""
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Force real labels to be Tensors so they match the synthetic dataset format
target_transform = transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.long))

# Load Real Training Dataset
real_trainset = torchvision.datasets.CIFAR10(
    root=real_data_root, train=True, download=True,
    transform=train_transform, target_transform=target_transform
)

# Load Synthetic Training Dataset (Loaded to CPU RAM first to prevent VRAM overflow)
print(f"Loading synthetic dataset from {fake_data_path}...")
fake_data = torch.load(fake_data_path, map_location='cpu') 
fake_trainset = TensorDataset(fake_data['images'], fake_data['labels'])

# Merge Real and Synthetic datasets into a single training pipeline
augmented_trainset = ConcatDataset([real_trainset, fake_trainset])
augmented_trainloader = DataLoader(augmented_trainset, batch_size=64, shuffle=True)

print(f"Merge Successful! Total training images: {len(augmented_trainset)}")

# ------------------------------------------
# B. TESTING DATA (REAL ONLY)
# ------------------------------------------
"""
The test dataset strictly excludes spatial augmentations (no crops or flips) to ensure 
a purely objective evaluation baseline. Images are only normalized to match the training scale.
"""
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

testset = torchvision.datasets.CIFAR10(root=real_data_root, train=False, download=True, transform=test_transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False) 

print(f"Test Set Ready! Total testing images: {len(testset)}")

# ==========================================
# 2. THE CLASSIFIER ARCHITECTURE
# ==========================================
"""
A high-capacity VGG-style Convolutional Neural Network designed to evaluate the quality 
of the injected synthetic data. The architecture utilizes three sequential convolutional blocks.

Note on Regularization: Explicit dropout layers in the classification head have been intentionally 
disabled. Because the injection of synthetic data fundamentally acts as a powerful structural 
regularizer, stacking artificial Dropout on top of it risks heavily underfitting the network.

--- Convolutional Feature Extractor ---
Block 1: [Batch, 3, 32, 32]   --> [Batch, 32, 16, 16] (2x Conv3x3, BN, ReLU, MaxPool)
Block 2: [Batch, 32, 16, 16]  --> [Batch, 64, 8, 8]   (2x Conv3x3, BN, ReLU, MaxPool)
Block 3: [Batch, 64, 8, 8]    --> [Batch, 128, 4, 4]  (2x Conv3x3, BN, ReLU, MaxPool)

--- Classification Head ---
Flatten: [Batch, 128, 4, 4]   --> [Batch, 2048]
Linear1: [Batch, 2048]        --> [Batch, 128]
Linear2: [Batch, 128]         --> [Batch, 64]
Linear3: [Batch, 64]          --> [Batch, 10] (Raw Logits)
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

# Instantiate the model and map to device
model = CIFARClassifier().to(device)

# Diagnostic batch pass to verify tensor shapes
dataiter = iter(augmented_trainloader)
images, labels = next(dataiter)
images = images.to(device)
output = model(images)

print(f"Diagnostic - Input batch shape: {images.size()}") # Expected: [64, 3, 32, 32]
print(f"Diagnostic - Output shape: {output.size()}")      # Expected: [64, 10]

# ==========================================
# 3. SUPERVISED TRAINING LOOP
# ==========================================
"""
SUMMARY OF THE CLASSIFICATION TRAINING LOOP
This section executes a standard supervised learning pipeline to optimize the CNN.

- Loss Function: CrossEntropyLoss (Computes softmax automatically).
- Optimizer: Adam (Learning Rate = 0.001).
- Scheduler: StepLR (Decays the learning rate by a factor of 0.1 every 20 epochs to 
  prevent overshooting local minima during late-stage convergence).
  
The loop tracks both batch-level and epoch-level statistics (Loss and Accuracy) for 
downstream analysis and plotting.
"""
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

epochs = 50

# Tracking lists for post-training visualization
history_loss = []
history_acc = []

print(f"--- Starting Training on {device} ---")

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for i, data in enumerate(augmented_trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 1. Clear previous gradients
        optimizer.zero_grad()
        
        # 2. Forward pass
        outputs = model(inputs)
        
        # 3. Compute loss against ground-truth labels
        loss = criterion(outputs, labels)
        
        # 4. Backpropagate error
        loss.backward()
        
        # 5. Update model weights
        optimizer.step()

        # --- Calculate Batch Statistics ---
        batch_loss = loss.item()
        _, predicted = torch.max(outputs.data, 1)
        batch_total = labels.size(0)
        batch_correct = (predicted == labels).sum().item()

        # Save granular batch metrics
        history_loss.append(batch_loss)
        history_acc.append(batch_correct / batch_total)

        # Accumulate for epoch averages
        running_loss += batch_loss
        total += batch_total
        correct += batch_correct

    # Decay the learning rate if the step boundary is reached
    scheduler.step()

    # --- Epoch Summary Logging ---
    epoch_loss = running_loss / len(augmented_trainloader)
    epoch_acc = 100 * correct / total
    current_lr = optimizer.param_groups[0]['lr']
    epoch_time = time.time() - start_time

    print(f"[Epoch {epoch + 1:02d}/{epochs}] Loss: {epoch_loss:.3f} | Acc: {epoch_acc:.2f}% | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

print("\n--- Finished Training! ---")

# Save the final structural weights for the evaluation script
save_path = '/content/drive/MyDrive/augmented_cifar_cnn.pth'
torch.save(model.state_dict(), save_path)
print(f"--- INTACT MODEL SAFELY SAVED AT: {save_path} ---")