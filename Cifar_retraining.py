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
try:
    from google.colab import drive
    # This will trigger a pop-up asking for permission to connect to your Google Drive
    drive.mount('/content/drive')
    is_colab = True
except ImportError:
    # If the 'google.colab' library doesn't exist, we are definitely on a local machine
    is_colab = False

# 2. Set the paths based on the environment
if is_colab and os.path.exists('/content/drive/MyDrive/'):
    print("Environment Detected: Google Colab")
    fake_data_path = '/content/drive/MyDrive/fake_cifar10_tensor.pt'
    real_data_root = '/content/data' # Saves CIFAR-10 to Colab's temporary storage
else:
    print("Environment Detected: Local Machine")
    # Assumes the .pt file is in the same folder as this script
    fake_data_path = './fake_cifar10_tensor.pt'
    real_data_root = './data'

# ==========================================
# 1. HARDWARE & DATASETS
# ==========================================

# Define your device to use that RTX 3050 (or Colab GPU)!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------
# A. TRAINING DATA (REAL + FAKE)
# ------------------------------------------
# The Augmented Training Pipeline (Used for Real Images)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Real Training Dataset
# Force real labels to be Tensors instead of Python ints so they match the fake labels!
target_transform = transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.long))

# Load Real Training Dataset
real_trainset = torchvision.datasets.CIFAR10(
    root=real_data_root,
    train=True,
    download=True,
    transform=train_transform,
    target_transform=target_transform # <--- THE FIX
)
# Load Fake Training Dataset
print(f"Loading synthetic dataset from {fake_data_path}...")
fake_data = torch.load(fake_data_path, map_location='cpu') # Load to CPU first to save VRAM
fake_trainset = TensorDataset(fake_data['images'], fake_data['labels'])

# Merge them!
augmented_trainset = ConcatDataset([real_trainset, fake_trainset])
augmented_trainloader = DataLoader(augmented_trainset, batch_size=64, shuffle=True)

print(f"Merge Successful! Total training images: {len(augmented_trainset)}")

# ------------------------------------------
# B. TESTING DATA (REAL ONLY)
# ------------------------------------------
# The Test Pipeline MUST NOT have random crops or flips!
# We just convert to tensor and normalize to match the training scale.
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Real Testing Dataset (train=False)
testset = torchvision.datasets.CIFAR10(root=real_data_root, train=False, download=True, transform=test_transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False) # No need to shuffle the test set

print(f"Test Set Ready! Total testing images: {len(testset)}")

# ==========================================
# 2. The Classifier CNN
# ==========================================

class CIFARClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # ==========================================
        # PART 1: FEATURE EXTRACTOR (The Funnel)
        # ==========================================

        # 1. First Block
        # Input: [Batch, 3, 32, 32]
        # Conv -> Extracts 32 feature maps. Size stays 32x32.
        # BatchNorm -> Re-centers outputs to mean=0, std=1. Prevents gradients from exploding/vanishing!
        # Pool -> Cuts spatial dimensions in half. Output shape: [Batch, 32, 16, 16]
        self.conv_1a = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn_1a = nn.BatchNorm2d(32)
        self.relu_1a = nn.ReLU()
        # repetition of the first block
        self.conv_1b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn_1b = nn.BatchNorm2d(32)
        self.relu_1b = nn.ReLU()
        # only now we shrink the image size thanks to the max pool (from 32x32 to 16x16)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 2. Second Block
        # Input: [Batch, 32, 16, 16]
        # Conv -> Extracts 64 feature maps.
        # BatchNorm -> Stabilizes the new 64 maps.
        # Pool -> Cuts dimensions in half. Output shape: [Batch, 64, 8, 8]
        self.conv_2a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_2a = nn.BatchNorm2d(64)
        self.relu_2a = nn.ReLU()
        self.conv_2b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_2b = nn.BatchNorm2d(64)
        self.relu_2b = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 3. Third Block
        # Input: [Batch, 64, 8, 8]
        # Conv -> Extracts 128 feature maps.
        # BatchNorm -> Stabilizes the deep 128 maps.
        # Pool -> Cuts dimensions in half. Output shape: [Batch, 128, 4, 4]
        self.conv_3a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn_3a = nn.BatchNorm2d(128)
        self.relu_3a = nn.ReLU()
        # repetition of the third block
        self.conv_3b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn_3b = nn.BatchNorm2d(128)
        self.relu_3b = nn.ReLU()
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # ==========================================
        # PART 2: CLASSIFICATION HEAD
        # ==========================================

        # Flatten takes the [Batch, 128, 4, 4] 3D maps and unrolls them into a 1D array of [Batch, 2048]
        self.flatten = nn.Flatten()

        # Fully Connected Layers
        # NOTE: Dropout is commented out to prevent "Regularization Stacking" since we are using Data Augmentation!
        self.fc_1 = nn.Linear(in_features=128 * 4 * 4, out_features=128)
        self.relu_4 = nn.ReLU()
        # self.dropout_1 = nn.Dropout(p=0.25)

        # By setting out_feature=2 MNIST (try Grad-Cam)
        self.fc_2 = nn.Linear(in_features=128, out_features=64)
        self.relu_5 = nn.ReLU()
        # self.dropout_2 = nn.Dropout(p=0.25)

        self.fc_3 = nn.Linear(in_features=64, out_features=10) # 10 final output classes

    def forward(self, x):
        # Pass the image through Block 1
        x = self.conv_1a(x)
        x = self.bn_1a(x) # Normalize before the activation function!
        x = self.relu_1a(x)
        x = self.conv_1b(x)
        x = self.bn_1b(x)
        x = self.relu_1b(x)
        x = self.pool_1(x)

        # Pass through Block 2
        x = self.conv_2a(x)
        x = self.bn_2a(x)
        x = self.relu_2a(x)
        x = self.conv_2b(x)
        x = self.bn_2b(x)
        x = self.relu_2b(x)
        x = self.pool_2(x)

        # Pass through Block 3
        x = self.conv_3a(x)
        x = self.bn_3a(x)
        x = self.relu_3a(x)
        x = self.conv_3b(x)
        x = self.bn_3b(x)
        x = self.relu_3b(x)
        x = self.pool_3(x)

        # Transition from Convolutions to Linear Layers
        x = self.flatten(x)

        # Pass through Classification Head
        x = self.fc_1(x)
        x = self.relu_4(x)
        # x = self.dropout_1(x)

        x = self.fc_2(x)
        x = self.relu_5(x)
        # x = self.dropout_2(x)

        # Final output (Remember: CrossEntropyLoss handles the Softmax for us automatically!)
        x = self.fc_3(x)

        return x


# Instantiate the model and send it to the GPU!
model = CIFARClassifier().to(device)

# Grab just ONE batch from the conveyor belt to test it
dataiter = iter(augmented_trainloader) # <--- CHANGED HERE
images, labels = next(dataiter)

# Move the images to the GPU and pass them through
images = images.to(device)
output = model(images)

print(f"Input batch shape: {images.size()}") # Should be [64, 3, 32, 32]
print(f"Output shape: {output.size()}")      # Should be [64, 10] (64 images, 10 class probabilities each)

# ==========================================
# 3. The Training Loop (CIFAR-10)
# ==========================================

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

epochs = 50

# --- NEW: Tracking lists for our plots! ---
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

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # --- Calculate Batch Statistics ---
        batch_loss = loss.item()
        _, predicted = torch.max(outputs.data, 1)
        batch_total = labels.size(0)
        batch_correct = (predicted == labels).sum().item()

        # --- NEW: Save every single batch metric ---
        history_loss.append(batch_loss)
        history_acc.append(batch_correct / batch_total)

        # For the epoch averages
        running_loss += batch_loss
        total += batch_total
        correct += batch_correct

    scheduler.step()

    # --- NEW: Print ONLY once per epoch ---
    epoch_loss = running_loss / len(augmented_trainloader)
    epoch_acc = 100 * correct / total
    current_lr = optimizer.param_groups[0]['lr']
    epoch_time = time.time() - start_time

    print(f"[Epoch {epoch + 1:02d}/{epochs}] Loss: {epoch_loss:.3f} | Acc: {epoch_acc:.2f}% | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

print("\n--- Finished Training! ---")

# SAVE THE INTACT MODEL
save_path = '/content/drive/MyDrive/augmented_cifar_cnn.pth'
torch.save(model.state_dict(), save_path)
print(f"--- INTACT MODEL SAFELY SAVED TO DRIVE AT: {save_path} ---")