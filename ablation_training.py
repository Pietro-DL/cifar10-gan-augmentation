import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader, Subset
import torch.nn as nn
import numpy as np
import time

# ==========================================
# UNIVERSAL PATH MANAGEMENT & DRIVE MOUNTING
# ==========================================
try:
    from google.colab import drive
    drive.mount('/content/drive')
    is_colab = True
except ImportError:
    is_colab = False

if is_colab and os.path.exists('/content/drive/MyDrive/'):
    print("Environment Detected: Google Colab")
    fake_data_path = '/content/drive/MyDrive/stylegan2_cifar10_tensor.pt'
    real_data_root = '/content/data'
else:
    print("Environment Detected: Local Machine")
    fake_data_path = './fake_cifar10_tensor.pt'
    real_data_root = './data'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. LOAD THE BASE DATASETS
# ==========================================
"""
To optimize the ablation loop, the full Real and Synthetic datasets are loaded into CPU RAM 
exactly once at the start of the script. The training loop will dynamically extract pointers 
(Subsets) to these master datasets rather than reloading the files from disk for every experiment.
"""
print("\n--- Loading Base Datasets into Memory ---")

# Real Data Setup (Includes spatial augmentations)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
target_transform = transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.long))

real_trainset = torchvision.datasets.CIFAR10(
    root=real_data_root, train=True, download=True,
    transform=train_transform, target_transform=target_transform
)

# Fake Data Setup
fake_data = torch.load(fake_data_path, map_location='cpu')
# CRITICAL FIX: Clamp synthetic outliers to perfectly match the [-1, 1] Tanh bounds
fake_images = fake_data['images'].clamp(-1.0, 1.0)
fake_trainset = TensorDataset(fake_images, fake_data['labels'])
print("Base Datasets Loaded Successfully!\n")

# ==========================================
# 2. DEFINING THE CLASSIFIER
# ==========================================
"""
The standard VGG-style CNN architecture used across all experiments. 
Refer to `CNN_training.py` for the detailed block-by-block structural breakdown.
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
# 3. THE AUTOMATED ABLATION PIPELINE
# ==========================================
"""
This section dictates the logic for mathematically controlling the data distributions 
across multiple consecutive training runs. It ensures every model is trained on exactly 
50,000 total images, but smoothly interpolates the ratio of Real-to-Synthetic data 
while strictly enforcing perfect class balance.
"""

def get_stratified_indices(dataset, num_samples_total, num_classes=10):
    """
    Extracts a mathematically perfect, class-balanced subset of indices from a dataset.
    
    How it works:
    1. Calculates exactly how many images are needed per class (e.g., 15,000 total / 10 = 1,500 per class).
    2. Identifies the dataset type and extracts its full array of labels.
    3. Loops through each class (0-9), isolates all available indices for that specific class, 
       and randomly selects the exact required amount without replacement.
    4. Shuffles the final list to ensure batches are heterogeneous during training.
    """
    samples_per_class = int(num_samples_total / num_classes)

    # Dynamically extract labels based on the dataset's internal structure
    if hasattr(dataset, 'targets'):
        all_labels = np.array(dataset.targets)  # Native torchvision dataset (if CIFAR10)
    else:
        all_labels = dataset.tensors[1].numpy() # Custom PyTorch TensorDataset (if fake dataset)

    stratified_indices = []

    # Iterate through classes to enforce uniform distribution
    for class_idx in range(num_classes):
        # Locate all absolute index positions for the current class
        valid_indices = np.where(all_labels == class_idx)[0]

        # Randomly sample the exact uniform amount from this specific class pool
        chosen_indices = np.random.choice(valid_indices, samples_per_class, replace=False)
        stratified_indices.extend(chosen_indices)

    # Shuffle the aggregated indices to prevent sequential class clustering in batches
    np.random.shuffle(stratified_indices)
    return stratified_indices

def train_experiment(experiment_name, num_real, num_fake):
    """
    Executes a complete, isolated training run for a specific Real/Synthetic ratio.
    
    How it interacts with stratification:
    This function acts as the "Mixer." It calls `get_stratified_indices` to pull balanced 
    pointers for both the real and synthetic master datasets. It then uses PyTorch's `Subset` 
    to virtually slice these datasets, and `ConcatDataset` to fuse them together into a 
    brand new, perfectly proportioned 50,000-image dataloader.
    """
    print(f"\n{'='*60}")
    print(f"🚀 STARTING EXPERIMENT: {experiment_name}")
    print(f"📊 Mix: {num_real} Real Images | {num_fake} Fake Images")
    print(f"{'='*60}")

    # 1. Dynamically construct the stratified dataset mix
    if num_real > 0:
        real_indices = get_stratified_indices(real_trainset, num_real)
        real_subset = Subset(real_trainset, real_indices)

    if num_fake > 0:
        fake_indices = get_stratified_indices(fake_trainset, num_fake)
        fake_subset = Subset(fake_trainset, fake_indices)

    # Handle edge cases (0% or 100% mixes) cleanly
    if num_real == 0:
        mixed_dataset = fake_subset
    elif num_fake == 0:
        mixed_dataset = real_subset
    else:
        mixed_dataset = ConcatDataset([real_subset, fake_subset])

    mixed_loader = DataLoader(mixed_dataset, batch_size=64, shuffle=True)

    # 2. ISOLATION: Instantiate a completely new model architecture with fresh weights
    # This prevents the network from inheriting knowledge (weight leakage) from the previous loop!
    model = CIFARClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    epochs = 50

    # 3. Execution of the Supervised Training Loop
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for inputs, labels in mixed_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()
        epoch_loss = running_loss / len(mixed_loader)
        epoch_acc = 100 * correct / total
        epoch_time = time.time() - start_time

        # Print reduced telemetry to maintain a clean terminal output during multi-hour runs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Epoch {epoch + 1:02d}/{epochs}] Loss: {epoch_loss:.3f} | Acc: {epoch_acc:.2f}% | Time: {epoch_time:.1f}s")

    # 4. Save the explicitly named experiment model
    # Ensure local directory fallback exists if not in Colab
    if is_colab:
        save_path = f'/content/drive/MyDrive/{experiment_name}.pth'
    else:
        os.makedirs('./model_weights', exist_ok=True)
        save_path = f'./model_weights/{experiment_name}.pth'

    torch.save(model.state_dict(), save_path)
    print(f"\n✅ {experiment_name} SAVED TO: {save_path}")

    # 5. VRAM CLEANUP
    # CRITICAL: Deep learning models hold onto GPU memory. By explicitly deleting the 
    # tensors and emptying the CUDA cache, we prevent Out-Of-Memory (OOM) crashes 
    # when the script attempts to load the next experiment.
    del model, mixed_loader, mixed_dataset, optimizer, criterion
    torch.cuda.empty_cache()


# ==========================================
# 4. EXECUTION PLAN
# ==========================================
# Define the discrete ablation steps. The sum of real + fake must always strictly equal 50,000.
experiments = [
    {"name": "cnn_100real_0fake", "num_real": 50000, "num_fake": 0},
    {"name": "cnn_70real_30fake", "num_real": 35000, "num_fake": 15000},
    {"name": "cnn_50real_50fake", "num_real": 25000, "num_fake": 25000},
    {"name": "cnn_30real_70fake", "num_real": 15000, "num_fake": 35000},
    {"name": "cnn_0real_100fake", "num_real": 0,     "num_fake": 50000}
]

# Execute the master loop
for exp in experiments:
    train_experiment(exp["name"], exp["num_real"], exp["num_fake"])

print("\n🎉 ALL ABLATION EXPERIMENTS COMPLETED SUCCESSFULLY!")