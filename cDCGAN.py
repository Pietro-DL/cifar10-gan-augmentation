import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

try:
    from google.colab import drive
    drive.mount('/content/drive')
    is_colab = True
except ImportError:
    is_colab = False


# Lists to keep track of progress
img_list = []

# =====================================================
# 0. Hyper Parameters
# =====================================================

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

image_size = 32      # CIFAR-10 image size
batch_size = 128     # Standard batch size for GANs
workers = 0          # Number of CPU cores for data loading
num_epochs = 100     # Number of training epochs (start with 50-100)
nz = 100           # Size of z latent vector (noise)
label_dim = 10     # Number of classes in CIFAR-10
ngf = 64           # Base size of feature maps in generator
ndf = 64           # Base size of feature maps in discriminator
nc = 3             # Number of channels (3 for RGB)

# =====================================================
# 1. Data Loading and Preprocessing
# =====================================================
"""
This section handles the ingestion of the CIFAR-10 dataset.
A critical preprocessing step is the normalization of RGB pixel values 
from [0, 1] to the range [-1, 1]. This ensures mathematical compatibility 
with the Generator's final Tanh activation function, allowing the Discriminator 
to evaluate real and synthetic images on an identical numerical scale.
"""

# CIFAR-10 dataset transformations
transform = transforms.Compose([
    transforms.Resize(image_size), # CIFAR-10 is already 32x32, but good for safety
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) 
])

# Load CIFAR-10 instead of MNIST
cifar10_data = dset.CIFAR10(root='./data',
                            train=True,
                            transform=transform,
                            download=True)

dataloader = torch.utils.data.DataLoader(dataset=cifar10_data,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=workers)

# =====================================================
# 2. Generator and Discriminator Architectures
# =====================================================
"""
This section defines the Conditional DCGAN architecture. 
To enable class-conditioned generation, both networks process joint representations:
- The Generator concatenates a random noise vector (z) with a one-hot label vector (y),
it processes those vectors, combines them in the spatial domain so that it learns to generate images conditioned on the label.
- The Discriminator expands the one-hot label into a spatial feature map and concatenates 
  it directly onto the RGB image channels, ensuring the class condition acts as a 
  persistent global context across all spatial convolutions.
"""

# =====================================================
# 2.a Generator and Discriminator Architectures
# =====================================================
# --- Generator Spatial Parameters ---
# Parameters to project 1x1 input to 4x4 spatial dimensions
proj_kernel = 4
proj_stride = 1
proj_pad = 0

# Parameters to exactly double the spatial dimensions (e.g., 4x4 -> 8x8)
up_kernel = 4
up_stride = 2
up_pad = 1


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # --- NOISE PATH ---
        # Input: Noise vector (z) of shape [Batch, 100, 1, 1]
        self.noise_block = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 2, proj_kernel, proj_stride, proj_pad, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
            # Output: z_out has a shape of [Batch, 128, 4, 4]
         )
        
        # --- LABEL PATH ---
        # Input: One-hot label vector of shape [Batch, 10, 1, 1]
        self.label_block = nn.Sequential(
            nn.ConvTranspose2d(label_dim, ngf * 2, proj_kernel, proj_stride, proj_pad, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
            # Output: l_out has a shape of [Batch, 128, 4, 4]
         )
        
        # --- COMBINED PATH (MAIN) ---
        # Input: Concatenated x of shape [Batch, 256, 4, 4] (128 + 128 channels)
        self.main = nn.Sequential(
            # Upsample 1: 4x4 -> 8x8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, up_kernel, up_stride, up_pad, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Output shape: [Batch, 128, 8, 8]
            
            # Upsample 2: 8x8 -> 16x16
            nn.ConvTranspose2d(ngf * 2, ngf, up_kernel, up_stride, up_pad, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Output shape: [Batch, 64, 16, 16]
            
            # Upsample 3: 16x16 -> 32x32
            nn.ConvTranspose2d(ngf, nc, up_kernel, up_stride, up_pad, bias=False),
            # The Tanh activation function squashes the output values to the range [-1, 1]
            # CIFAR-10 images where squashed into that exact same $[-1, 1]$ range. 
            # This is why we used transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # so that the discriminator can learn to distinguish between real and fake images
            nn.Tanh()
            # Final Output shape: [Batch, 3, 32, 32] -> An RGB image!
        )

    def forward(self, noise, labels):
        # 1. Reshape 1D vectors into 4D tensors representing "1x1 images"
        # View dynamically adjusts to the current batch size (-1)
        noise = noise.view(-1, nz, 1, 1)
        labels = labels.view(-1, label_dim, 1, 1)
        
        # 2. Project both into 4x4 spatial maps
        z_out = self.noise_block(noise)
        l_out = self.label_block(labels)
        
        # 3. Concatenate along the channel dimension (dim=1)
        x = torch.cat([z_out, l_out], dim=1) 
        
        # 4. Generate the final 32x32 image
        return self.main(x)


# =====================================================
# 2.b DISCRIMINATOR
# =====================================================

# --- Discriminator Spatial Parameters ---
# Parameters to exactly halve the spatial dimensions (e.g., 32x32 -> 16x16)
down_kernel = 4
down_stride = 2
down_pad = 1

# Parameters to reduce a 4x4 spatial dimension into a 1x1 scalar probability
scalar_kernel = 4
scalar_stride = 1
scalar_pad = 0

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # The Discriminator takes the concatenated Image (3 channels) + Label Map (10 channels)
        # Total input channels = nc + label_dim = 3 + 10 = 13 channels
        
        self.main = nn.Sequential(
            # Input: [Batch, 13, 32, 32]
            # Downsample 1: 32x32 -> 16x16
            nn.Conv2d(nc + label_dim, ndf, down_kernel, down_stride, down_pad, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Output shape: [Batch, 64, 16, 16]

            # Downsample 2: 16x16 -> 8x8
            nn.Conv2d(ndf, ndf * 2, down_kernel, down_stride, down_pad, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Output shape: [Batch, 128, 8, 8]

            # Downsample 3: 8x8 -> 4x4
            nn.Conv2d(ndf * 2, ndf * 4, down_kernel, down_stride, down_pad, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Output shape: [Batch, 256, 4, 4]

            # Final Layer: Collapse 4x4 into a 1x1 scalar
            nn.Conv2d(ndf * 4, 1, scalar_kernel, scalar_stride, scalar_pad, bias=False),
            nn.Sigmoid() 
            # Output shape: [Batch, 1, 1, 1] -> A single probability between 0 and 1!
        )

    def forward(self, image, labels):
        # 1. Retrieve the batch size
        b_size = image.size(0)
        
        # 2. Reshape the 1D one-hot labels to [Batch, 10, 1, 1]
        labels = labels.view(b_size, label_dim, 1, 1)
        
        # 3. THE TRICK: Expand the 1x1 label tensor to match the image size (32x32)
        # This creates a "label channel" where every pixel in the 32x32 grid contains the label vector
        labels = labels.expand(b_size, label_dim, image.size(2), image.size(3))
        # Now labels shape is: [Batch, 10, 32, 32]
        
        # 4. Concatenate the Image and the Label Channel on dim=1
        x = torch.cat([image, labels], dim=1)
        # Combined shape: [Batch, 13, 32, 32]
        
        # 5. Evaluate
        return self.main(x)

# =====================================================
# 3. Initialization & Device Setup
# =====================================================

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Initialize Conv and ConvTranspose weights
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Initialize BatchNorm weights and biases
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Instantiate the Generator and apply the weights_init function
netG = Generator().to(device)
netG.apply(weights_init)
print(netG)

# Instantiate the Discriminator and apply the weights_init function
netD = Discriminator().to(device)
netD.apply(weights_init)
print(netD)

# =====================================================
# 4. Optimizers and Loss Function
# =====================================================
"""
We utilize the Binary Cross Entropy (BCE) Loss. Since the Discriminator acts 
as a binary classifier estimating authenticity, BCE naturally measures the error 
between the network's predicted probability (0 to 1) and the actual mathematical target.
"""
# Initialize BCELoss function
criterion = nn.BCELoss()

# Establish convention for real and fake labels during training targets:
# 1.0 represents a ground-truth "Real" image from the CIFAR-10 dataset.
# 0.0 represents a "Fake" image synthesized by the Generator.
# The Generator's ultimate goal is to force the Discriminator to predict 1.0 for fake images.
real_label_val = 1.0
fake_label_val = 0.0

# Setup Adam optimizers for both G and D using standard GAN hyperparameters
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# =====================================================
# 5. The Training Loop
# =====================================================
"""
SUMMARY OF THE cDCGAN TRAINING LOOP
The training process is a zero-sum minimax game divided into two distinct phases per batch:

PHASE 1: Update the Discriminator (D)
Objective: Maximize log(D(x|y)) + log(1 - D(G(z|y)|y))
- Step 1.1: Pass a batch of Real images to D. Calculate BCE loss against a target of 1.0 (Real). Accumulate gradients.
- Step 1.2: Generate a batch of Fake images using G. 
- Step 1.3: Pass the Fake images to D. Use .detach() to sever the graph and prevent gradients from flowing back into G.
- Step 1.4: Calculate BCE loss against a target of 0.0 (Fake). Accumulate gradients and update D's weights.

PHASE 2: Update the Generator (G)
Objective: Maximize log(D(G(z|y)|y)) (i.e., fool the Discriminator)
- Step 2.1: Pass the same Fake images to D, but this time WITHOUT .detach() to keep the computational graph intact.
- Step 2.2: Calculate BCE loss against a target of 1.0 (Real). A high loss here means G failed to fool D.
- Step 2.3: Backpropagate the error through D directly into G's weights, and update G.
"""

# Create a fixed batch of 64 noise vectors and random labels
fixed_noise = torch.randn(64, nz, device=device)
fixed_labels_int = torch.randint(0, label_dim, (64,), device=device)
fixed_labels_onehot = F.one_hot(fixed_labels_int, num_classes=label_dim).float()

# Lists to keep track of progress
img_list = []

print("Starting Training Loop...")

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        
        # Extract images and integer labels from the dataloader
        real_images = data[0].to(device)
        class_labels_int = data[1].to(device)
        b_size = real_images.size(0)
        
        # Convert integer labels to One-Hot Vectors [Batch, 10] (integers from 0 to 9 are mapped into binary vectors of length 10)
        # and cast to float so they can be processed by neural net layers
        class_labels_onehot = F.one_hot(class_labels_int, num_classes=label_dim).float()

        # -----------------------------------------------------------
        # (1) Update D network: maximize log(D(x|y)) + log(1 - D(G(z|y)|y))
        # -----------------------------------------------------------
        netD.zero_grad()

        # --- Train with All-Real Batch ---
        # Create a tensor of 1s for the real labels
        label_tensor = torch.full((b_size,), real_label_val, dtype=torch.float, device=device)
        
        # Forward pass real batch through D
        # Note: netD outputs shape [Batch, 1, 1, 1], so we use .view(-1) to flatten it to [Batch]
        output = netD(real_images, class_labels_onehot).view(-1)
        
        # Calculate loss on all-real batch
        errD_real = criterion(output, label_tensor)
        errD_real.backward() # Calculate gradients for D
        D_x = output.mean().item() # For logging

        # --- Train with All-Fake Batch ---
        # Generate batch of latent vectors (noise)
        noise = torch.randn(b_size, nz, device=device)
        
        # Generate fake image batch with G
        fake_images = netG(noise, class_labels_onehot)
        
        # Fill label tensor with 0s for fake labels
        label_tensor.fill_(fake_label_val)
        
        # Classify all fake batch with D
        # CRITICAL: We use .detach() here so gradients don't flow back into G
        output = netD(fake_images.detach(), class_labels_onehot).view(-1)
        
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label_tensor)
        errD_fake.backward() # Calculate gradients for D
        D_G_z1 = output.mean().item() # For logging
        
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        # -----------------------------------------------------------
        # (2) Update G network: maximize log(D(G(z|y)|y))
        # -----------------------------------------------------------
        netG.zero_grad()
        
        # Fill label tensor with 1s (Generator's goal is to fool D)
        label_tensor.fill_(real_label_val)  
        
        # Since we just updated D, perform another forward pass of all-fake batch through D
        # NOTE: We do NOT detach here, because we WANT gradients to flow back to G!
        output = netD(fake_images, class_labels_onehot).view(-1)
        
        # Calculate G's loss based on this output
        errG = criterion(output, label_tensor)
        
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item() # For logging
        
        # Update G
        optimizerG.step()

        # -----------------------------------------------------------
        # (3) Print Progress
        # -----------------------------------------------------------
        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                  f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                  f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

    # === End of Epoch Evaluation ===
    # Notice this is indented 4 spaces, aligning with the "for i, data..." loop
    with torch.no_grad():
        # Generate images from the fixed noise
        fake_display = netG(fixed_noise, fixed_labels_onehot).detach().cpu()
    
    # Create a grid and save it to our list
    grid = vutils.make_grid(fake_display, padding=2, normalize=True)
    img_list.append(grid)
    
    # Plot the grid right in the notebook
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title(f"Generated Images at Epoch {epoch}")
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()

# === Saving Weights (Unindented, runs once at the very end) ===
if is_colab:
    save_path_G = '/content/drive/MyDrive/cDCGAN_netG.pth'
    save_path_D = '/content/drive/MyDrive/cDCGAN_netD.pth'
else:
    os.makedirs('./model_weights', exist_ok=True)
    save_path_G = './model_weights/cDCGAN_netG.pth'
    save_path_D = './model_weights/cDCGAN_netD.pth'

# Save the weights
torch.save(netG.state_dict(), save_path_G)
torch.save(netD.state_dict(), save_path_D)

print(f"Weights successfully saved to {save_path_G} and {save_path_D}!")