# =====================================================
# WGAN-GP: Imports and Setup
# =====================================================
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Google Colab Drive Mount
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    # Create a dedicated folder for WGAN-GP outputs
    image_dir = '/content/drive/MyDrive/cWGAN_GP_Images'
    os.makedirs(image_dir, exist_ok=True)
    print("Drive mounted and output directory created!")
    is_colab = True
except ImportError:
    print("Not running in Google Colab. Images will be saved locally.")
    image_dir = './cWGAN_GP_Images'
    os.makedirs(image_dir, exist_ok=True)
    is_colab = False

# =====================================================
# 0. Hyperparameters
# =====================================================
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Architectural parameters (Keeping our upgraded 128 capacity)
image_size = 32   
batch_size = 128
workers = 0
num_epochs = 100
nz = 100
label_dim = 10
ngf = 128
ndf = 128
nc = 3

# --- WGAN-GP Specific Hyperparameters ---
n_critic = 5       # Number of critic iterations per generator iteration
lambda_gp = 10     # Gradient penalty lambda hyperparameter
lr = 1e-4          # Learning rate (WGAN-GP uses 0.0001)
b1 = 0.0           # Adam beta1 (WGAN-GP recommends 0.0 instead of 0.5)
b2 = 0.9           # Adam beta2

# =====================================================
# 1. Data Loading (CIFAR-10)
# =====================================================
# Normalize to [-1, 1] because our Generator will still use a Tanh output
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

cifar10_data = dset.CIFAR10(root='./data', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset=cifar10_data, batch_size=batch_size, shuffle=True, num_workers=workers)

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# =====================================================
# 2. WGAN-GP Architectures (ResNet Upgrade)
# =====================================================
# Increased base capacity
ngf = 256  # Base size of feature maps in generator
ndf = 256  # Base size of feature maps in critic

# -----------------------------------------------------
# RESIDUAL BLOCK: UPSAMPLING (GENERATOR)
# -----------------------------------------------------
# --- Main Path ---
# 1) BatchNorm2d : [Batch, in_ch, H, W]   --> [Batch, in_ch, H, W]       (Normalizes incoming features)
# 2) ReLU        : [Batch, in_ch, H, W]   --> [Batch, in_ch, H, W]       (Applies non-linearity)
# 3) Upsample    : [Batch, in_ch, H, W]   --> [Batch, in_ch, 2H, 2W]     (Nearest-neighbor spatial doubling)
# 4) Conv2d      : [Batch, in_ch, 2H, 2W] --> [Batch, out_ch, 2H, 2W]    (Maps to new channel depth without shrinking)
# 5) BatchNorm2d : [Batch, out_ch, 2H, 2W]--> [Batch, out_ch, 2H, 2W]    (Normalizes expanded channels)
# 6) ReLU        : [Batch, out_ch, 2H, 2W]--> [Batch, out_ch, 2H, 2W]    (Applies non-linearity)
# 7) Conv2d      : [Batch, out_ch, 2H, 2W]--> [Batch, out_ch, 2H, 2W]    (Refines learned features)
# 
# --- Shortcut Path ---
# 1) Upsample    : [Batch, in_ch, H, W]   --> [Batch, in_ch, 2H, 2W]     (Matches spatial size of main path)
# 2) Conv2d(1x1) : [Batch, in_ch, 2H, 2W] --> [Batch, out_ch, 2H, 2W]    (Channel mixer to match out_ch for addition)
#
# Output = Main + Shortcut
class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlockUp, self).__init__()
        
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        )
        
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


# -----------------------------------------------------
# RESIDUAL BLOCK: DOWNSAMPLING (CRITIC)
# -----------------------------------------------------
# --- Main Path ---
# 1) InstanceNorm: [Batch, in_ch, H, W]   --> [Batch, in_ch, H, W]       (WGAN-GP constraint: Independent batch evaluation)
# 2) LeakyReLU   : [Batch, in_ch, H, W]   --> [Batch, in_ch, H, W]       (Prevents dead gradients in Critic)
# 3) Conv2d      : [Batch, in_ch, H, W]   --> [Batch, in_ch, H, W]       (Extracts features at current resolution)
# 4) InstanceNorm: [Batch, in_ch, H, W]   --> [Batch, in_ch, H, W]       (Normalizes intermediate features)
# 5) LeakyReLU   : [Batch, in_ch, H, W]   --> [Batch, in_ch, H, W]       (Applies non-linearity)
# 6) Conv2d      : [Batch, in_ch, H, W]   --> [Batch, out_ch, H/2, W/2]  (Stride=2 halves spatial dims, increases channels)
# 
# --- Shortcut Path ---
# 1) Conv2d(1x1) : [Batch, in_ch, H, W]   --> [Batch, out_ch, H/2, W/2]  (Stride=2 matches spatial halving and channel expansion)
#
# Output = Main + Shortcut
class ResBlockDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlockDown, self).__init__()
        
        self.main = nn.Sequential(
            nn.InstanceNorm2d(in_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False) 
        )
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=2, padding=0, bias=False)
        )

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


# -----------------------------------------------------
# GENERATOR ARCHITECTURE
# -----------------------------------------------------
# --- Forward Pass Sequential Flow ---
# 1) Concat(noise, label) : 100-dim + 10-dim --> [Batch, 110]           (Fuses noise and condition)
# 2) Linear (fc)          : [Batch, 110] --> [Batch, ngf * 16]      (Projects into high-dimensional space)
# 3) View (Reshape)       : [Batch, ngf * 16] --> [Batch, ngf, 4, 4]     (Folds into spatial 4x4 feature map)
# 4) ResBlockUp 1         : [Batch, ngf, 4, 4] --> [Batch, ngf, 8, 8]     (Doubles spatial resolution)
# 5) ResBlockUp 2         : [Batch, ngf, 8, 8] --> [Batch, ngf/2, 16, 16] (Doubles resolution, halves channels)
# 6) ResBlockUp 3         : [Batch, ngf/2, 16, 16] --> [Batch, ngf/4, 32, 32] (Doubles resolution, halves channels)
# 7) Final Sequential     : [Batch, ngf/4, 32, 32] --> [Batch, 3, 32, 32]     (BN, ReLU, Conv2d squashes to RGB image, Tanh)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(nz + label_dim, ngf * 4 * 4)

        self.res_blocks = nn.Sequential(
            ResBlockUp(ngf, ngf),       
            ResBlockUp(ngf, ngf // 2),  
            ResBlockUp(ngf // 2, ngf // 4) 
        )

        self.final = nn.Sequential(
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),
            nn.Conv2d(ngf // 4, nc, 3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        noise = noise.view(noise.size(0), -1)
        labels = labels.view(labels.size(0), -1)

        x = torch.cat([noise, labels], dim=1)
        x = self.fc(x)
        x = x.view(-1, ngf, 4, 4)

        x = self.res_blocks(x)
        return self.final(x)


# -----------------------------------------------------
# CRITIC ARCHITECTURE
# -----------------------------------------------------
# --- Forward Pass Sequential Flow ---
# 1) Expand Label         : [Batch, 10] --> [Batch, 10, 32, 32]    (Stretches label into a spatial channel)
# 2) Concat(img, label)   : 3-ch + 10-ch --> [Batch, 13, 32, 32]    (Fuses RGB and class condition)
# 3) Init Conv2d          : [Batch, 13, 32, 32] --> [Batch, ndf/4, 32, 32] (Initial feature extraction)
# 4) ResBlockDown 1       : [Batch, ndf/4, 32, 32] --> [Batch, ndf/2, 16, 16] (Halves resolution, doubles channels)
# 5) ResBlockDown 2       : [Batch, ndf/2, 16, 16] --> [Batch, ndf, 8, 8]     (Halves resolution, doubles channels)
# 6) ResBlockDown 3       : [Batch, ndf, 8, 8] --> [Batch, ndf, 4, 4]     (Halves resolution, maintains channels)
# 7) Final Sequential     : [Batch, ndf, 4, 4] --> [Batch, 1]             (LeakyReLU, Flatten, Linear maps to raw score)
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.init_conv = nn.Conv2d(nc + label_dim, ndf // 4, 3, stride=1, padding=1)

        self.res_blocks = nn.Sequential(
            ResBlockDown(ndf // 4, ndf // 2), 
            ResBlockDown(ndf // 2, ndf),      
            ResBlockDown(ndf, ndf)            
        )

        self.final = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(ndf * 4 * 4, 1) 
        )

    def forward(self, img, label):
        b_size = img.size(0)
        label = label.view(b_size, label_dim, 1, 1).expand(b_size, label_dim, img.size(2), img.size(3))

        x = torch.cat([img, label], dim=1)
        x = self.init_conv(x)
        x = self.res_blocks(x)
        return self.final(x)

# Initialization
netG = Generator().to(device)
netC = Critic().to(device)

optimizerC = optim.Adam(netC.parameters(), lr=lr, betas=(b1, b2))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(b1, b2))

print("ResNet Generator and Critic Initialized!")

# Initialization
netG = Generator().to(device)
netC = Critic().to(device)

optimizerC = optim.Adam(netC.parameters(), lr=lr, betas=(b1, b2))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(b1, b2))

print("ResNet Generator and Critic Initialized!")

# =====================================================
# 4. The Gradient Penalty Function
# =====================================================
"""
THE 1-LIPSCHITZ CONSTRAINT AND GRADIENT PENALTY
For the Earth Mover's (Wasserstein) Distance to be mathematically valid, the Critic network 
must be 1-Lipschitz continuous (its evaluation slope cannot exceed a steepness of 1 anywhere).
Instead of clipping weights, WGAN-GP enforces this by calculating the exact gradient of the 
Critic's score with respect to the input images, and penalizing the model if the magnitude 
(L2 norm) of this gradient deviates from 1.
"""
def gradient_penalty(y, x):
    """
    Computes the gradient penalty: lambda * (L2_norm(dy/dx) - 1)**2.
    Args:
        y: The Critic's evaluation scores [Batch, 1].
        x: The interpolated images [Batch, Channels, Height, Width].
    """
    # Create a dummy tensor of 1s to serve as the incoming gradient for autograd
    weight = torch.ones(y.size()).to(device)
    
    # torch.autograd.grad computes the partial derivatives of the output 'y' 
    # with respect to the input pixels 'x'.
    # CRITICAL: create_graph=True tells PyTorch to build a computational graph OF this 
    # gradient calculation, allowing the Adam optimizer to backpropagate through the penalty!
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    # Flatten the spatial dimensions of the gradient tensor
    dydx = dydx.view(dydx.size(0), -1)
    
    # Calculate the L2 norm (magnitude) of the gradients for each image in the batch
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    
    # Return the Mean Squared Error between the actual gradient norm and the target norm of 1
    return torch.mean((dydx_l2norm - 1)**2)


# =====================================================
# 5. The WGAN-GP Training Loop
# =====================================================
"""
SUMMARY OF THE WGAN-GP TRAINING LOOP
Unlike standard GANs that use BCE loss and probabilities, WGAN-GP uses the Wasserstein distance 
and outputs unbounded "realism scores". Because the Critic must be trained to near-optimality 
to provide accurate distance metrics, it is updated multiple times for every single Generator update.

PHASE 1: Update the Critic (C) -- Runs 'n_critic' times (e.g., 5 times)
Objective: Maximize the distance between Real Scores and Fake Scores + Enforce Gradient Penalty
- Step 1.1: Pass Real images through C to get 'mean_real_score'.
- Step 1.2: Pass Fake images (detached) through C to get 'mean_fake_score'.
- Step 1.3: Generate interpolated images between real and fake batches.
- Step 1.4: Calculate the Gradient Penalty on the interpolated images.
- Step 1.5: Calculate Critic Loss: (Fake Score - Real Score) + (lambda * Penalty).
            (In PyTorch, minimizing this equation pushes Real Scores -> +Infinity and Fake Scores -> -Infinity).
- Step 1.6: Backpropagate and update C.

PHASE 2: Update the Generator (G) -- Runs 1 time
Objective: Maximize the Critic's evaluation of the Fake images (Fool the Critic)
- Step 2.1: Pass Fake images (NOT detached) through C.
- Step 2.2: Calculate Generator Loss: -(mean_fake_score).
            (Minimizing a negative score forces the network to push the actual score towards +Infinity).
- Step 2.3: Backpropagate through C into G, and update G.
"""

# Create a fixed batch of 64 noise vectors and random labels for visual tracking across epochs
fixed_noise = torch.randn(64, nz, device=device)
fixed_labels_int = torch.randint(0, label_dim, (64,), device=device)
fixed_labels_onehot = F.one_hot(fixed_labels_int, num_classes=label_dim).float()

# Lists to keep track of progress and losses
img_list = []
G_losses = []
C_losses = [] 

print("Starting WGAN-GP Training Loop...")

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):

        # Extract images and integer labels from the dataloader
        real_images = data[0].to(device)
        class_labels_int = data[1].to(device)
        b_size = real_images.size(0)

        # Convert integer labels to One-Hot Vectors [Batch, 10]
        class_labels_onehot = F.one_hot(class_labels_int, num_classes=label_dim).float()

        # -----------------------------------------------------------
        # PHASE 1: Update Critic Network (Runs continuously)
        # -----------------------------------------------------------
        netC.zero_grad()

        # 1.a) Real Image Evaluation
        real_scores = netC(real_images, class_labels_onehot).view(-1)
        mean_real_score = real_scores.mean()

        # 1.b) Fake Image Generation
        noise = torch.randn(b_size, nz, device=device)
        fake_images = netG(noise, class_labels_onehot)

        # 1.c) Fake Image Evaluation (Detached)
        # Detaching prevents gradients from flowing back into the Generator during the Critic update
        fake_scores = netC(fake_images.detach(), class_labels_onehot).view(-1)
        mean_fake_score = fake_scores.mean()

        # 1.d) Gradient Penalty Calculation
        # Create random blend ratios (alpha) [Batch, 1, 1, 1]
        alpha = torch.rand(b_size, 1, 1, 1, device=device)

        # Blend real and fake images. requires_grad_(True) tracks the Pixel Gradients for the penalty.
        interpolates = (alpha * real_images + ((1 - alpha) * fake_images.detach())).requires_grad_(True)

        # Evaluate interpolated images
        d_interpolates = netC(interpolates, class_labels_onehot)

        # Compute the penalty using the explicit formula
        gp_val = gradient_penalty(y=d_interpolates, x=interpolates)

        # 1.e) Critic Loss Formulation & Update
        # By minimizing (Fake - Real), the optimizer pushes Fake down and Real up.
        loss_C = mean_fake_score - mean_real_score + (lambda_gp * gp_val)
        
        loss_C.backward()
        optimizerC.step()

        # -----------------------------------------------------------
        # PHASE 2: Update Generator Network (Runs every 'n_critic' steps)
        # -----------------------------------------------------------
        if i % n_critic == 0:
            netG.zero_grad()

            # 2.a) Fresh Evaluation of Fake Images (NOT Detached)
            # We keep the computational graph intact so the error can backpropagate to G's weights
            fake_scores_for_G = netC(fake_images, class_labels_onehot).view(-1)

            # 2.b) Generator Loss Formulation & Update
            # The Generator wants fake images to have a HIGH positive score. 
            # Minimizing the negative score achieves exactly this.
            loss_G = -fake_scores_for_G.mean()

            loss_G.backward()
            optimizerG.step()

        # Save Losses for plotting
        G_losses.append(loss_G.item())
        C_losses.append(loss_C.item())

        # -----------------------------------------------------------
        # 3. Print Progress and Metrics
        # -----------------------------------------------------------
        if i % 50 == 0:
            # Empirical Wasserstein Distance = Real Score - Fake Score
            # A higher positive distance indicates the Critic is successfully separating the distributions.
            w_dist = mean_real_score.item() - mean_fake_score.item()
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                  f'Loss_C: {loss_C.item():.4f} Loss_G: {loss_G.item():.4f} '
                  f'W-Distance: {w_dist:.4f} Penalty: {gp_val.item():.4f}')

    # === End of Epoch Visual Evaluation ===
    with torch.no_grad():
        fake_display = netG(fixed_noise, fixed_labels_onehot).detach().cpu()

    # Clear the massive scrolling text output (Prevents Colab from lagging)
    clear_output(wait=True)

    # Save the image to Drive only every 10 epochs (or the very last epoch)
    if epoch % 10 == 0 or epoch == (num_epochs - 1):
        save_path = f"{image_dir}/epoch_{epoch:03d}.png"
        vutils.save_image(fake_display, save_path, padding=2, normalize=True)
        print(f"Saved checkpoint image to: {save_path}")

    # Create grid and append to list for final visualization
    grid = vutils.make_grid(fake_display, padding=2, normalize=True)
    img_list.append(grid)

    # Display the newest image in the notebook
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title(f"Generated Images at Epoch {epoch}")
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()
    plt.close(fig) # Prevent RAM buildup

# =====================================================
# 6. Plotting and Saving Final Assets
# =====================================================

# Plot the training losses
plt.figure(figsize=(10,5))
plt.title("Generator and Critic Loss During WGAN-GP Training")
plt.plot(G_losses, label="Generator")
plt.plot(C_losses, label="Critic")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()

# Save the plot directly to Google Drive
plot_save_path = '/content/drive/MyDrive/cWGAN_GP_loss_plot.png'
plt.savefig(plot_save_path)
print(f"Loss plot saved to {plot_save_path}")
plt.show()

# === Saving Weights ===
if is_colab:
    save_path_G = '/content/drive/MyDrive/cWGAN_GP_netG.pth'
    save_path_C = '/content/drive/MyDrive/cWGAN_GP_netC.pth'
else:
    os.makedirs('./model_weights', exist_ok=True)
    save_path_G = './model_weights/cWGAN_GP_netG.pth'
    save_path_C = './model_weights/cWGAN_GP_netC.pth'

# Save the learned parameters
torch.save(netG.state_dict(), save_path_G)
torch.save(netC.state_dict(), save_path_C)

print(f"Training Complete! WGAN-GP Weights successfully saved to {save_path_G} and {save_path_C}!")