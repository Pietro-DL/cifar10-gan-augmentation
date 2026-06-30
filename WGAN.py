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
except ImportError:
    print("Not running in Google Colab. Images will be saved locally.")
    image_dir = './cWGAN_GP_Images'
    os.makedirs(image_dir, exist_ok=True)

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
workers = 2
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

class ResBlockUp(nn.Module):
    """
    Residual Block for the Generator.
    Purpose: Progressively double the spatial resolution (Height/Width) 
             while adjusting the channel depth.
    Input Shape:  [Batch, in_ch, H, W]
    Output Shape: [Batch, out_ch, 2H, 2W]
    """
    def __init__(self, in_ch, out_ch):
        super(ResBlockUp, self).__init__()
        
        # --- THE MAIN PATH ---
        # This path does the heavy lifting: learning new textures and expanding the image.
        self.main = nn.Sequential(
            # 1. Normalize the incoming feature maps. 
            # Shape remains: [Batch, in_ch, H, W]
            nn.BatchNorm2d(in_ch),
            
            # 2. Apply non-linearity. 
            # Shape remains: [Batch, in_ch, H, W]
            nn.ReLU(True),
            
            # 3. Nearest-Neighbor Upsampling. 
            # This physically doubles the spatial grid. No weights are learned here.
            # Shape changes: [Batch, in_ch, H*2, W*2]
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # 4. First Convolution.
            # Maps the old channel depth to the new desired channel depth.
            # Kernel 3, Stride 1, Padding 1 ensures spatial dimensions do not shrink.
            # Shape changes: [Batch, out_ch, H*2, W*2]
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False),
            
            # 5. Normalize the newly created channels.
            # Shape remains: [Batch, out_ch, H*2, W*2]
            nn.BatchNorm2d(out_ch),
            
            # 6. Apply non-linearity.
            # Shape remains: [Batch, out_ch, H*2, W*2]
            nn.ReLU(True),
            
            # 7. Second Convolution (Refinement).
            # Learns complex features within the newly expanded resolution.
            # Shape remains: [Batch, out_ch, H*2, W*2]
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        )
        
        # --- THE SHORTCUT (SKIP CONNECTION) ---
        # This path bypasses the deep convolutions to pass raw, high-frequency 
        # spatial data directly forward, preventing the "washing out" of edges.
        self.shortcut = nn.Sequential(
            # 1. Double the spatial grid to match the Main Path's output size.
            # Shape changes: [Batch, in_ch, H*2, W*2]
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # 2. 1x1 Convolution.
            # This acts as a mathematical "channel mixer". It does not look at neighboring 
            # pixels (kernel=1), it just mathematically compresses/expands the in_ch to out_ch 
            # so the shortcut tensor can be legally added to the main tensor.
            # Shape changes: [Batch, out_ch, H*2, W*2]
            nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        # Element-wise addition of the learned features (main) and the raw features (shortcut).
        # Both tensors are exactly [Batch, out_ch, H*2, W*2].
        return self.main(x) + self.shortcut(x)


class ResBlockDown(nn.Module):
    """
    Residual Block for the Critic.
    Purpose: Progressively halve the spatial resolution (Height/Width) 
             while adjusting the channel depth, evaluating image realism.
    Input Shape:  [Batch, in_ch, H, W]
    Output Shape: [Batch, out_ch, H/2, W/2]
    """
    def __init__(self, in_ch, out_ch):
        super(ResBlockDown, self).__init__()
        
        # --- THE MAIN PATH ---
        self.main = nn.Sequential(
            # 1. Normalize the incoming feature maps.
            # CRITICAL WGAN-GP RULE: We use InstanceNorm instead of BatchNorm so 
            # each image in the batch is evaluated completely independently.
            # Shape remains: [Batch, in_ch, H, W]
            nn.InstanceNorm2d(in_ch, affine=True),
            
            # 2. Apply non-linearity (LeakyReLU prevents dead gradients in the Critic).
            # Shape remains: [Batch, in_ch, H, W]
            nn.LeakyReLU(0.2, inplace=True),
            
            # 3. First Convolution (Feature Extraction).
            # Evaluates the current resolution.
            # Shape remains: [Batch, in_ch, H, W]
            nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1, bias=False),
            
            # 4. Normalize intermediate features.
            # Shape remains: [Batch, in_ch, H, W]
            nn.InstanceNorm2d(in_ch, affine=True),
            
            # 5. Apply non-linearity.
            # Shape remains: [Batch, in_ch, H, W]
            nn.LeakyReLU(0.2, inplace=True),
            
            # 6. Second Convolution (Downsampling).
            # Kernel 3, Stride 2 physically halves the spatial dimensions while 
            # increasing the channel depth to out_ch.
            # Shape changes: [Batch, out_ch, H/2, W/2]
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False) 
        )
        
        # --- THE SHORTCUT (SKIP CONNECTION) ---
        self.shortcut = nn.Sequential(
            # 1. 1x1 Strided Convolution.
            # We must halve the resolution and change the channels to match the main path.
            # A 1x1 conv with stride=2 skips every other pixel, effectively downsampling 
            # the image while simultaneously doing the math to map in_ch to out_ch.
            # Shape changes: [Batch, out_ch, H/2, W/2]
            nn.Conv2d(in_ch, out_ch, 1, stride=2, padding=0, bias=False)
        )

    def forward(self, x):
        # The final output is the sum of the deep features and the raw shortcut.
        return self.main(x) + self.shortcut(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Project the 110-dim concatenated vector (100 noise + 10 label) into a 4x4 spatial map
        self.fc = nn.Linear(nz + label_dim, ngf * 4 * 4)

        self.res_blocks = nn.Sequential(
            ResBlockUp(ngf, ngf),       # 4x4 -> 8x8
            ResBlockUp(ngf, ngf // 2),  # 8x8 -> 16x16
            ResBlockUp(ngf // 2, ngf // 4) # 16x16 -> 32x32
        )

        self.final = nn.Sequential(
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),
            nn.Conv2d(ngf // 4, nc, 3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Flatten noise and labels to 2D matrices
        noise = noise.view(noise.size(0), -1)
        labels = labels.view(labels.size(0), -1)

        # Concatenate in 1D, then project and reshape to 4D
        x = torch.cat([noise, labels], dim=1)
        x = self.fc(x)
        x = x.view(-1, ngf, 4, 4)

        x = self.res_blocks(x)
        return self.final(x)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # Initial convolution to handle the 13-channel input (3 img + 10 label)
        self.init_conv = nn.Conv2d(nc + label_dim, ndf // 4, 3, stride=1, padding=1)

        self.res_blocks = nn.Sequential(
            ResBlockDown(ndf // 4, ndf // 2), # 32x32 -> 16x16
            ResBlockDown(ndf // 2, ndf),      # 16x16 -> 8x8
            ResBlockDown(ndf, ndf)            # 8x8 -> 4x4
        )

        self.final = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(ndf * 4 * 4, 1) # Output a single raw score
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

# =====================================================
# 4. The Gradient Penalty Function
# =====================================================

def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

# =====================================================
# 5. The WGAN-GP Training Loop
# =====================================================
import torch.nn.functional as F

# Create a fixed batch of 64 noise vectors and random labels for visual tracking
fixed_noise = torch.randn(64, nz, device=device)
fixed_labels_int = torch.randint(0, label_dim, (64,), device=device)
fixed_labels_onehot = F.one_hot(fixed_labels_int, num_classes=label_dim).float()

# Lists to keep track of progress and losses
img_list = []
G_losses = []
C_losses = [] # Changed from D_losses to C_losses

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
        # (1) Update Critic Network: Maximize (Real Score - Fake Score)
        # In PyTorch, we minimize, so we do: Fake Score - Real Score + Penalty
        # -----------------------------------------------------------
        netC.zero_grad()

        # 1.a) Pass Real Images through the Critic
        real_scores = netC(real_images, class_labels_onehot).view(-1)
        mean_real_score = real_scores.mean()

        # 1.b) Generate Fake Images
        noise = torch.randn(b_size, nz, device=device)
        fake_images = netG(noise, class_labels_onehot)

        # 1.c) Pass Fake Images through the Critic
        # (Using .detach() so we don't calculate gradients for the Generator yet)
        fake_scores = netC(fake_images.detach(), class_labels_onehot).view(-1)
        mean_fake_score = fake_scores.mean()

        # 1.d) Calculate the Gradient Penalty (Using Professor's function)

        # Create random alphas [Batch, 1, 1, 1]
        alpha = torch.rand(b_size, 1, 1, 1, device=device)

        # Blend the real and fake images (This is the 'x')
        # Requires grad is TRUE so PyTorch tracks the Pixel Gradient
        interpolates = (alpha * real_images + ((1 - alpha) * fake_images.detach())).requires_grad_(True)

        # Pass blended images to Critic (This is the 'y')
        d_interpolates = netC(interpolates, class_labels_onehot)

        # Call the professor's pure math function and save to a uniquely named variable!
        gp_val = gradient_penalty(y=d_interpolates, x=interpolates)

        # 1.e) Calculate Total Critic Loss
        # Mathematically: We want to push Fake scores DOWN and Real scores UP.
        loss_C = mean_fake_score - mean_real_score + (lambda_gp * gp_val)

        # 1.f) Backpropagate and Update Critic Weights
        loss_C.backward()
        optimizerC.step()

        # -----------------------------------------------------------
        # (2) Update Generator Network
        # WGAN-GP explicitly trains the Critic more often than the Generator.
        # We only update the Generator every `n_critic` (e.g., 5) steps.
        # -----------------------------------------------------------
        if i % n_critic == 0:
            netG.zero_grad()

            # 2.a) We need a fresh forward pass of the fake images through the Critic
            # This time WITHOUT .detach() so the gradients can flow all the way back to G
            fake_scores_for_G = netC(fake_images, class_labels_onehot).view(-1)

            # 2.b) Calculate Generator Loss
            # The Generator wants the Critic to give its fake images a HIGH score.
            # To minimize this in PyTorch, we make the score negative.
            loss_G = -fake_scores_for_G.mean()

            # 2.c) Backpropagate and Update Generator Weights
            loss_G.backward()
            optimizerG.step()

        # Save Losses for plotting
        G_losses.append(loss_G.item())
        C_losses.append(loss_C.item())

       # Print Progress every 50 batches
        if i % 50 == 0:
            # We also calculate the Wasserstein Distance for logging!
            # W-Distance = Real Score - Fake Score (Higher is better, meaning distributions are far apart)
            w_dist = mean_real_score.item() - mean_fake_score.item()
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                  f'Loss_C: {loss_C.item():.4f} Loss_G: {loss_G.item():.4f} '
                  f'W-Distance: {w_dist:.4f} Penalty: {gp_val.item():.4f}')

    # === End of Epoch Evaluation ===
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
save_path_G = '/content/drive/MyDrive/cWGAN_GP_netG.pth'
save_path_C = '/content/drive/MyDrive/cWGAN_GP_netC.pth'

# Save the learned parameters
torch.save(netG.state_dict(), save_path_G)
torch.save(netC.state_dict(), save_path_C)

print("Training Complete! WGAN-GP Weights successfully saved to Google Drive!")