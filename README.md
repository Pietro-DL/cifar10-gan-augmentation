# Deep Generative Models for Data Augmentation: Repository Guide

This repository contains the codebase for generating artificial CIFAR-10 images using Generative Adversarial Networks (GANs) to evaluate their efficacy as a data augmentation technique for Convolutional Neural Networks (CNNs). 
The project explores the transition from a baseline conditional DCGAN to a Wasserstein GAN with Gradient Penalty (WGAN-GP), and ultimately benchmarks against State-of-the-Art models (StyleGAN2-ADA).

Below is the architectural index of the scripts used in this project, listed in order. 
Each entry breaks down the internal sections of the code to serve as a navigable roadmap for review.

---

## 1. `cDCGAN.py`
This script establishes the baseline Conditional Deep Convolutional Generative Adversarial Network. 
It is responsible for learning the initial statistical distribution of the CIFAR-10 dataset and synthesizing class-conditioned images.

*   **0. Hyper Parameters:** 
    *   Initializes the environmental variables, including the random seed for mathematical reproducibility, batch size (128), latent noise vector size (100), and the base channel depths of the feature maps for both networks.
*   **1. Data Loading:** 
    *   Handles the ingestion of the CIFAR-10 dataset and applies crucial spatial transformations. 
    Specifically, it normalizes the RGB pixel values from standard bounds to the `[-1, 1]` range, ensuring strict mathematical compatibility with the Generator's final `Tanh` activation function.
*   **2.a Generator Architecture:** 
    *   Defines the `Generator` module. It details the projection of the 1D noise and one-hot label vectors into spatial feature maps, their concatenation, and the subsequent progressive upsampling via transposed convolutions to synthesize a 3x32x32 RGB image.
*   **2.b Discriminator Architecture:** 
    *   Defines the `Discriminator` module. It illustrates the programmatic technique of spatially expanding the 1D condition label into a full 32x32 tensor, combining it directly with the image channels, and downsampling the input through strided convolutions to output a single authenticity probability via a `Sigmoid` activation.
*   **3. Initialization & Device Setup:** 
    *   Detects the available hardware (CPU vs. CUDA) and applies a custom normal weight initialization function (`weights_init`) to the convolutional and batch normalization layers of both networks to stabilize early training.
*   **4. Optimizers and Loss Function:** 
    *   Initializes the Binary Cross-Entropy (BCE) loss criterion and sets the target conventions (`1.0` for real data, `0.0` for fake data). It also configures the Adam optimizers for both networks with GAN-specific hyperparameters.
*   **5. The Training Loop:** 
    *   Executes the two-phase zero-sum minimax game. 
    *   *Phase 1:* Updates the Discriminator using `.detach()` on fake images to safely sever the computational graph and prevent gradient leakage into the Generator.
    *   *Phase 2:* Updates the Generator by pushing non-detached fake images through the Discriminator to backpropagate the error. 
    *   This section also periodically evaluates a fixed noise batch for visual progress tracking and saves the final `.pth` weights to disk at the end of the epoch cycle.

## 2. `WGAN.py`
This script represents a major structural and mathematical upgrade over the baseline DCGAN. It implements a Wasserstein GAN with Gradient Penalty (WGAN-GP) to eliminate vanishing gradients and mode collapse, and replaces standard convolutions with high-capacity Residual Networks (ResNets) to improve spatial feature extraction.

*   **0. Hyperparameters:** 
    *   Introduces WGAN-GP specific parameters that dictate the new training dynamics, notably `n_critic = 5` (number of Critic updates per Generator update) and `lambda_gp = 10` (the weight of the gradient penalty). 
    *   It also adjusts the Adam optimizer's momentum parameter to `b1 = 0.0` as mathematically recommended for Wasserstein stability[cite: 3].
*   **1. Data Loading:** 
    *   Maintains the identical CIFAR-10 data pipeline from the baseline, keeping the crucial `[-1, 1]` spatial normalizations to match the Generator's output.
*   **2. WGAN-GP Architectures (ResNet Upgrade):** 
    *   Overhauls the network structures using custom `ResBlockUp` (Generator) and `ResBlockDown` (Critic) classes, which utilize skip-connections to preserve high-frequency spatial data across deep layers. 
    *   Critically, the Critic drops the final `Sigmoid` activation to output raw, unbounded continuous scores instead of probabilities, and replaces `BatchNorm2d` with `InstanceNorm2d` to ensure images are evaluated completely independently.
*   **4. The Gradient Penalty Function:** 
    *   Defines the `gradient_penalty` function required to enforce the 1-Lipschitz continuity constraint of the Wasserstein distance. 
    *   It actively uses PyTorch's `autograd.grad` to compute the derivatives of the Critic's scores with respect to interpolated real/fake images, applying an L2 norm penalty if the gradient magnitude deviates from 1.
*   **5. The WGAN-GP Training Loop:** 
    *   Replaces the Binary Cross-Entropy (BCE) minimax game with the continuous Earth Mover's (Wasserstein) Distance.
    *   *Phase 1 (Critic):* Runs `n_critic` times per loop. It calculates the difference between Fake and Real scores, adds the computed Gradient Penalty, and backpropagates to maximize the distributional distance. 
    *   *Phase 2 (Generator):* Runs exactly once per loop. It passes non-detached fake images through the Critic and minimizes the negative fake score to actively fool the Critic's evaluation.
*   **6. Plotting and Saving Final Assets:** 
    *   Generates a continuous line plot of the Generator and Critic losses across all iterations and exports the final `.pth` structural weights to disk.

## 3. `CNN_training.py`
This script evaluates the utility of the generated synthetic data by training a Convolutional Neural Network (CNN) classifier on a hybrid dataset combining real CIFAR-10 images with the synthetic generations. 

*   **1. Hardware & Datasets (The Hybrid Pipeline):** 
    *   Constructs the augmented training pipeline by applying standard geometric augmentations (crops, flips, rotations) to the real images and concatenating them with the synthetic tensor. 
    *   Crucially implements a `target_transform` to force the real Python integer labels into PyTorch tensors, ensuring mathematical compatibility during the dataset merge. 
    *   Prepares a strictly non-augmented test set to guarantee an objective evaluation baseline.
*   **2. The Classifier Architecture:** 
    *   Defines a high-capacity VGG-style CNN featuring three sequential convolutional blocks (Conv2d, BatchNorm, ReLU, MaxPool) and a linear classification head. 
    *   Explicitly omits artificial Dropout layers, relying entirely on the injected synthetic data to act as the primary structural regularizer.
*   **3. Supervised Training Loop:** 
    *   Executes a standard supervised classification pipeline using Cross-Entropy Loss and an Adam optimizer. 
    *   Implements a `StepLR` scheduler to decay the learning rate by a factor of 0.1 every 20 epochs, preventing the network from overshooting local minima during late-stage convergence. 
    *   Tracks granular batch-level and epoch-level statistics (Loss and Accuracy), ultimately exporting the final `.pth` classification weights to disk for downstream analysis.ù

## 4. `Comparison_script.py`
This script executes the final objective evaluation of the trained CNN classifiers. It systematically compares the baseline model (trained purely on real data) against the augmented models to quantify the downstream impact of synthetic data injection.

*   **1. Setup & Dynamic Pathing:** 
    *   Utilizes PyTorch and Python's `os` module to dynamically resolve the repository directory, ensuring seamless cross-platform execution on external machines.
*   **2. Defining the Classifier Architecture:** 
    *   Reinstantiates the architectural blueprint of the VGG-style CNN to provide the necessary spatial mapping for PyTorch to successfully load the pre-trained `.pth` weights into memory.
*   **3. Preparing the Strict Test Set:** 
    *   Loads the CIFAR-10 test set with a strict exclusion of geometric augmentations, providing a mathematically sterile baseline for evaluation.
*   **4. Evaluation Function:** 
    *   Executes model inference using `torch.no_grad()` to sever the computational graph and conserve memory. 
    *   Leverages `scikit-learn` to extract both global macro-averaged metrics (Accuracy, Precision, Recall, F1) and granular per-class arrays to identify localized performance shifts.
*   **5. Execute and Plot:** 
    *   Generates comprehensive visual analytics, including side-by-side Seaborn Confusion Matrices and grouped bar charts plotting localized Precision and Recall deviations from the global mean.Exports these visuals directly to the `/images` directory.

## 5. `Ablation_training.py`
This script orchestrates the automated ablation study, sequentially training multiple CNN classifiers on strictly controlled ratios of real and synthetic data to map the optimal mathematical regularizer limit.

*   **1. Memory Optimization:** 
    *   Loads the master Real and Synthetic datasets into CPU RAM exactly once, bypassing redundant disk read operations during the automated loop transitions.
    *   Applies a pixel-clamping correction to the synthetic dataset to guarantee mathematical parity with the `[-1, 1]` normalized real images.
*   **3. The Automated Ablation Pipeline:** 
    *   **Stratification (`get_stratified_indices`):** Implements a custom subsetting algorithm that guarantees perfect class distribution regardless of the requested dataset size. It dynamically extracts label structures from both `torchvision` and `TensorDataset` formats, ensuring the network is never subjected to categorical imbalance.
    *   **Experiment Isolation (`train_experiment`):** Acts as the dataset mixer and training orchestrator. For every iteration, it instantiates a completely fresh CNN architecture to prevent residual weight leakage between experiments.
    *   **VRAM Management:** Explicitly deletes computation graphs and dataloaders at the conclusion of each experiment, triggering `torch.cuda.empty_cache()` to prevent Out-Of-Memory (OOM) failures over the multi-hour ablation run.
*   **4. Execution Plan:** 
    *   Iterates through a predefined dictionary of ablation steps (e.g., 30% Fake, 50% Fake, 70% Fake, 100% Fake), automatically generating and archiving the resulting `.pth` weight files for downstream statistical evaluation.