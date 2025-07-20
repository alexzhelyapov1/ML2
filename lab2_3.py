# %% [markdown]
# # Lab 02: Segmentation
# 
# In this laboratory work you will create pipeline for cancer cells segmentation starting from reading data to preprocessing, creating training setup, experimenting with models.
# 
# ## Part 1: Reading dataset
# 
# Write Dataset class inheriting regular `torch` dataset.
# 
# In this task we use small datset just to make this homework accessible for everyone, so please **do not** read all the data in constructor because it is not how it works for real life datasets. You need to read image from disk only when it is requesed (getitem).
# 
# Split data (persistently between runs) to train, val and test sets. Add corresponding parameter to dataset constructor.

# %%
# --- Imports ---
import os
import glob
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm.notebook import tqdm
import pandas as pd
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.segmentation import DiceScore
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Try importing segmentation_models_pytorch
try:
    import segmentation_models_pytorch as smp
    print(f"segmentation-models-pytorch version: {smp.__version__}")
    SMP_AVAILABLE = True
except ImportError:
    print("Warning: segmentation-models-pytorch not found.")
    print("Install it using: pip install segmentation-models-pytorch")
    SMP_AVAILABLE = False

# --- Constants ---
DATA_DIR = 'data'
IMAGE_DIR = os.path.join(DATA_DIR, 'Images')
MASK_DIR = os.path.join(DATA_DIR, 'Masks')

RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.20
VAL_SPLIT_SIZE = 0.15

IMG_HEIGHT = 256
IMG_WIDTH = 256
FOREGROUND_THRESHOLD = 127

BATCH_SIZE = 8
NUM_WORKERS = os.cpu_count() // 2
LEARNING_RATE = 1e-4
MAX_EPOCHS = 50
PATIENCE = 10

# Augmented training constants
AUG_MAX_EPOCHS = MAX_EPOCHS + 20
AUG_PATIENCE = PATIENCE + 5

# Directories for outputs
CHECKPOINT_DIR = 'checkpoints_part3'
LOG_DIR = "tb_logs_part3"
CHECKPOINT_DIR_AUG = 'checkpoints_part4_aug'
LOG_DIR_AUG = "tb_logs_part4_aug"

# --- Initial Setup ---
# Set seed for reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR_AUG, exist_ok=True)
os.makedirs(LOG_DIR_AUG, exist_ok=True)


# %%
# --- Download and Extract Data ---
if not os.path.exists('breast-cancer-cells-segmentation.zip'):
    print("Downloading dataset...")
    # Ensure curl is available or use Python's requests/urllib
    os.system("curl -JLO 'https://www.dropbox.com/scl/fi/gs3kzp6b8k6faf667m5tt/breast-cancer-cells-segmentation.zip?rlkey=md3mzikpwrvnaluxnhms7r4zn'")
    print("Unzipping dataset...")
    os.system("unzip -q -o breast-cancer-cells-segmentation.zip -d data") # Use -q for quiet
    print("Dataset downloaded and extracted.")
elif not os.path.exists(IMAGE_DIR) or not os.path.exists(MASK_DIR):
    print("Dataset zip found, but data directories missing. Unzipping...")
    os.system("unzip -q -o breast-cancer-cells-segmentation.zip -d data")
    print("Dataset extracted.")
else:
    print("Dataset already downloaded and extracted.")


# %%
# --- Dataset Class Definition ---
class CancerCellDataset(Dataset):
    def __init__(self, image_dir, mask_dir, sample_list, transform=None, foreground_threshold=127):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.samples = sample_list
        self.transform = transform
        self.foreground_threshold = foreground_threshold

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file, mask_file = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        try:
            image = cv2.imread(img_path)
            if image is None: raise IOError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: raise IOError(f"Failed to load mask: {mask_path}")

            mask = (mask > self.foreground_threshold).astype(np.float32)

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
                if len(mask.shape) == 2:
                     mask = mask.unsqueeze(0)
                elif len(mask.shape) == 3 and mask.shape[0] != 1:
                     # Basic attempt to fix unexpected mask shape from transform
                     mask = mask.permute(2, 0, 1)
                     if mask.shape[0] != 1:
                          mask = mask[0, :, :].unsqueeze(0) # Take first channel

            # Sanity checks after transform
            if not image.shape[1:] == mask.shape[1:]:
                 raise ValueError(f"Shape mismatch after transform: Image {image.shape} vs Mask {mask.shape} for {img_file}")
            if not mask.shape[0] == 1:
                 raise ValueError(f"Mask should have 1 channel, got {mask.shape} for {mask_file}")

            return image, mask

        except Exception as e:
            print(f"Error processing sample index {idx}, image file: {img_file}, mask file: {mask_file}")
            print(f"Error details: {e}")
            raise e


# %%
# --- Data Splitting ---
all_image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.tif') and not f.endswith('.xml')])
print(f"Found {len(all_image_files)} potential image files.")

paired_samples_with_type = []
missing_masks_count = 0
unknown_type_count = 0

for img_file in all_image_files:
    base_name = img_file.split('_ccd.tif')[0]
    mask_file = f"{base_name}.TIF"
    mask_path = os.path.join(MASK_DIR, mask_file)

    parts = base_name.split('_')
    tumor_type = "unknown"
    possible_types = ['benign', 'malignant']
    if parts and parts[0].lower() in possible_types:
        tumor_type = parts[0].lower()
    elif parts and parts[-1].lower() in possible_types:
        tumor_type = parts[-1].lower()
    else:
        unknown_type_count += 1

    if os.path.exists(mask_path):
        paired_samples_with_type.append({'image': img_file, 'mask': mask_file, 'type': tumor_type})
    else:
        missing_masks_count += 1

print(f"Successfully paired {len(paired_samples_with_type)} images with masks.")
if missing_masks_count > 0:
    print(f"Warning: {missing_masks_count} images were skipped due to missing masks.")
if unknown_type_count > 0:
     print(f"Warning: Could not determine tumor type for {unknown_type_count} samples.")

paired_samples = [(item['image'], item['mask']) for item in paired_samples_with_type]

n_total = len(paired_samples)
n_test = int(n_total * TEST_SPLIT_SIZE)
n_val = int(n_total * VAL_SPLIT_SIZE)
n_train = n_total - n_test - n_val

print(f"\nTotal samples: {n_total}")
print(f"Target split: Train={n_train}, Validation={n_val}, Test={n_test}")

train_val_samples, test_samples = train_test_split(
    paired_samples,
    test_size=n_test,
    random_state=RANDOM_STATE
)

val_size_relative = n_val / (n_train + n_val) if (n_train + n_val) > 0 else 0
train_samples, val_samples = train_test_split(
    train_val_samples,
    test_size=val_size_relative,
    random_state=RANDOM_STATE
)

print(f"Actual split: Train={len(train_samples)}, Validation={len(val_samples)}, Test={len(test_samples)}")
assert len(train_samples) + len(val_samples) + len(test_samples) == n_total, "Split sizes don't add up!"

# Helper to get types for EDA
def get_types_for_samples(sample_list, all_data_with_type):
    lookup = { (item['image'], item['mask']): item['type'] for item in all_data_with_type }
    return [lookup[sample] for sample in sample_list if sample in lookup]

train_types = get_types_for_samples(train_samples, paired_samples_with_type)
val_types = get_types_for_samples(val_samples, paired_samples_with_type)
test_types = get_types_for_samples(test_samples, paired_samples_with_type)


# %%
# --- Transformations ---
# Basic transform for validation/testing (applied later)
val_test_transform = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH, interpolation=cv2.INTER_NEAREST),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(transpose_mask=True),
])

# Simple transform for initial training (Part 1-3)
train_transform_simple = val_test_transform # No augmentation initially

# Augmented transform (defined later in Part 4)
train_transform_augmented = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH, interpolation=cv2.INTER_NEAREST),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7,
                       border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2,
                       border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(transpose_mask=True),
])

# %%
# --- Create Dataset Instances (using simple transform initially) ---
train_dataset = CancerCellDataset(IMAGE_DIR, MASK_DIR, train_samples, transform=train_transform_simple, foreground_threshold=FOREGROUND_THRESHOLD)
val_dataset = CancerCellDataset(IMAGE_DIR, MASK_DIR, val_samples, transform=val_test_transform, foreground_threshold=FOREGROUND_THRESHOLD)
test_dataset = CancerCellDataset(IMAGE_DIR, MASK_DIR, test_samples, transform=val_test_transform, foreground_threshold=FOREGROUND_THRESHOLD)

print(f"\nCreated Dataset instances:")
print(f"- Train: {len(train_dataset)} samples")
print(f"- Validation: {len(val_dataset)} samples")
print(f"- Test: {len(test_dataset)} samples")

# %%
# --- Visualization Function ---
def visualize_sample(dataset, index=0, title="Sample"):
    """Displays an image and its corresponding mask from the dataset."""
    try:
        image, mask = dataset[index] # __getitem__ returns tensors

        # Convert image tensor back to numpy for display: [C, H, W] -> [H, W, C]
        image_np = image.permute(1, 2, 0).cpu().numpy()

        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)

        # Convert mask tensor back to numpy: [1, H, W] -> [H, W]
        mask_np = mask.squeeze().cpu().numpy()

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.suptitle(f"{title} (Index: {index})", fontsize=14)

        plt.subplot(1, 2, 1)
        plt.imshow(image_np)
        plt.title(f"Image\nShape: {image.shape}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mask_np, cmap='gray')
        plt.title(f"Mask\nShape: {mask.shape}, Unique vals: {np.unique(mask_np)}")
        plt.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    except Exception as e:
        print(f"Error visualizing sample index {index}: {e}")


# Visualize a sample from the training set
print("\nVisualizing a sample from the training dataset:")
if len(train_dataset) > 0:
    visualize_sample(train_dataset, index=np.random.randint(len(train_dataset)), title="Train Sample")
else:
    print("Training dataset is empty, cannot visualize sample.")


# %% [markdown]
# ## Part 1.1: Analyzing dataset
# 
# Each time you build model you first should make EDA to understand your data.
# 
# You should answer to the following questions:
# - how many classes do you have?
# - what is class balance?
# - how many cells (roughly) do you have in train data?
# 
# Advanced part: think of questions which could help you in your future models building and then answer them below.

# %%
# --- Exploratory Data Analysis (EDA) Function ---
def analyze_dataset_eda(samples, image_dir, mask_dir, split_name, sample_limit_for_pixels=100, sample_limit_for_cells=100):
    print(f"\n--- Analyzing Dataset Split: {split_name} ---")
    num_samples = len(samples)
    print(f"Total samples in this split: {num_samples}")
    if num_samples == 0: return

    # 1. Segmentation Classes
    print("\n1. Segmentation Classes:")
    print("- Task Type: Binary Semantic Segmentation")
    print("- Classes: Background (0), Cancer Cell (1)")
    print(f"- Mask Foreground Threshold: {FOREGROUND_THRESHOLD}")

    # 2. Class Balance (Image Type)
    print("\n2. Class Balance (Image Type - based on filename):")
    split_types = get_types_for_samples(samples, paired_samples_with_type)
    type_counts = Counter(split_types)
    if not type_counts:
        print("- No type information available for this split.")
    else:
        for tumor_type, count in type_counts.most_common():
            percentage = (count / num_samples) * 100
            print(f"- {tumor_type}: {count} ({percentage:.1f}%)")
        if 'benign' in type_counts and 'malignant' in type_counts and type_counts['malignant'] > 0:
             ratio = type_counts['benign'] / type_counts['malignant']
             print(f"- Benign/Malignant Ratio: {ratio:.2f}")

    # 3. Class Balance (Pixel Level)
    print("\n3. Class Balance (Pixel Level):")
    total_pixels, foreground_pixels = 0, 0
    analyzed_count_pixels = 0
    samples_to_analyze_pixels = samples[:min(num_samples, sample_limit_for_pixels)]
    print(f"Analyzing pixel balance on {len(samples_to_analyze_pixels)} mask(s)...")
    for _, mask_file in samples_to_analyze_pixels:
        mask_path = os.path.join(mask_dir, mask_file)
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is not None:
            total_pixels += mask_raw.size
            foreground_pixels += np.sum(mask_raw > FOREGROUND_THRESHOLD)
            analyzed_count_pixels += 1

    if total_pixels > 0:
        background_pixels = total_pixels - foreground_pixels
        fg_percentage = (foreground_pixels / total_pixels) * 100
        bg_percentage = 100 - fg_percentage
        print(f"Based on {analyzed_count_pixels} analyzed masks:")
        print(f"- Foreground (Cell) Pixels: {fg_percentage:.2f}%")
        print(f"- Background Pixels: {bg_percentage:.2f}%")
        ratio = background_pixels / foreground_pixels if foreground_pixels > 0 else float('inf')
        print(f"- Background/Foreground Ratio: ~{ratio:.1f} : 1")
    else:
        print("Could not analyze pixel balance.")

    # 4. Estimated Cell Count
    print("\n4. Estimated Cell Count:")
    total_cells, analyzed_count_cells = 0, 0
    min_cell_area_threshold = 20
    samples_to_analyze_cells = samples[:min(num_samples, sample_limit_for_cells)]
    print(f"Analyzing cell count on {len(samples_to_analyze_cells)} mask(s)...")
    for _, mask_file in samples_to_analyze_cells:
        mask_path = os.path.join(mask_dir, mask_file)
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is not None:
            _, mask_binary = cv2.threshold(mask_raw, FOREGROUND_THRESHOLD, 255, cv2.THRESH_BINARY)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary.astype(np.uint8), connectivity=8)
            cells_in_mask = sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_cell_area_threshold)
            total_cells += cells_in_mask
            analyzed_count_cells += 1

    if analyzed_count_cells > 0:
        avg_cells_per_image = total_cells / analyzed_count_cells
        estimated_total_cells_in_split = avg_cells_per_image * num_samples
        print(f"Based on {analyzed_count_cells} analyzed masks (min area {min_cell_area_threshold} px):")
        print(f"- Average cells per analyzed image: {avg_cells_per_image:.1f}")
        print(f"- Estimated total cells in '{split_name}' split: {estimated_total_cells_in_split:.0f}")
    else:
        print("Could not analyze cell count.")

    print("\n5. Advanced Questions & Considerations:")
    print("- Are original image sizes consistent? (Resize transform standardizes this).")
    print("- Do masks contain zero cells? (Possible).")
    print("- How variable are cell sizes/shapes? (Requires further analysis).")
    print("- Intensity distribution?")
    print("--- End of Analysis ---")

# %%
# --- Run EDA on Splits ---
analyze_dataset_eda(train_samples, IMAGE_DIR, MASK_DIR, "Train")
analyze_dataset_eda(val_samples, IMAGE_DIR, MASK_DIR, "Validation")
analyze_dataset_eda(test_samples, IMAGE_DIR, MASK_DIR, "Test")
# %% [markdown]
# ## Part 2: Unet model
# 
# Implement class of Unet model according with [the original paper](https://arxiv.org/pdf/1505.04597).
# Ajust size of the network according with your input data.

# %%
# --- U-Net Building Blocks ---

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2) # Note: in_channels here is C_skip + C_up
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels) # Note: in_channels here is C_skip + C_up(from ConvT)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to the size of x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# %%
# --- U-Net Model Definition ---

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        base_channels = 64

        # Encoder
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor) # Bottleneck

        # Decoder
        # Input channels for Up = channels from skip connection + channels from layer below
        # The Up block's `in_channels` parameter expects this sum.
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)

        # Output Layer
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path with skip connections
        x = self.up1(x5, x4) # Argument order: (from_below, from_skip)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output convolution
        logits = self.outc(x)
        return logits

# %%
# --- Model Verification ---
N_INPUT_CHANNELS = 3 # Defined in constants
N_CLASSES = 1        # Defined in constants

# Instantiate the model
unet_model_test = UNet(n_channels=N_INPUT_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(device)

# Create a dummy batch
dummy_batch_size = 2
dummy_input = torch.randn(dummy_batch_size, N_INPUT_CHANNELS, IMG_HEIGHT, IMG_WIDTH).to(device)

# Perform a forward pass
unet_model_test.eval()
with torch.no_grad():
    output = unet_model_test(dummy_input)

# Print shapes to verify
print(f"\n--- U-Net Verification ---")
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")

expected_shape = (dummy_batch_size, N_CLASSES, IMG_HEIGHT, IMG_WIDTH)
assert output.shape == expected_shape, f"Output shape mismatch! Expected {expected_shape}, got {output.shape}"

print("U-Net model structure verified successfully.")

# Clean up test model
del unet_model_test, dummy_input, output

# %% [markdown]
# ### Результат:
# Модель научилась просто выдавать черную маску так как это дает хороший результат, надо попробовать другую функцию потерь в следующей части чтобы отучить ее от этого.

# %% [markdown]
# ## Part 3: Unet training with different losses
# 
# Train model in three setups:
# - Crossentropy loss
# - Dice loss
# - Composition of CE and Dice
# 
# Advanced:\
# For training procedure use one of frameworks for models training - Lightning, (Hugging Face, Catalyst, Ignite).\
# _Hint: this will make your life easier!_
# 
# Save all three trained models to disk!
# 
# Use validation set to evaluate models.

# %%
# --- Loss Function Definitions ---

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        preds = torch.sigmoid(logits)
        preds_flat = preds.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)
        intersection = (preds_flat * targets_flat).sum()
        pred_sum = preds_flat.sum()
        target_sum = targets_flat.sum()
        dice_score = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        return 1.0 - dice_score

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, dice_smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        # print(f"CombinedLoss initialized with alpha (BCE weight) = {alpha}") # Keep if useful

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        combined = self.alpha * bce + (1 - self.alpha) * dice
        return combined

# %%
# --- PyTorch Lightning Module ---

class CancerSegmentationModule(pl.LightningModule):
    def __init__(self, model_arch, loss_fn, learning_rate=LEARNING_RATE):
        super().__init__()
        self.model = model_arch
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

        # Metrics for validation
        self.val_dice = DiceScore(num_classes=1) # Assumes binary output, logits input
        self.val_iou = BinaryJaccardIndex()      # Expects probabilities/logits input

        # Metrics for testing
        self.test_dice = DiceScore(num_classes=1)
        self.test_iou = BinaryJaccardIndex()

        self.save_hyperparameters(ignore=['model_arch', 'loss_fn'])

    def forward(self, x):
        return self.model(x)

    def _calculate_metrics(self, logits, targets, metric_dice, metric_iou):
        preds = torch.sigmoid(logits)
        targets_int = targets.int()
        try:
            metric_dice.update(preds, targets_int)
            metric_iou.update(preds, targets_int)
        except Exception as e:
             print(f"Error updating metrics: {e}. Preds shape: {preds.shape}, Targets shape: {targets_int.shape}")

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_float = y.float()
        try:
            loss = self.loss_fn(logits, y_float)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        except Exception as e:
             print(f"Error calculating training loss: {e}")
             loss = torch.tensor(float('nan'), device=self.device)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_float = y.float()
        try:
            loss = self.loss_fn(logits, y_float)
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self._calculate_metrics(logits, y_float, self.val_dice, self.val_iou)
        except Exception as e:
             print(f"Error in validation step: {e}")
             loss = torch.tensor(float('nan'), device=self.device)

    def on_validation_epoch_end(self):
        try:
            dice_epoch = self.val_dice.compute()
            iou_epoch = self.val_iou.compute()
            self.log('val_dice', dice_epoch, prog_bar=True, logger=True)
            self.log('val_iou', iou_epoch, prog_bar=True, logger=True)
        except Exception as e:
             print(f"Error computing epoch validation metrics: {e}")
             # Log defaults to prevent crashing callbacks/loggers expecting these keys
             self.log('val_dice', 0.0, prog_bar=True, logger=True)
             self.log('val_iou', 0.0, prog_bar=True, logger=True)
        finally:
            self.val_dice.reset()
            self.val_iou.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_float = y.float()
        try:
            loss = self.loss_fn(logits, y_float)
            self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self._calculate_metrics(logits, y_float, self.test_dice, self.test_iou)
        except Exception as e:
             print(f"Error in test step: {e}")
             loss = torch.tensor(float('nan'), device=self.device)

    def on_test_epoch_end(self):
        try:
            dice_epoch = self.test_dice.compute()
            iou_epoch = self.test_iou.compute()
            self.log('test_dice', dice_epoch, prog_bar=True, logger=True)
            self.log('test_iou', iou_epoch, prog_bar=True, logger=True)
        except Exception as e:
            print(f"Error computing epoch test metrics: {e}")
            self.log('test_dice', 0.0, prog_bar=True, logger=True)
            self.log('test_iou', 0.0, prog_bar=True, logger=True)
        finally:
            self.test_dice.reset()
            self.test_iou.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # Example scheduler (optional):
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_dice"}
        return optimizer

# %%
# --- Data Loaders Setup ---
# Datasets (train_dataset, val_dataset, test_dataset) are assumed to be defined from Part 1

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0
)

print(f"DataLoaders created:")
print(f"- Train: {len(train_loader)} batches")
print(f"- Validation: {len(val_loader)} batches")
print(f"- Test: {len(test_loader)} batches")

# %%
# --- Training Loop for Different Losses ---

loss_functions = {
    'BCE': nn.BCEWithLogitsLoss(),
    'Dice': DiceLoss(),
    'Combined': CombinedLoss(alpha=0.6) # Example alpha, could be tuned
}

trained_models_paths = {}
validation_results = {}

# Model parameters N_INPUT_CHANNELS, N_CLASSES assumed to be defined

for loss_name, loss_fn_instance in loss_functions.items():
    print(f"\n--- Training U-Net with Loss: {loss_name} ---")

    # 1. Instantiate Model (UNet definition assumed available)
    unet_instance = UNet(n_channels=N_INPUT_CHANNELS, n_classes=N_CLASSES, bilinear=True)

    # 2. Instantiate Lightning Module
    lightning_model = CancerSegmentationModule(
        model_arch=unet_instance,
        loss_fn=loss_fn_instance,
        learning_rate=LEARNING_RATE
    )

    # 3. Configure Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_dice',
        dirpath=CHECKPOINT_DIR,
        filename=f'unet_{loss_name}_best_dice={{val_dice:.4f}}',
        save_top_k=1,
        mode='max',
        save_last=True
    )
    early_stop_callback = EarlyStopping(
        monitor='val_dice',
        patience=PATIENCE,
        verbose=False, # Quieter output
        mode='max'
    )

    # 4. Logger
    tensorboard_logger = TensorBoardLogger(
        save_dir=LOG_DIR,
        name=f"unet_{loss_name}"
    )

    # 5. Trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=tensorboard_logger,
        log_every_n_steps=10,
        deterministic=False,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        enable_progress_bar=False # Set False for cleaner logs if running non-interactively
    )

    # 6. Train
    print(f"Starting training for {loss_name}...")
    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Training finished for {loss_name}.")

    # 7. Store best model path and get validation score
    best_path = checkpoint_callback.best_model_path
    model_saved_path = None
    if best_path and os.path.exists(best_path):
        print(f"Best model checkpoint: {best_path}")
        model_saved_path = best_path
    else:
        last_path = checkpoint_callback.last_model_path
        if last_path and os.path.exists(last_path):
             print(f"Using last checkpoint: {last_path}")
             model_saved_path = last_path
        else:
             print(f"Warning: No checkpoint found for {loss_name}.")

    trained_models_paths[loss_name] = model_saved_path

    if model_saved_path:
        # Validate the saved model to get final validation metrics
        print(f"Validating model from {model_saved_path}...")
        # Load the model structure again for loading the checkpoint
        unet_load_instance = UNet(n_channels=N_INPUT_CHANNELS, n_classes=N_CLASSES, bilinear=True)
        try:
             best_model_loaded = CancerSegmentationModule.load_from_checkpoint(
                 checkpoint_path=model_saved_path,
                 model_arch=unet_load_instance,
                 loss_fn=loss_fn_instance # Need loss for potential validation step loss calculation
             )
             val_result = trainer.validate(best_model_loaded, dataloaders=val_loader, verbose=False)
             if val_result:
                 validation_results[loss_name] = val_result[0] # Store the metrics dict
                 print(f"Validation results for {loss_name}: {val_result[0]}")
             else:
                 validation_results[loss_name] = {'val_loss': float('inf'), 'val_dice': 0.0, 'val_iou': 0.0}
             del best_model_loaded
        except Exception as e:
            print(f"Error loading or validating checkpoint {model_saved_path}: {e}")
            validation_results[loss_name] = {'val_loss': float('inf'), 'val_dice': 0.0, 'val_iou': 0.0}
        del unet_load_instance
    else:
         validation_results[loss_name] = {'val_loss': float('inf'), 'val_dice': 0.0, 'val_iou': 0.0}

    # 8. Clean up
    del unet_instance, lightning_model, trainer, checkpoint_callback, early_stop_callback, tensorboard_logger
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %%
# --- Summarize Validation Results ---

print("\n--- Training Summary ---")
print("Paths to best/last saved models:")
for name, path in trained_models_paths.items():
    print(f"- {name}: {path if path else 'Not saved/found'}")

print("\nValidation Results:")
# Check if validation_results dict is not empty
if validation_results:
    results_df = pd.DataFrame(validation_results).T
    # Standardize column names (if they come from logger)
    if 'val_loss' in results_df.columns:
         results_df = results_df.rename(columns={
             'val_loss': 'Validation Loss',
             'val_dice': 'Validation Dice',
             'val_iou': 'Validation IoU'
         })
    # Format and print
    print(results_df.to_string(float_format="%.4f"))

    # Identify best based on Dice score
    try:
        best_loss_name = results_df['Validation Dice'].idxmax()
        print(f"\nBest loss function based on Validation Dice: '{best_loss_name}'")
    except Exception as e:
        print(f"\nCould not determine best loss function: {e}")
        best_loss_name = None # Signal that determination failed
else:
    print("No validation results were collected.")
    best_loss_name = None # Ensure variable exists but is None

# %% [markdown]
# ## Part 3.1: Losses conclusion
# 
# Analyse results of the three models above using metrics, losses and visualizations you know (all three parts are required).
# 
# Make motivated conclusion on which setup is better. Provide your arguments.
# 
# Calculate loss and metrics of the best model on test set.

# %%
# --- Evaluation and Plotting Utilities ---

def evaluate_model_on_test(model_module, dataloader, device):
    """Evaluates a loaded LightningModule on a dataloader using its test metrics."""
    model_module.eval()
    model_module.to(device)

    # Get the trainer's test metrics logic (runs test_step and test_epoch_end)
    # We can simulate this by creating a temporary trainer
    temp_trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=False, # No need to log during this evaluation
        enable_progress_bar=False,
        enable_checkpointing=False
        )

    test_results = temp_trainer.test(model_module, dataloaders=dataloader, verbose=False)

    # Extract metrics logged during test_epoch_end
    # The structure of test_results is a list of dictionaries
    if test_results:
        # Metrics are usually in the first dictionary
        final_metrics = test_results[0]
        # Standardize keys slightly
        return {
            'test_loss': final_metrics.get('test_loss', float('nan')),
            'test_dice': final_metrics.get('test_dice', float('nan')),
            'test_iou': final_metrics.get('test_iou', float('nan'))
        }
    else:
        print("Warning: trainer.test() returned empty results.")
        return {'test_loss': float('nan'), 'test_dice': float('nan'), 'test_iou': float('nan')}


def load_model_from_checkpoint(model_class, checkpoint_path, loss_fn_instance, n_channels, n_classes, **kwargs):
    """Loads a CancerSegmentationModule model from a checkpoint."""
    try:
        # Instantiate the base architecture (e.g., UNet, PSPNet)
        arch_instance = model_class(n_channels=n_channels, n_classes=n_classes, **kwargs)

        # Load the LightningModule
        model = CancerSegmentationModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_arch=arch_instance,
            loss_fn=loss_fn_instance # Pass the loss function instance
        )
        print(f"Model loaded successfully from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {checkpoint_path}: {e}")
        return None


def plot_training_curves(log_dir_base, configs_to_plot, metrics_to_plot=['val_loss', 'val_dice', 'val_iou', 'train_loss_epoch']):
    """Plots training curves from TensorBoard logs for given configurations."""
    print(f"\n--- Plotting Training Curves from {log_dir_base} ---")
    if not os.path.exists(log_dir_base):
        print(f"Error: Log directory '{log_dir_base}' not found.")
        return

    num_metrics = len(metrics_to_plot)
    plt.figure(figsize=(18, 5 * num_metrics))
    plot_num = 1

    for metric in metrics_to_plot:
        ax = plt.subplot(num_metrics, 1, plot_num)
        plt.title(f"Training History: {metric.replace('_', ' ').title()}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.split('_')[-1].capitalize())
        plt.grid(True)
        has_data_for_metric = False

        for config_name, log_pattern in configs_to_plot.items():
            full_log_dir = None
            config_log_path = os.path.join(log_dir_base, log_pattern)
            if os.path.exists(config_log_path):
                version_dirs = sorted([d for d in os.listdir(config_log_path) if d.startswith('version_')])
                if version_dirs:
                    full_log_dir = os.path.join(config_log_path, version_dirs[-1]) # Latest version

            if full_log_dir and os.path.exists(full_log_dir):
                # print(f"Reading '{metric}' logs for '{config_name}' from: {full_log_dir}") # Optional verbose
                try:
                    event_acc = EventAccumulator(full_log_dir)
                    event_acc.Reload()
                    if metric in event_acc.Tags()['scalars']:
                        events = event_acc.Scalars(metric)
                        steps = [e.step for e in events]
                        values = [e.value for e in events]
                        if steps and values:
                            plt.plot(steps, values, marker='.', linestyle='-', label=config_name)
                            has_data_for_metric = True
                except Exception as e:
                    print(f"  - Error reading logs for {config_name}: {e}")

        if has_data_for_metric:
            plt.legend()
            plot_num += 1
        else:
            # If no data plotted, add text and still increment plot index
            plt.text(0.5, 0.5, f'No data found for {metric}', ha='center', va='center', transform=ax.transAxes)
            plot_num += 1


    plt.tight_layout()
    plt.show()


def plot_predictions(models_dict, dataset, indices, device, num_samples=3, threshold=0.5):
    """Plots input, GT, and predictions from multiple models (keyed by name)."""
    if not models_dict:
        print("Cannot plot predictions: No models provided.")
        return

    num_models = len(models_dict)
    num_cols = 2 + num_models # Input + GT + N models
    plt.figure(figsize=(5 * num_cols, 5 * num_samples))
    plt.suptitle(f"Prediction Comparison on Test Set", fontsize=16, y=1.02)

    plot_row_idx = 0
    for i, data_idx in enumerate(indices):
        try:
            image, mask_gt = dataset[data_idx]

            # --- Prepare data for plotting ---
            image_display = image.permute(1, 2, 0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_display = std * image_display + mean
            image_display = np.clip(image_display, 0, 1)
            mask_gt_display = mask_gt.squeeze().cpu().numpy()

            # --- Plot Input and Ground Truth ---
            base_plot_idx = plot_row_idx * num_cols + 1
            plt.subplot(num_samples, num_cols, base_plot_idx)
            plt.imshow(image_display)
            plt.title(f"Input (Idx: {data_idx})")
            plt.axis('off')

            plt.subplot(num_samples, num_cols, base_plot_idx + 1)
            plt.imshow(mask_gt_display, cmap='gray')
            plt.title("Ground Truth")
            plt.axis('off')

            # --- Plot Predictions for each model ---
            model_plot_offset = 0
            for model_name, model in models_dict.items():
                model.eval()
                model.to(device)
                image_input = image.unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(image_input)
                    preds_prob = torch.sigmoid(logits)
                    preds_binary = (preds_prob > threshold).squeeze().cpu().numpy()

                plt.subplot(num_samples, num_cols, base_plot_idx + 2 + model_plot_offset)
                plt.imshow(preds_binary, cmap='gray')
                plt.title(f"Pred ({model_name})")
                plt.axis('off')
                model_plot_offset += 1

            plot_row_idx += 1

        except Exception as e:
            print(f"Error visualizing sample index {data_idx}: {e}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()


# %%
# --- Analyze Results from Part 3 ---

print("\n--- Part 3.1: Losses Conclusion ---")

# Validation results summary (already printed at the end of training loop)
# 'best_loss_name' is determined there too.

# --- Load Best Model ---
best_model_part3 = None
test_results_best_model = None

if best_loss_name and best_loss_name in trained_models_paths:
    best_model_path = trained_models_paths.get(best_loss_name)
    if best_model_path:
        print(f"\nLoading best model ({best_loss_name}) from Part 3: {best_model_path}")
        best_model_part3 = load_model_from_checkpoint(
            model_class=UNet, # We know it's UNet in Part 3
            checkpoint_path=best_model_path,
            loss_fn_instance=loss_functions[best_loss_name],
            n_channels=N_INPUT_CHANNELS,
            n_classes=N_CLASSES,
            bilinear=True # Make sure params match original instantiation
        )
    else:
        print(f"Path for best loss '{best_loss_name}' not found or model wasn't saved.")
else:
    print("Could not identify the best loss function from validation results.")

# --- Evaluate Best Model on Test Set ---
if best_model_part3:
    print(f"\nEvaluating best model ({best_loss_name}) on the test set...")
    test_results_best_model = evaluate_model_on_test(best_model_part3, test_loader, device)
    print("Test Set Results (Best Loss from Part 3):")
    print(pd.Series(test_results_best_model).to_string(float_format="%.4f"))
else:
    print("\nSkipping test set evaluation (best model not loaded).")

# --- Load All Models from Part 3 for Visualization ---
loaded_models_part3 = {}
if trained_models_paths: # Check if the dict has paths
     for loss_name, model_path in trained_models_paths.items():
         if model_path: # Check if path exists for this loss
             model = load_model_from_checkpoint(
                 model_class=UNet,
                 checkpoint_path=model_path,
                 loss_fn_instance=loss_functions[loss_name],
                 n_channels=N_INPUT_CHANNELS,
                 n_classes=N_CLASSES,
                 bilinear=True
             )
             if model:
                 loaded_models_part3[loss_name] = model
         else:
             print(f"Skipping loading for {loss_name} as path is missing.")
else:
     print("Warning: trained_models_paths dictionary is empty or not defined.")


# --- Visualize Predictions (All Losses) ---
if loaded_models_part3 and len(test_dataset) > 0:
    num_viz_samples = min(3, len(test_dataset))
    plot_indices = np.random.choice(len(test_dataset), num_viz_samples, replace=False).tolist()
    print(f"\nVisualizing prediction comparison for Part 3 models on indices: {plot_indices}")
    plot_predictions(loaded_models_part3, test_dataset, plot_indices, device, num_samples=num_viz_samples)
else:
    print("\nSkipping Part 3 prediction visualization (no models loaded or test_dataset empty).")


# --- Plot Training Curves ---
configs_to_plot_part3 = {loss_name: f"unet_{loss_name}" for loss_name in loss_functions.keys()}
plot_training_curves(LOG_DIR, configs_to_plot_part3)


# --- Conclusion Text ---
print("\n--- Analysis and Conclusion ---")
if best_loss_name:
     best_val_dice = results_df.loc[best_loss_name, 'Validation Dice'] if 'results_df' in locals() else 'N/A'
     print(f"Based on validation results, '{best_loss_name}' performed best (Val Dice: {best_val_dice:.4f}).")
else:
     print("Could not determine the best loss function from validation.")

print("\nArguments:")
print("- BCE Loss: Pixel-independent, can struggle with imbalance.")
print("- Dice Loss: Optimizes region overlap (Dice score), better for imbalance.")
print("- Combined Loss: Balances BCE and Dice.")
print("\nObservations:")
print("- Graphs typically show Dice/Combined achieving better validation Dice/IoU.")
print("- Visualizations show qualitative differences between loss functions.")
if test_results_best_model:
     print(f"- Best model ('{best_loss_name}') achieved Test Dice: {test_results_best_model['test_dice']:.4f}, Test IoU: {test_results_best_model['test_iou']:.4f}.")
else:
     print(f"- Test evaluation for the best model ('{best_loss_name}') was not performed or failed.")

print(f"\nFinal Verdict (Part 3): The '{best_loss_name if best_loss_name else 'undetermined'}' loss seems most effective among the three for this U-Net setup.")

# --- Clean up ---
del loaded_models_part3, best_model_part3 # Keep test_results_best_model for later comparison
if 'unet_load_instance' in locals(): del unet_load_instance
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# %% [markdown]
# ## Результаты
# Ну с новыми функциями потерь дело явно бодрее идет, но модель если тренить много эпох из-за маленького разнобрая (40 семплов) капец перееобучается и трейн начинает пытаться учить
# НО лучше всего себя Dice показывает

# %% [markdown]
# ## Part 4: Augmentations and advanced model
# 
# Choose set of augmentations relevant for this case (at least 5 of them) using [Albumentations library](https://albumentations.ai/).
# Apply them to dataset (of course dynamicaly during reading from disk).
# 
# One more thing to improve is model: use [PSPnet](https://arxiv.org/pdf/1612.01105v2) (either use library [implementation](https://smp.readthedocs.io/en/latest/models.html#pspnet) or implement yourself) as improved version of Unet.
# 
# Alternatively you may use model of your choice (it should be more advanced than Unet ofc).
# 
# Train Unet and second model on augmented data.

# %%
# --- Create Augmented Datasets and DataLoaders ---

# train_transform_augmented and val_test_transform are defined in the first code cell

train_dataset_augmented = CancerCellDataset(
    IMAGE_DIR, MASK_DIR, train_samples,
    transform=train_transform_augmented,
    foreground_threshold=FOREGROUND_THRESHOLD
)

# train_loader, val_loader, test_loader are defined in Part 3

train_loader_augmented = DataLoader(
    train_dataset_augmented,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0,
    drop_last=True
)

print(f"\nAugmented Training Dataset: {len(train_dataset_augmented)} samples")
print(f"Augmented Training DataLoader: {len(train_loader_augmented)} batches")

# Visualize a sample from the *augmented* dataset
print("\nVisualizing a sample from the *augmented* training dataset:")
if len(train_dataset_augmented) > 0:
     visualize_sample(train_dataset_augmented, index=np.random.randint(len(train_dataset_augmented)), title="Augmented Train Sample")
else:
     print("Augmented training dataset is empty.")

# %%
# --- Prepare Models for Augmented Training ---

# best_loss_name and loss_functions are assumed available from Part 3
if 'best_loss_name' not in locals() or best_loss_name is None:
    print("Warning: 'best_loss_name' not found from Part 3. Defaulting to 'Dice'.")
    best_loss_name = 'Dice'
    if best_loss_name not in loss_functions:
         best_loss_name = list(loss_functions.keys())[0] # Fallback further

best_loss_fn = loss_functions[best_loss_name]
print(f"\nUsing Loss Function: '{best_loss_name}' for augmented training.")

models_to_train_aug = {}

# 1. U-Net (new instance)
models_to_train_aug['UNet_Aug'] = UNet(n_channels=N_INPUT_CHANNELS, n_classes=N_CLASSES, bilinear=True)
print("Added U-Net (new instance) for augmented training.")

# 2. PSPNet (if library available)
if SMP_AVAILABLE:
    try:
        # Define PSPNet parameters
        PSPNet_ENCODER = "resnet34"
        PSPNet_WEIGHTS = "imagenet"

        pspnet_model = smp.PSPNet(
            encoder_name=PSPNet_ENCODER,
            encoder_weights=PSPNet_WEIGHTS,
            in_channels=N_INPUT_CHANNELS,
            classes=N_CLASSES,
            # activation=None # Keep logits output
        )
        models_to_train_aug['PSPNet_Aug'] = pspnet_model
        print(f"Added PSPNet ({PSPNet_ENCODER}, {PSPNet_WEIGHTS} weights) for augmented training.")
    except Exception as e:
        print(f"Error creating PSPNet model: {e}. PSPNet will not be trained.")
else:
    print("Skipping PSPNet training (segmentation-models-pytorch not available).")

# %%
# --- Training Loop for Augmented Data ---

trained_models_aug_paths = {}
validation_results_aug = {}

for model_name, model_arch_instance in models_to_train_aug.items():
    print(f"\n--- Training {model_name} with Augmentations & Loss: {best_loss_name} ---")

    # 1. Lightning Module
    lightning_model_aug = CancerSegmentationModule(
        model_arch=model_arch_instance,
        loss_fn=best_loss_fn,
        learning_rate=LEARNING_RATE
    )

    # 2. Callbacks
    checkpoint_callback_aug = ModelCheckpoint(
        monitor='val_dice',
        dirpath=CHECKPOINT_DIR_AUG,
        filename=f'{model_name}_best_dice={{val_dice:.4f}}',
        save_top_k=1,
        mode='max',
        save_last=True
    )
    early_stop_callback_aug = EarlyStopping(
        monitor='val_dice',
        patience=AUG_PATIENCE,
        verbose=False,
        mode='max'
    )

    # 3. Logger
    tensorboard_logger_aug = TensorBoardLogger(
        save_dir=LOG_DIR_AUG,
        name=f"{model_name}_{best_loss_name}_aug"
    )

    # 4. Trainer
    trainer_aug = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=AUG_MAX_EPOCHS,
        callbacks=[checkpoint_callback_aug, early_stop_callback_aug],
        logger=tensorboard_logger_aug,
        log_every_n_steps=10,
        deterministic=False,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        enable_progress_bar=False
    )

    # 5. Train
    print(f"Starting training for {model_name}...")
    trainer_aug.fit(lightning_model_aug, train_dataloaders=train_loader_augmented, val_dataloaders=val_loader)
    print(f"Training finished for {model_name}.")

    # 6. Store Path and Validate
    best_path_aug = checkpoint_callback_aug.best_model_path
    model_saved_path = None
    if best_path_aug and os.path.exists(best_path_aug):
        model_saved_path = best_path_aug
    else:
        last_path_aug = checkpoint_callback_aug.last_model_path
        if last_path_aug and os.path.exists(last_path_aug):
             print(f"Using last checkpoint for {model_name}: {last_path_aug}")
             model_saved_path = last_path_aug
        else:
             print(f"Warning: No checkpoint found for {model_name}.")

    trained_models_aug_paths[model_name] = model_saved_path

    if model_saved_path:
        print(f"Validating {model_name} model from: {model_saved_path}")
        # Determine model class for loading
        model_cls_to_load = None
        load_kwargs = {}
        if 'UNet' in model_name:
             model_cls_to_load = UNet
             load_kwargs = {'bilinear': True} # Example specific kwarg for UNet
        elif 'PSPNet' in model_name and SMP_AVAILABLE:
             model_cls_to_load = smp.PSPNet
             # Kwargs needed to reconstruct the arch, weights loaded from ckpt
             load_kwargs = {'encoder_name': PSPNet_ENCODER, 'encoder_weights': None}

        if model_cls_to_load:
            loaded_model_aug = load_model_from_checkpoint(
                model_class=model_cls_to_load,
                checkpoint_path=model_saved_path,
                loss_fn_instance=best_loss_fn,
                n_channels=N_INPUT_CHANNELS,
                n_classes=N_CLASSES,
                **load_kwargs
            )
            if loaded_model_aug:
                 val_result_aug = trainer_aug.validate(loaded_model_aug, dataloaders=val_loader, verbose=False)
                 if val_result_aug:
                     validation_results_aug[model_name] = val_result_aug[0]
                     print(f"Validation results for {model_name}: {val_result_aug[0]}")
                 else:
                     validation_results_aug[model_name] = {'val_loss': float('inf'), 'val_dice': 0.0, 'val_iou': 0.0}
                 del loaded_model_aug
            else:
                 validation_results_aug[model_name] = {'val_loss': float('inf'), 'val_dice': 0.0, 'val_iou': 0.0}
        else:
             print(f"Could not determine model class for {model_name} to load checkpoint.")
             validation_results_aug[model_name] = {'val_loss': float('inf'), 'val_dice': 0.0, 'val_iou': 0.0}
    else:
         validation_results_aug[model_name] = {'val_loss': float('inf'), 'val_dice': 0.0, 'val_iou': 0.0}

    # 7. Clean up
    del model_arch_instance, lightning_model_aug, trainer_aug
    del checkpoint_callback_aug, early_stop_callback_aug, tensorboard_logger_aug
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %%
# --- Summarize Augmented Training Results ---

print("\n--- Augmented Training Summary ---")
print("Paths to best/last saved models (Augmented Training):")
for name, path in trained_models_aug_paths.items():
    print(f"- {name}: {path if path else 'Not saved/found'}")

print("\nValidation Results (Augmented Training):")
if validation_results_aug:
    results_df_aug = pd.DataFrame(validation_results_aug).T
    if 'val_dice' in results_df_aug.columns:
        results_df_aug = results_df_aug.rename(columns={
            'val_loss': 'Validation Loss', 'val_dice': 'Validation Dice', 'val_iou': 'Validation IoU'
        })
    print(results_df_aug.to_string(float_format="%.4f"))
else:
    print("No validation results were collected for augmented training.")


# %% [markdown]
# ## Part 4.2: Augmentations and advanced model conclusion
# 
# Compare three setups:
# - Unet without augmentations (with best loss)
# - Unet with augmentations
# - Advanced model with augmentations
# 
# _Hint: with augs and more complex model you may want to have more iterations._
# 
# Save all three trained models to disk!
# 
# Once again provide comprehensive arguments and your insights.
# 
# Wich setup is better?
# 
# Compute losses and metrics on test set. Measure improvement over first test evaluation.

# %%
# --- Final Comparison Setup ---
print("\n--- Part 4.2: Final Comparison and Conclusion ---")

paths_to_compare = {}
final_model_names = {} # Map descriptive name to internal key if needed

# 1. Baseline Model (Best from Part 3)
baseline_key = f"U-Net (No Aug, {best_loss_name})"
if best_loss_name in trained_models_paths and trained_models_paths[best_loss_name]:
    paths_to_compare[baseline_key] = trained_models_paths[best_loss_name]
else:
    print(f"Warning: Baseline model '{baseline_key}' path not found.")

# 2. Augmented U-Net (from Part 4)
unet_aug_key = 'UNet_Aug'
unet_aug_name = f"U-Net (Aug, {best_loss_name})"
if unet_aug_key in trained_models_aug_paths and trained_models_aug_paths[unet_aug_key]:
    paths_to_compare[unet_aug_name] = trained_models_aug_paths[unet_aug_key]
else:
    print(f"Warning: Augmented U-Net model '{unet_aug_name}' path not found.")

# 3. Augmented Advanced Model (PSPNet from Part 4)
pspnet_aug_key = 'PSPNet_Aug'
pspnet_aug_name = f"PSPNet (Aug, {best_loss_name})"
if pspnet_aug_key in trained_models_aug_paths and trained_models_aug_paths[pspnet_aug_key]:
    paths_to_compare[pspnet_aug_name] = trained_models_aug_paths[pspnet_aug_key]
elif SMP_AVAILABLE: # Only warn if it should have been trained
    print(f"Warning: Augmented PSPNet model '{pspnet_aug_name}' path not found.")

if not paths_to_compare:
    raise RuntimeError("No models found for final comparison. Ensure Parts 3 and 4 ran successfully.")

print("\nModels selected for final comparison:")
for name in paths_to_compare.keys(): print(f"- {name}")

# %%
# --- Evaluate Selected Models on Test Set ---
final_test_results = {}
loaded_models_final = {}

print("\n--- Evaluating final models on Test Set ---")
for model_name, checkpoint_path in paths_to_compare.items():
    print(f"\nEvaluating: {model_name}")

    # Determine model class and kwargs for loading
    model_cls_to_load = None
    load_kwargs = {}
    if "U-Net" in model_name:
         model_cls_to_load = UNet
         load_kwargs = {'bilinear': True}
    elif "PSPNet" in model_name and SMP_AVAILABLE:
         model_cls_to_load = smp.PSPNet
         load_kwargs = {'encoder_name': PSPNet_ENCODER, 'encoder_weights': None}

    if model_cls_to_load:
        loaded_model = load_model_from_checkpoint(
            model_class=model_cls_to_load,
            checkpoint_path=checkpoint_path,
            loss_fn_instance=best_loss_fn, # Use the consistent best loss
            n_channels=N_INPUT_CHANNELS,
            n_classes=N_CLASSES,
            **load_kwargs
        )
        if loaded_model:
            loaded_models_final[model_name] = loaded_model
            results = evaluate_model_on_test(loaded_model, test_loader, device)
            final_test_results[model_name] = results
            print(f"Results: {results}")
        else:
            final_test_results[model_name] = {'test_loss': float('nan'), 'test_dice': float('nan'), 'test_iou': float('nan')}
    else:
        print(f"Cannot determine architecture for {model_name}. Skipping evaluation.")
        final_test_results[model_name] = {'test_loss': float('nan'), 'test_dice': float('nan'), 'test_iou': float('nan')}

# %%
# --- Summarize Final Test Results and Improvement ---
print("\n--- Final Test Results Summary ---")
best_model_final_name = None
best_final_metrics = None
baseline_metrics = None

if final_test_results:
    results_df_final = pd.DataFrame(final_test_results).T.sort_values(by='test_dice', ascending=False)
    results_df_final = results_df_final.rename(columns={
        'test_loss': 'Test Loss', 'test_dice': 'Test Dice', 'test_iou': 'Test IoU'
    })
    print(results_df_final.to_string(float_format="%.4f"))

    if not results_df_final.empty:
        best_model_final_name = results_df_final.index[0]
        best_final_metrics = results_df_final.iloc[0]
        print(f"\nOverall Best Performing Setup on Test Set: '{best_model_final_name}'")

        # Find baseline results for comparison
        if baseline_key in results_df_final.index:
             baseline_metrics = results_df_final.loc[baseline_key]
        elif 'test_results_best_model' in locals() and test_results_best_model:
             print("(Comparing with baseline results loaded from Part 3.1)")
             baseline_metrics = pd.Series(test_results_best_model).rename({
                 'test_loss': 'Test Loss', 'test_dice': 'Test Dice', 'test_iou': 'Test IoU'
             }) # Use results from Part 3.1 test eval
        else:
             print("Warning: Baseline results not found for comparison.")

        if best_final_metrics is not None and baseline_metrics is not None:
             dice_improvement = best_final_metrics['Test Dice'] - baseline_metrics['Test Dice']
             iou_improvement = best_final_metrics['Test IoU'] - baseline_metrics['Test IoU']
             print("\nImprovement over Baseline (U-Net No Aug):")
             print(f"  Dice Improvement: {dice_improvement:+.4f}")
             print(f"  IoU Improvement:  {iou_improvement:+.4f}")

else:
    print("No final test results were collected.")


# %%
# --- Visualize Final Predictions Comparison ---
if loaded_models_final and len(test_dataset) > 0:
    print("\n--- Visualizing Final Model Predictions ---")
    num_viz_samples = min(3, len(test_dataset))
    plot_indices_final = np.random.choice(len(test_dataset), num_viz_samples, replace=False).tolist()
    print(f"Visualizing final comparison for indices: {plot_indices_final}")
    # Use the generic plot_predictions function
    plot_predictions(loaded_models_final, test_dataset, plot_indices_final, device, num_samples=num_viz_samples)
else:
    print("\nSkipping final prediction visualization (models not loaded or test_dataset empty).")


# %%
# --- Plot Training Curves (Part 4 Models) ---
# Check if models were trained and paths exist before attempting to plot
configs_to_plot_part4 = {}
if unet_aug_key in trained_models_aug_paths and trained_models_aug_paths[unet_aug_key]:
     configs_to_plot_part4[unet_aug_name] = f"{unet_aug_key}_{best_loss_name}_aug"
if pspnet_aug_key in trained_models_aug_paths and trained_models_aug_paths[pspnet_aug_key]:
     configs_to_plot_part4[pspnet_aug_name] = f"{pspnet_aug_key}_{best_loss_name}_aug"

if configs_to_plot_part4:
    plot_training_curves(LOG_DIR_AUG, configs_to_plot_part4)
else:
    print("\nNo Part 4 models found to plot training curves.")


# %%
# --- Conclusion Text ---
print("\n--- Итоги и Выводы (Part 4.2) ---")

print("Сравнение производительности моделей на тестовом наборе:")
if 'results_df_final' in locals():
    print(results_df_final.to_string(float_format="%.4f"))
else:
    print("Результаты тестирования отсутствуют.")

print("\nАнализ:")
print("1. Влияние Аугментаций (U-Net No Aug vs U-Net Aug):")
unet_no_aug_key = f"U-Net (No Aug, {best_loss_name})" # Reconstruct baseline key
# Check if both keys exist in the final results dataframe
if 'results_df_final' in locals() and unet_no_aug_key in results_df_final.index and unet_aug_name in results_df_final.index:
    dice_no_aug = results_df_final.loc[unet_no_aug_key, 'Test Dice']
    dice_aug = results_df_final.loc[unet_aug_name, 'Test Dice']
    print(f" - U-Net без аугментаций -> Test Dice = {dice_no_aug:.4f}.")
    print(f" - U-Net с аугментациями -> Test Dice = {dice_aug:.4f} ({dice_aug - dice_no_aug:+.4f}).")
    if dice_aug > dice_no_aug: print(" - Вывод: Аугментации улучшили результат U-Net.")
    else: print(" - Вывод: Аугментации не улучшили/ухудшили результат U-Net.")
else:
    print(" - Не удалось сравнить U-Net с/без аугментаций (результаты отсутствуют).")

print("\n2. Влияние Архитектуры (U-Net Aug vs PSPNet Aug):")
# Check if both augmented models' results exist
if 'results_df_final' in locals() and unet_aug_name in results_df_final.index and pspnet_aug_name in results_df_final.index:
    dice_unet_aug = results_df_final.loc[unet_aug_name, 'Test Dice']
    dice_pspnet_aug = results_df_final.loc[pspnet_aug_name, 'Test Dice']
    print(f" - U-Net с аугментациями -> Test Dice = {dice_unet_aug:.4f}.")
    print(f" - PSPNet с аугментациями -> Test Dice = {dice_pspnet_aug:.4f} ({dice_pspnet_aug - dice_unet_aug:+.4f}).")
    if dice_pspnet_aug > dice_unet_aug: print(" - Вывод: PSPNet показал лучший результат, чем U-Net (с аугментациями).")
    else: print(" - Вывод: PSPNet не показал лучшего результата, чем U-Net (с аугментациями).")
elif SMP_AVAILABLE: # Check if PSPNet should have been compared
    print(" - Не удалось сравнить U-Net (Aug) и PSPNet (Aug) (результаты отсутствуют).")

print("\nОбщий вывод:")
if best_model_final_name:
    print(f" - Лучшей конфигурацией по метрике Test Dice является: '{best_model_final_name}'.")
    if best_final_metrics is not None and baseline_metrics is not None:
         dice_improvement = best_final_metrics['Test Dice'] - baseline_metrics['Test Dice']
         print(f" - Это обеспечило улучшение Dice на {dice_improvement:+.4f} по сравнению с базовой моделью U-Net без аугментаций.")
    print(" - Использование аугментаций данных является важным фактором.")
    if "PSPNet" in best_model_final_name: print(" - Переход к архитектуре PSPNet также внес вклад в улучшение.")
    elif "U-Net" in best_model_final_name and "Aug" in best_model_final_name: print(" - U-Net с аугментациями показал лучший результат.")
else:
    print(" - Не удалось определить лучшую модель из-за отсутствия результатов.")

print("\nОграничения и Возможные Улучшения:")
print(" - Малый размер датасета, зависимость от разделения.")
print(" - Отсутствие оптимизации гиперпараметров.")
print(" - Возможность использования других архитектур/бэкбонов.")
print(" - Пост-обработка.")

# %%
# --- Clean up Final Models ---``
if 'loaded_models_final' in locals():
    del loaded_models_final
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()