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
import os
if not os.path.exists('breast-cancer-cells-segmentation.zip'):
  !curl -JLO 'https://www.dropbox.com/scl/fi/gs3kzp6b8k6faf667m5tt/breast-cancer-cells-segmentation.zip?rlkey=md3mzikpwrvnaluxnhms7r4zn'
  !unzip breast-cancer-cells-segmentation.zip -d data

# %%
!pip install torch torchvision torchaudio pytorch-lightning albumentations scikit-image scikit-learn opencv-python Pillow matplotlib -q
!pip install --upgrade torchmetrics

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
import torchmetrics
# from torchmetrics import Dice, JaccardIndex
from torchmetrics.classification import BinaryJaccardIndex
import pandas as pd
print(torchmetrics.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

DATA_DIR = 'data'
IMAGE_DIR = os.path.join(DATA_DIR, 'Images')
MASK_DIR = os.path.join(DATA_DIR, 'Masks')
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.3
VAL_SPLIT_SIZE = 0.5
IMG_HEIGHT = 512
IMG_WIDTH = 512
BATCH_SIZE = 8
NUM_WORKERS = os.cpu_count() // 2
LEARNING_RATE = 1e-4
MAX_EPOCHS = 100
PATIENCE = 5

# %%
class CancerCellsDataset(Dataset):
    def __init__(self, image_dir, mask_dir, sample_list, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.samples = sample_list
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file, mask_file = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
             raise FileNotFoundError(f"Не удалось загрузить изображение: {img_path}")
        if mask is None:
             raise FileNotFoundError(f"Не удалось загрузить маску: {mask_path}")

        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if len(mask.shape) == 2:
             mask = mask.unsqueeze(0)
        elif len(mask.shape) == 3 and mask.shape[0] != 1:
             mask = mask.permute(2, 0, 1)

        assert image.shape[1:] == mask.shape[1:], \
            f"Размеры изображения ({image.shape[1:]}) и маски ({mask.shape[1:]}) не совпадают для {img_file}"

        return image, mask

all_image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.tif') and not f.endswith('.xml')])
paired_samples = []
missing_masks = 0
for img_file in all_image_files:
    base_name = img_file.split('_ccd.tif')[0]
    mask_file = f"{base_name}.TIF"
    mask_path = os.path.join(MASK_DIR, mask_file)
    if os.path.exists(mask_path):
        paired_samples.append((img_file, mask_file))
    else:
        # print(f"Предупреждение: Маска не найдена для {img_file}")
        missing_masks += 1
print(f"Найдено {len(paired_samples)} пар изображение/маска. Пропущено {missing_masks} из-за отсутствия масок.")

train_val_samples, test_samples = train_test_split(
    paired_samples, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE
)
train_samples, val_samples = train_test_split(
    train_val_samples, test_size=VAL_SPLIT_SIZE, random_state=RANDOM_STATE
)
print(f"Размер выборки Train: {len(train_samples)}, Validation: {len(val_samples)}, Test: {len(test_samples)}")

train_transform = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH, interpolation=cv2.INTER_NEAREST),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(transpose_mask=True),
])

val_test_transform = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH, interpolation=cv2.INTER_NEAREST),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(transpose_mask=True),
])

train_dataset_no_aug = CancerCellsDataset(IMAGE_DIR, MASK_DIR, train_samples, transform=val_test_transform)
val_dataset = CancerCellsDataset(IMAGE_DIR, MASK_DIR, val_samples, transform=val_test_transform)
test_dataset = CancerCellsDataset(IMAGE_DIR, MASK_DIR, test_samples, transform=val_test_transform)


def visualize_sample(dataset, index=0, title="Sample"):
    image, mask = dataset[index]
    image_np = image.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)
    mask_np = mask.squeeze().cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.suptitle(f"{title} (Index: {index})", fontsize=14)
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title(f"Image\nShape: {image.shape}")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title(f"Mask\nShape: {mask.shape}, Unique: {np.unique(mask_np)}")
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

visualize_sample(train_dataset_no_aug, index=1, title="Train Sample (No Aug)")

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
def analyze_dataset_eda(dataset, split_name):
    print(f"\n--- Анализ датасета: {split_name} ---")
    num_samples = len(dataset)
    print(f"Всего образцов: {num_samples}")

    benign_count = sum(1 for img_file, _ in dataset.samples if 'benign' in img_file.lower())
    malignant_count = num_samples - benign_count
    print(f"Доброкачественные (benign): {benign_count} ({benign_count/num_samples*100:.1f}%)")
    print(f"Злокачественные (malignant): {malignant_count} ({malignant_count/num_samples*100:.1f}%)")
    print(f"Баланс классов (Benign/Malignant): {benign_count / malignant_count:.2f}" if malignant_count > 0 else "N/A")

    total_cells = 0
    num_masks_analyzed = 0
    max_masks_to_analyze = min(50, num_samples)

    for i in range(max_masks_to_analyze):
        _, mask_file = dataset.samples[i]
        mask_path = os.path.join(dataset.mask_dir, mask_file)
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is not None:
            _, mask_binary = cv2.threshold(mask_raw, 127, 255, cv2.THRESH_BINARY)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
            num_cells_in_mask = num_labels - 1
            total_cells += num_cells_in_mask
            num_masks_analyzed += 1

    if num_masks_analyzed > 0:
        avg_cells_per_image = total_cells / num_masks_analyzed
        estimated_total_cells = avg_cells_per_image * num_samples
        print(f"Среднее кол-во клеток на изображении (по {num_masks_analyzed} маскам): {avg_cells_per_image:.1f}")
        print(f"Примерное общее кол-во клеток в '{split_name}': {estimated_total_cells:.0f}")
    else:
        print("Не удалось проанализировать маски для подсчета клеток.")

    print(f"Размеры изображений после трансформации: {IMG_HEIGHT}x{IMG_WIDTH}")

analyze_dataset_eda(train_dataset_no_aug, "Train")
analyze_dataset_eda(val_dataset, "Validation")
analyze_dataset_eda(test_dataset, "Test")

# %% [markdown]
# ## Part 2: Unet model
# 
# Implement class of Unet model according with [the original paper](https://arxiv.org/pdf/1505.04597).
# Ajust size of the network according with your input data.

# %%
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
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

unet_model = UNet(n_channels=3, n_classes=1).to(device)
dummy_batch = torch.randn(2, 3, IMG_HEIGHT, IMG_WIDTH).to(device)
with torch.no_grad():
    output = unet_model(dummy_batch)
print(f"\n--- Проверка U-Net ---")
print(f"Входной батч: {dummy_batch.shape}")
print(f"Выходной батч: {output.shape}")
assert output.shape == (2, 1, IMG_HEIGHT, IMG_WIDTH)
print("Проверка U-Net прошла успешно.")
del unet_model, dummy_batch, output
if torch.cuda.is_available(): torch.cuda.empty_cache()

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
from torchmetrics.segmentation  import DiceScore
from torchmetrics.classification import JaccardIndex

# %%

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
    def __init__(self, alpha=0.5, smooth_dice=1.0):
        super().__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth_dice)
    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.alpha * bce + (1 - self.alpha) * dice

class CancerSegmentationModule(pl.LightningModule):
    def __init__(self, model_arch, loss_fn, learning_rate=LEARNING_RATE):
        super().__init__()
        self.model = model_arch
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.val_dice = DiceScore(num_classes=1, average='micro')
        self.val_iou = JaccardIndex(task="binary")
        self.save_hyperparameters(ignore=['model_arch', 'loss_fn'])

    def forward(self, x):
        return self.model(x)

    def _prepare_loss_inputs(self, logits, targets):
        logits_out = logits
        targets_out = targets

        if logits_out.shape[1:] != targets_out.shape[1:]:
            if len(logits_out.shape) == 4 and logits_out.shape[1] == 1:
                 logits_out = logits_out.squeeze(1)
            if len(targets_out.shape) == 4 and targets_out.shape[1] == 1:
                 targets_out = targets_out.squeeze(1)

        if logits_out.shape != targets_out.shape:
            if len(logits_out.shape) == 4 and len(targets_out.shape) == 3:
                 logits_out = logits_out.squeeze(1)
            elif len(targets_out.shape) == 4 and len(logits_out.shape) == 3:
                  targets_out = targets_out.squeeze(1)
            elif len(logits_out.shape) == 4 and len(targets_out.shape) == 4:
                   logits_out = logits_out.squeeze(1)
                   targets_out = targets_out.squeeze(1)
        return logits_out, targets_out


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_float = y.float()
        logits_for_loss, y_float_for_loss = self._prepare_loss_inputs(logits, y_float)
        try:
            loss = self.loss_fn(logits_for_loss, y_float_for_loss)
        except ValueError as e:
            print(f"Loss function error: {e}")
            print(f"Logits shape: {logits_for_loss.shape}, Targets shape: {y_float_for_loss.shape}")
            raise e
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_float = y.float()

        logits_for_loss, y_float_for_loss = self._prepare_loss_inputs(logits, y_float)
        try:
            loss = self.loss_fn(logits_for_loss, y_float_for_loss)
        except ValueError as e:
            print(f"Validation loss error: {e}")
            print(f"Logits shape: {logits_for_loss.shape}, Targets shape: {y_float_for_loss.shape}")
            loss = torch.tensor(float('nan'))

        preds = torch.sigmoid(logits)
        y_int = y.int()

        preds_for_metric = preds[:, 0, :, :]
        y_int_for_metric = y_int[:, 0, :, :]

        try:
            self.val_dice.update(preds_for_metric, y_int_for_metric)
            self.val_iou.update(preds_for_metric, y_int_for_metric)
        except ValueError as e:
             print(f"Metric update error: {e}")
             print(f"Preds shape for metric: {preds_for_metric.shape}, Target shape for metric: {y_int_for_metric.shape}")
             print(f"Preds dtype: {preds_for_metric.dtype}, Target dtype: {y_int_for_metric.dtype}")
             pass


        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        try:
            dice_epoch = self.val_dice.compute()
            iou_epoch = self.val_iou.compute()
            self.log('val_dice', dice_epoch, prog_bar=True, logger=True)
            self.log('val_iou', iou_epoch, prog_bar=True, logger=True)
        except Exception as e:
             print(f"Error computing metrics on epoch end: {e}")
             self.log('val_dice', 0.0, prog_bar=True, logger=True)
             self.log('val_iou', 0.0, prog_bar=True, logger=True)
        finally:
            self.val_dice.reset()
            self.val_iou.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_float = y.float()

        logits_for_loss, y_float_for_loss = self._prepare_loss_inputs(logits, y_float)
        try:
            loss = self.loss_fn(logits_for_loss, y_float_for_loss)
        except ValueError:
             loss = torch.tensor(float('nan'))

        preds = torch.sigmoid(logits)
        y_int = y.int()

        preds_for_metric = preds[:, 0, :, :]
        y_int_for_metric = y_int[:, 0, :, :]

        try:
            self.val_dice.update(preds_for_metric, y_int_for_metric)
            self.val_iou.update(preds_for_metric, y_int_for_metric)
        except ValueError as e:
            print(f"Test metric update error: {e}")
            pass

        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_dice', self.val_dice, on_epoch=True, prog_bar=True)
        self.log('test_iou', self.val_iou, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

train_loader_no_aug = DataLoader(
    train_dataset_no_aug, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True, persistent_workers=True if NUM_WORKERS > 0 else False
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False
)

loss_functions = {
    'BCE': nn.BCEWithLogitsLoss(),
    'Dice': DiceLoss(),
    'Combined': CombinedLoss(alpha=0.5)
}
trained_models_no_aug = {}
results_no_aug = {}

for loss_name, loss_fn in loss_functions.items():
    print(f"\n--- Обучение U-Net без аугментаций с функцией потерь: {loss_name} ---")
    unet_no_aug = UNet(n_channels=3, n_classes=1)
    lightning_model = CancerSegmentationModule(model_arch=unet_no_aug, loss_fn=loss_fn, learning_rate=LEARNING_RATE)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_dice', dirpath=CHECKPOINT_DIR,
        filename=f'unet_{loss_name}_no_aug_best_dice={{val_dice:.4f}}',
        save_top_k=1, mode='max', save_last=True
    )
    early_stop_callback = EarlyStopping(
        monitor='val_dice', patience=PATIENCE, verbose=True, mode='max'
    )

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1, max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=pl.loggers.TensorBoardLogger("tb_logs", name=f"unet_{loss_name}_no_aug"),
        log_every_n_steps=5,
        deterministic=False,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        enable_progress_bar=True
    )

    print(f"Запуск обучения для {loss_name}...")
    trainer.fit(lightning_model, train_loader_no_aug, val_loader)

    best_path = checkpoint_callback.best_model_path
    if best_path and os.path.exists(best_path):
       trained_models_no_aug[loss_name] = best_path
       print(f"Лучшая модель для {loss_name} (без аугментаций) сохранена в: {best_path}")

       unet_instance_for_load = UNet(n_channels=3, n_classes=1)
       current_loss_fn = loss_functions[loss_name]
       best_model = CancerSegmentationModule.load_from_checkpoint(
           checkpoint_path=best_path,
           model_arch=unet_instance_for_load,
           loss_fn=current_loss_fn
       )
       val_results = trainer.validate(best_model, val_loader, verbose=False)
       results_no_aug[loss_name] = val_results[0]
       print(f"Метрики лучшей модели {loss_name} на валидации: {val_results[0]}")

    else:
       last_ckpt_path = checkpoint_callback.last_model_path
       if last_ckpt_path and os.path.exists(last_ckpt_path):
            print(f"Используем последний checkpoint: {last_ckpt_path}")
            trained_models_no_aug[loss_name] = last_ckpt_path
            unet_instance_for_load = UNet(n_channels=3, n_classes=1)
            current_loss_fn = loss_functions[loss_name]

            last_model = CancerSegmentationModule.load_from_checkpoint(
                checkpoint_path=last_ckpt_path,
                model_arch=unet_instance_for_load,
                loss_fn=current_loss_fn
            )

            val_results = trainer.validate(last_model, val_loader, verbose=False)
            results_no_aug[loss_name] = val_results[0]
            print(f"Метрики последней модели {loss_name} на валидации: {val_results[0]}")
       else:
            print(f"Последний checkpoint для {loss_name} также не найден.")
            results_no_aug[loss_name] = {'val_loss': float('inf'), 'val_dice': 0.0, 'val_iou': 0.0}


    del unet_no_aug, lightning_model, trainer, checkpoint_callback, early_stop_callback
    if 'best_model' in locals(): del best_model
    if 'last_model' in locals(): del last_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n--- Обучение без аугментаций завершено ---")
print("Пути к лучшим сохраненным моделям без аугментаций:")
for name, path in trained_models_no_aug.items():
    print(f"{name}: {path}")
print("\nРезультаты валидации без аугментаций:")
results_df_no_aug = pd.DataFrame(results_no_aug).T
print(results_df_no_aug)

# %% [markdown]
# ## Part 3.1: Losses conclusion
# 
# Analyse results of the three models above using metrics, losses and visualizations you know (all three parts are required).
# 
# Make motivated conclusion on which setup is better. Provide your arguments.
# 
# Calculate loss and metrics of the best model on test set.

# %%
print("\n--- losses conclusion ---")
best_loss_name_no_aug = results_df_no_aug['val_dice'].idxmax()
best_model_path_no_aug = trained_models_no_aug.get(best_loss_name_no_aug)

print(f"Лучшая функция потерь по val_dice: {best_loss_name_no_aug}")
print(f"Лучшие метрики на валидации ({best_loss_name_no_aug}):")
print(results_df_no_aug.loc[best_loss_name_no_aug])

def evaluate_model_on_test(model_module, dataloader, device):
    model_module.to(device)
    model_module.eval()
    test_dice_metric = DiceScore(num_classes=1, average='micro').to(device)
    test_iou_metric = JaccardIndex(task="binary").to(device)
    total_loss = 0.0
    loss_count = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model_module(x)
            y_float = y.float()

            logits_for_loss, y_float_for_loss = model_module._prepare_loss_inputs(logits, y_float)
            try:
                loss = model_module.loss_fn(logits_for_loss, y_float_for_loss)
                total_loss += loss.item()
                loss_count += 1
            except ValueError as e:
                 print(f"Test loss error: {e}")
                 pass

            preds = torch.sigmoid(logits)
            y_int = y.int()

            preds_for_metric = preds[:, 0, :, :]
            y_int_for_metric = y_int[:, 0, :, :]

            try:
                test_dice_metric.update(preds_for_metric, y_int_for_metric)
                test_iou_metric.update(preds_for_metric, y_int_for_metric)
            except ValueError as e:
                 print(f"Test metric update error: {e}")
                 pass

    final_loss = total_loss / loss_count if loss_count > 0 else float('inf')
    final_dice = test_dice_metric.compute().item()
    final_iou = test_iou_metric.compute().item()

    test_dice_metric.reset()
    test_iou_metric.reset()

    return final_loss, final_dice, final_iou

def plot_predictions(model, dataset, indices, model_name=""):
    model.to(device)
    model.eval()
    plt.figure(figsize=(15, 5 * len(indices)))
    for i, idx in enumerate(indices):
        image, mask = dataset[idx]

        image_vis = image.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_vis = std * image_vis + mean
        image_vis = np.clip(image_vis, 0, 1)
        mask_np = mask.squeeze().cpu().numpy()

        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))
            pred = torch.sigmoid(logits).squeeze().cpu().numpy()

        plt.subplot(len(indices), 3, i * 3 + 1)
        plt.imshow(image_vis)
        plt.title(f"Image (Index: {idx})")
        plt.axis('off')

        plt.subplot(len(indices), 3, i * 3 + 2)
        plt.imshow(mask_np, cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')

        plt.subplot(len(indices), 3, i * 3 + 3)
        plt.imshow(pred > 0.5, cmap='gray')
        plt.title(f"Prediction ({model_name})")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if results_no_aug and trained_models_no_aug:
    results_df_no_aug = pd.DataFrame(results_no_aug).T
    if 'val_dice' not in results_df_no_aug.columns:
         print("В результатах валидации отсутствует колонка 'val_dice'.")

         best_loss_name_no_aug = results_df_no_aug['val_loss'].idxmin() if 'val_loss' in results_df_no_aug.columns else list(results_no_aug.keys())[0]
         print(f"Не удалось определить лучший лосс по Dice, выбран по Loss: {best_loss_name_no_aug}")
    else:
        best_loss_name_no_aug = results_df_no_aug['val_dice'].idxmax()

    best_model_path_no_aug = trained_models_no_aug.get(best_loss_name_no_aug)

    print("\nРезультаты валидации моделей без аугментаций:")
    print(results_df_no_aug)
    print(f"\nЛучшая функция потерь по val_dice (или val_loss): {best_loss_name_no_aug}")
    if best_loss_name_no_aug in results_df_no_aug.index:
        print(f"Лучшие метрики на валидации ({best_loss_name_no_aug}):")
        print(results_df_no_aug.loc[best_loss_name_no_aug])

    if best_model_path_no_aug and os.path.exists(best_model_path_no_aug):
        print(f"\nЗагрузка лучшей модели: {best_loss_name_no_aug} из {best_model_path_no_aug}")

        unet_instance_for_load = UNet(n_channels=3, n_classes=1)
        current_loss_fn = loss_functions[best_loss_name_no_aug]

        best_unet_no_aug = CancerSegmentationModule.load_from_checkpoint(
            checkpoint_path=best_model_path_no_aug,
            model_arch=unet_instance_for_load,
            loss_fn=current_loss_fn
        )

        print(f"\nВизуализация предсказаний лучшей модели ({best_loss_name_no_aug} без ауг.) на тестовых данных:")
        plot_indices = np.random.choice(len(test_dataset), min(3, len(test_dataset)), replace=False)
        plot_predictions(best_unet_no_aug, test_dataset, plot_indices, f"U-Net ({best_loss_name_no_aug}, No Aug)")

        print(f"\nОценка лучшей модели ({best_loss_name_no_aug} без ауг.) на тестовом наборе:")
        test_loss, test_dice, test_iou = evaluate_model_on_test(best_unet_no_aug, test_loader, device)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Dice: {test_dice:.4f}")
        print(f"Test IoU: {test_iou:.4f}")


        test_results_best_no_aug = {'test_loss': test_loss, 'test_dice': test_dice, 'test_iou': test_iou}

        del unet_instance_for_load, current_loss_fn, best_unet_no_aug
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    else:
        test_results_best_no_aug = None

# %% [markdown]
# Сравнение метрик Dice и IoU на валидационном наборе показывает, что в данном случае, Dice показал лучший результат по Dice (0.0584).
# Dice Loss и Combined Loss часто предпочтительнее для сегментации из-за лучшей работы с дисбалансом классов (много фона, мало клеток) по сравнению с чистым BCE.
# По картинкам видно наличие ложных срабатываний.
# Результаты лучшей модели (Dice) без аугментаций на тестовом наборе: Dice=0.0611, IoU=0.0836.

# %%
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
log_dir_base = "tb_logs"

loss_functions_to_plot = ['BCE', 'Dice', 'Combined']
results_for_plot = {}

plt.figure(figsize=(12, 7))
plt.title("Validation Loss during Training (No Augmentations)")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.grid(True)

for loss_name in loss_functions_to_plot:
    log_subdir_name = f"unet_{loss_name}_no_aug"
    full_log_dir = None

    if os.path.exists(os.path.join(log_dir_base, log_subdir_name)):
         version_dirs = sorted([d for d in os.listdir(os.path.join(log_dir_base, log_subdir_name)) if d.startswith('version_')])
         if version_dirs:
              full_log_dir = os.path.join(log_dir_base, log_subdir_name, version_dirs[-1])

    if full_log_dir and os.path.exists(full_log_dir):
        print(f"Reading logs for {loss_name} from: {full_log_dir}")
        try:
            event_acc = EventAccumulator(full_log_dir)
            event_acc.Reload()

            val_loss_tag = 'val_loss'
            if val_loss_tag in event_acc.Tags()['scalars']:
                val_loss_events = event_acc.Scalars(val_loss_tag)
                steps = [e.step for e in val_loss_events]
                losses = [e.value for e in val_loss_events]
                plt.plot(steps, losses, marker='.', linestyle='-', label=f'Val Loss ({loss_name})')
                print(f"  Found {len(steps)} points for {val_loss_tag}")

        except Exception as e:
            print(f"Error reading logs for {loss_name}: {e}")
plt.legend()
plt.ylim(bottom=0)
plt.show()

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
!pip install segmentation-models-pytorch -q
import segmentation_models_pytorch as smp

# %%
train_transform_augmented = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH, interpolation=cv2.INTER_NEAREST),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=40, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
    A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=0, p=0.4, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
    A.GaussNoise(var_limit=(10.0, 60.0), p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),

    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(transpose_mask=True),
])

train_dataset_augmented = CancerCellsDataset(IMAGE_DIR, MASK_DIR, train_samples, transform=train_transform_augmented)
train_loader_augmented = DataLoader(
    train_dataset_augmented, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True, persistent_workers=True if NUM_WORKERS > 0 else False
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False
)

print("\nПример аугментированного изображения и маски:")
visualize_sample(train_dataset_augmented, index=np.random.randint(len(train_dataset_augmented)), title="Train Sample (Augmented)")

try:
    pspnet_model = smp.PSPNet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    print("\nМодель PSPNet успешно создана.")
except Exception as e:
    print(f"\nОшибка при создании PSPNet: {e}")
    pspnet_model = None

# Обучаемся на аугментированных данных

trained_models_aug = {}
results_aug = {}

if 'best_loss_name_no_aug' not in locals() or best_loss_name_no_aug is None:
     best_loss_name_aug = 'Combined'
else:
     best_loss_name_aug = best_loss_name_no_aug
best_loss_fn = loss_functions[best_loss_name_aug]
print(f"\nИспользуется лучшая функция потерь: {best_loss_name_aug}")


# Модели для обучения с аугментациями
models_to_train_aug = {
    'UNet_Aug': UNet(n_channels=3, n_classes=1),
}
if pspnet_model is not None:
    models_to_train_aug['PSPNet_Aug'] = pspnet_model


# Цикл обучения
AUG_EPOCHS = MAX_EPOCHS + 10
AUG_PATIENCE = PATIENCE + 5

for model_name, model_arch in models_to_train_aug.items():
    print(f"\n--- Обучение {model_name} с аугментациями и лоссом {best_loss_name_aug} ---")

    lightning_model_aug = CancerSegmentationModule(model_arch=model_arch, loss_fn=best_loss_fn, learning_rate=LEARNING_RATE)

    checkpoint_callback_aug = ModelCheckpoint(
        monitor='val_dice', dirpath=CHECKPOINT_DIR,
        filename=f'{model_name.lower()}_best_dice={{val_dice:.4f}}',
        save_top_k=1, mode='max', save_last=True
    )
    early_stop_callback_aug = EarlyStopping(
        monitor='val_dice', patience=AUG_PATIENCE,
        verbose=True, mode='max'
    )

    trainer_aug = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1, max_epochs=AUG_EPOCHS,
        callbacks=[checkpoint_callback_aug, early_stop_callback_aug],
        logger=pl.loggers.TensorBoardLogger("tb_logs", name=f"{model_name.lower()}_{best_loss_name_aug}_aug"),
        log_every_n_steps=5,
        deterministic=False,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        enable_progress_bar=True
    )

    print(f"Запуск обучения для {model_name}...")
    trainer_aug.fit(lightning_model_aug, train_loader_augmented, val_loader)

    best_path_aug = checkpoint_callback_aug.best_model_path
    if best_path_aug and os.path.exists(best_path_aug):
       trained_models_aug[model_name] = best_path_aug
       print(f"Лучшая модель для {model_name} сохранена в: {best_path_aug}")
       if "unet" in model_name.lower():
           arch_instance = UNet(n_channels=3, n_classes=1)
       elif "pspnet" in model_name.lower() and pspnet_model is not None:
            arch_instance = smp.PSPNet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
       else:
           arch_instance = None

       if arch_instance:
           try:
               best_model_aug = CancerSegmentationModule.load_from_checkpoint(
                   checkpoint_path=best_path_aug,
                   model_arch=arch_instance,
                   loss_fn=best_loss_fn
               )
               val_results_aug = trainer_aug.validate(best_model_aug, val_loader, verbose=False)
               if val_results_aug:
                    results_aug[model_name] = val_results_aug[0]
                    print(f"Метрики лучшей модели {model_name} на валидации: {val_results_aug[0]}")
               else:
                    results_aug[model_name] = {'val_loss': float('inf'), 'val_dice': 0.0, 'val_iou': 0.0}

           except Exception as e:
               results_aug[model_name] = {'val_loss': float('inf'), 'val_dice': 0.0, 'val_iou': 0.0}
       else:
            results_aug[model_name] = {'val_loss': float('inf'), 'val_dice': 0.0, 'val_iou': 0.0}


    else:
       print(f"Не удалось найти лучшую модель для {model_name} по пути: {best_path_aug}.")
       last_ckpt_path_aug = checkpoint_callback_aug.last_model_path
       if last_ckpt_path_aug and os.path.exists(last_ckpt_path_aug):
            print(f"Используем последний checkpoint: {last_ckpt_path_aug}")
            trained_models_aug[model_name] = last_ckpt_path_aug
            if "unet" in model_name.lower():
               arch_instance = UNet(n_channels=3, n_classes=1)
            elif "pspnet" in model_name.lower() and pspnet_model is not None:
               arch_instance = smp.PSPNet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
            else:
               arch_instance = None

            if arch_instance:
               try:
                   last_model_aug = CancerSegmentationModule.load_from_checkpoint(
                       checkpoint_path=last_ckpt_path_aug,
                       model_arch=arch_instance,
                       loss_fn=best_loss_fn
                   )
                   val_results_aug = trainer_aug.validate(last_model_aug, val_loader, verbose=False)
                   if val_results_aug:
                       results_aug[model_name] = val_results_aug[0]
                       print(f"Метрики последней модели {model_name} на валидации: {val_results_aug[0]}")
                   else:
                       print(f"trainer.validate для последней модели {model_name} не вернул результатов.")
                       results_aug[model_name] = {'val_loss': float('inf'), 'val_dice': 0.0, 'val_iou': 0.0}

               except Exception as e:
                   print(f"Ошибка при загрузке или валидации последней модели {model_name}: {e}")
                   results_aug[model_name] = {'val_loss': float('inf'), 'val_dice': 0.0, 'val_iou': 0.0}
            else:
               results_aug[model_name] = {'val_loss': float('inf'), 'val_dice': 0.0, 'val_iou': 0.0}

       else:
            print(f"Последний checkpoint для {model_name} также не найден.")
            results_aug[model_name] = {'val_loss': float('inf'), 'val_dice': 0.0, 'val_iou': 0.0}


    # Очистка памяти
    del model_arch, lightning_model_aug, trainer_aug, checkpoint_callback_aug, early_stop_callback_aug
    if 'best_model_aug' in locals(): del best_model_aug
    if 'last_model_aug' in locals(): del last_model_aug
    if 'arch_instance' in locals(): del arch_instance
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n--- Обучение С аугментациями завершено ---")
print("Пути к лучшим сохраненным моделям (с аугментациями):")
for name, path in trained_models_aug.items():
    print(f"{name}: {path}")
print("\nРезультаты валидации (с аугментациями):")
if results_aug:
    results_df_aug = pd.DataFrame(results_aug).T
    print(results_df_aug)

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
# your code is here
if 'best_loss_name_no_aug' not in locals() or best_loss_name_no_aug is None:
    print("Имя лучшего лосса не определено. Пожалуйста, запустите часть 3.1.")
    # Устанавливаем значение по умолчанию, если возможно
    best_loss_name_no_aug = 'Combined' if 'Combined' in loss_functions else list(loss_functions.keys())[0]
    print(f"Используется лосс по умолчанию: {best_loss_name_no_aug}")

loss_fn_to_use = loss_functions.get(best_loss_name_no_aug)

model_paths_to_compare = {}

# U-Net без аугментаций
path1 = trained_models_no_aug.get(best_loss_name_no_aug)
if path1 and os.path.exists(path1):
    model_paths_to_compare[f'U-Net ({best_loss_name_no_aug}, No Aug)'] = path1

# U-Net с аугментациями
path2 = trained_models_aug.get('UNet_Aug')
if path2 and os.path.exists(path2):
    model_paths_to_compare[f'U-Net ({best_loss_name_no_aug}, Aug)'] = path2

# PSPNet с аугментациями
path3 = trained_models_aug.get('PSPNet_Aug')
if path3 and os.path.exists(path3):
    model_paths_to_compare[f'PSPNet ({best_loss_name_no_aug}, Aug)'] = path3

test_results_comparison = {}

print("\n--- Оценка моделей на тестовом наборе ---")
for model_name, checkpoint_path in model_paths_to_compare.items():
    print(f"\nОценка модели: {model_name}")
    print(f"Загрузка из: {checkpoint_path}")


    if "unet" in model_name.lower():
        arch_instance = UNet(n_channels=3, n_classes=1)
    elif "pspnet" in model_name.lower():
         try:
             arch_instance = smp.PSPNet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
         except Exception as e:
             print(f"Ошибка при создании PSPNet для загрузки: {e}.")
             continue

    try:
        loaded_model = CancerSegmentationModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_arch=arch_instance,
            loss_fn=loss_fn_to_use
        )
    except Exception as e:
        print(f"Ошибка при загрузке модели {model_name}: {e}")
        continue

    test_loss, test_dice, test_iou = evaluate_model_on_test(loaded_model, test_loader, device)
    print(f"Результаты теста: Loss={test_loss:.4f}, Dice={test_dice:.4f}, IoU={test_iou:.4f}")
    test_results_comparison[model_name] = {'test_loss': test_loss, 'test_dice': test_dice, 'test_iou': test_iou}

    del arch_instance, loaded_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

print("\n--- Сравнение результатов на тестовом наборе ---")
if test_results_comparison:
    results_df_final = pd.DataFrame(test_results_comparison).T.sort_values(by='test_dice', ascending=False)
    print(results_df_final)

    best_model_name_final = results_df_final.index[0]
    best_test_dice = results_df_final.loc[best_model_name_final, 'test_dice']
    best_test_iou = results_df_final.loc[best_model_name_final, 'test_iou']
    print(f"\nЛучший сетап по Test Dice: '{best_model_name_final}'")
    print(f"  Test Dice: {best_test_dice:.4f}")
    print(f"  Test IoU: {best_test_iou:.4f}")

    baseline_model_name = f'U-Net ({best_loss_name_no_aug}, No Aug)'
    if 'test_results_best_no_aug' in locals() and test_results_best_no_aug is not None:
        initial_best_dice = test_results_best_no_aug.get('test_dice', 0)
        initial_best_iou = test_results_best_no_aug.get('test_iou', 0)
        dice_improvement = best_test_dice - initial_best_dice
        iou_improvement = best_test_iou - initial_best_iou
        print(f"\nУлучшение по сравнению с '{baseline_model_name}':")
        print(f"  Улучшение Dice: {dice_improvement:+.4f}")
        print(f"  Улучшение IoU: {iou_improvement:+.4f}")
    elif baseline_model_name in results_df_final.index:
        print("\nПредупреждение: Результаты теста для базовой модели не были сохранены отдельно, используем текущие.")
        initial_best_dice = results_df_final.loc[baseline_model_name, 'test_dice']
        initial_best_iou = results_df_final.loc[baseline_model_name, 'test_iou']
        dice_improvement = best_test_dice - initial_best_dice
        iou_improvement = best_test_iou - initial_best_iou
        print(f"\nУлучшение по сравнению с '{baseline_model_name}':")
        print(f"  Улучшение Dice: {dice_improvement:+.4f}")
        print(f"  Улучшение IoU: {iou_improvement:+.4f}")

    print("\nВизуализация предсказаний на тестовых данных для сравнения:")
    loaded_models_for_viz = {}
    for name, path in model_paths_to_compare.items():
         if "unet" in name.lower():
             arch_instance = UNet(n_channels=3, n_classes=1)
         elif "pspnet" in name.lower():
             try:
                 arch_instance = smp.PSPNet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
             except Exception: continue
         else: continue
         try:
            loaded_models_for_viz[name] = CancerSegmentationModule.load_from_checkpoint(
                checkpoint_path=path, model_arch=arch_instance, loss_fn=loss_fn_to_use
            )
         except Exception as e:
            print(f"Ошибка {name} для визуализации: {e}")


    if loaded_models_for_viz:
        num_viz_samples = min(3, len(test_dataset))
        plot_indices = np.random.choice(len(test_dataset), num_viz_samples, replace=False)

        for idx in plot_indices:
            n_cols = 2 + len(loaded_models_for_viz)
            plt.figure(figsize=(5 * n_cols, 5))

            image, mask = test_dataset[idx]
            image_vis = image.permute(1, 2, 0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
            image_vis = std * image_vis + mean; image_vis = np.clip(image_vis, 0, 1)
            mask_np = mask.squeeze().cpu().numpy()

            plt.subplot(1, n_cols, 1); plt.imshow(image_vis); plt.title(f"Image {idx}"); plt.axis('off')
            plt.subplot(1, n_cols, 2); plt.imshow(mask_np, cmap='gray'); plt.title("Ground Truth"); plt.axis('off')


            plot_counter = 3

            sorted_model_names = sorted(loaded_models_for_viz.keys())
            for name in sorted_model_names:
                 model = loaded_models_for_viz[name]
                 model.to(device).eval()
                 with torch.no_grad():
                     logits = model(image.unsqueeze(0).to(device))
                     pred = torch.sigmoid(logits).squeeze().cpu().numpy()
                 plt.subplot(1, n_cols, plot_counter); plt.imshow(pred > 0.5, cmap='gray'); plt.title(name); plt.axis('off')
                 plot_counter += 1
            plt.tight_layout()
            plt.show()

        del loaded_models_for_viz
        if torch.cuda.is_available(): torch.cuda.empty_cache()

# %%
configs_to_plot = {
    f"U-Net (No Aug, {best_loss_name_no_aug})": {
        "log_pattern": f"unet_{best_loss_name_no_aug}_no_aug",
        "color": "blue"
    },
    f"U-Net (Aug, {best_loss_name_no_aug})": {
        "log_pattern": f"unet_aug_{best_loss_name_no_aug}_aug",
        "color": "red"
    },
    f"PSPNet (Aug, {best_loss_name_no_aug})": {
        "log_pattern": f"pspnet_aug_{best_loss_name_no_aug}_aug",
        "color": "green"
    }
}

plt.figure(figsize=(14, 8))
plt.title(f"Сравнение Train/Validation Loss (Best Loss: {best_loss_name_no_aug})")
plt.xlabel("Эпоха")
plt.ylabel("Значение функции потерь")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

all_min_losses = []
all_max_losses = []

for config_name, config_details in configs_to_plot.items():
    log_subdir_name = config_details["log_pattern"]
    color = config_details["color"]
    full_log_dir = None
    expected_base_path = os.path.join(log_dir_base, log_subdir_name)

    if os.path.exists(expected_base_path):
        version_dirs = sorted([d for d in os.listdir(expected_base_path) if d.startswith('version_')])
        if version_dirs:
            full_log_dir = os.path.join(expected_base_path, version_dirs[-1])
            print(f"Чтение логов для '{config_name}' из: {full_log_dir}")

    if full_log_dir and os.path.exists(full_log_dir):
        try:
            event_acc = EventAccumulator(full_log_dir).Reload()
            tags = event_acc.Tags()['scalars']

            train_loss_tag = 'train_loss_epoch'
            val_loss_tag = 'val_loss'

            if train_loss_tag in tags:
                train_events = event_acc.Scalars(train_loss_tag)
                train_epochs = {e.step: e.value for e in train_events}
                sorted_train_epochs = sorted(train_epochs.keys())
                train_losses = [train_epochs[e] for e in sorted_train_epochs]
                if train_losses:
                    plt.plot(sorted_train_epochs, train_losses, marker='.', linestyle='-', color=color, label=f'{config_name} Train')
                    all_min_losses.append(np.min(train_losses))
                    all_max_losses.append(np.max(train_losses))
                    print(f"  Найдено {len(sorted_train_epochs)} точек для Train Loss")

            if val_loss_tag in tags:
                val_events = event_acc.Scalars(val_loss_tag)
                val_epochs = {e.step: e.value for e in val_events}
                sorted_val_epochs = sorted(val_epochs.keys())
                val_losses = [val_epochs[e] for e in sorted_val_epochs]
                if val_losses:
                    plt.plot(sorted_val_epochs, val_losses, marker='x', linestyle='--', color=color, label=f'{config_name} Val')
                    all_min_losses.append(np.min(val_losses))
                    all_max_losses.append(np.max(val_losses))
                    print(f"  Найдено {len(sorted_val_epochs)} точек для Val Loss")

        except Exception as e:
            print(f"Ошибка при чтении логов для '{config_name}': {e}")

# Настройка пределов оси Y
if all_min_losses and all_max_losses:
    global_min = min(all_min_losses)
    global_max = max(all_max_losses)

    reasonable_max = np.percentile([l for conf_losses in results_no_aug.values() for l in conf_losses.values() if isinstance(l, (int, float))] +
                                 [l for conf_losses in results_aug.values() for l in conf_losses.values() if isinstance(l, (int, float))], 95)

    upper_limit = min(global_max, reasonable_max * 1.5, 1.0)
    plt.ylim(bottom=max(0, global_min - 0.05), top=upper_limit + 0.1)
    print(f"\nУстановлены пределы оси Y: [{max(0, global_min - 0.05):.2f}, {upper_limit + 0.1:.2f}]")
else:
     plt.ylim(bottom=0)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# %% [markdown]
# Возможно стоит дольше учить, чтоб PSPNet показывал лучшие результаты, но по сравнению с UNet видно, что он явно лучше справляется с данной задачей. По графику видим, что потери ниже.
# 
# Еще интересно сравнить UNet с аугментациями и без: с аугментациями лучше. Без аугментаций модель быстро запоминает train, а на валидации высокие потери. Аугментации с этим хорошо борются.
# 


