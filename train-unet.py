import os
import cv2
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

import matplotlib.pyplot as plt

def visualize_prediction(model, dataset, idx=0, device='cuda', save_path='prediction.png', compare=True):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model.eval()
    image, mask = dataset[idx]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        pred = torch.sigmoid(model(image_tensor))
        pred = (pred > 0.5).float().cpu().squeeze().numpy()

    image_np = image.permute(1, 2, 0).cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    
    min_val = image_np.min()
    max_val = image_np.max()
    if max_val > min_val: # Avoid division by zero if the image is flat
        image_np = (image_np - min_val) / (max_val - min_val)
    image_np = np.clip(image_np, 0, 1) # Ensure it's strictly within [0,1]

    if compare:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(image_np)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("GT Mask")
        plt.imshow(mask_np, cmap='gray')
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(pred, cmap='gray')
    else:
        # Only save the predicted mask, resized to original image dimensions
        
        # Get original image dimensions
        original_img_path = dataset.image_paths[idx] # Assumes dataset has image_paths
        original_image_for_dims = cv2.imread(original_img_path)
        
        if original_image_for_dims is None:
            print(f"Warning: Could not read original image {original_img_path} to get dimensions. Saving prediction at model resolution.")
            pred_to_save = pred
        else:
            h_orig, w_orig = original_image_for_dims.shape[:2]
            # Resize predicted mask (pred) to original dimensions
            # pred is currently at model's output resolution (e.g., img_size x img_size)
            pred_resized_to_original = cv2.resize(pred, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            pred_to_save = pred_resized_to_original

        # The figsize here is for the Matplotlib figure. 
        # The actual pixel dimensions of the saved image content are determined by pred_to_save.
        plt.figure(dpi=300, figsize=(w_orig / 100, h_orig / 100))  # Adjust the figsize based on original image size
        plt.imshow(pred_to_save, cmap='gray')
    
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    
    plt.close()


# ============ 1. Dataset from YOLO ============
class YoloLineSegDataset(Dataset):
    def __init__(self, dataset_path, phase = 'train', transform=None, img_size=256):
        self.image_dir = os.path.join(dataset_path, 'images', phase)
        self.image_paths = sorted(glob(os.path.join(self.image_dir, '*.jpg')) + glob(os.path.join(self.image_dir, '*.png')))
        self.label_dir = os.path.join(dataset_path, 'labels', phase)
        self.class_to_index_file = os.path.join(dataset_path, 'class_to_index.yaml')
        self.class_to_index = yaml.safe_load(open(self.class_to_index_file, 'r'))
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def yolo_to_mask(self, label_path, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        if not os.path.exists(label_path):
            return mask
        with open(label_path, 'r') as f:
            for line in f.readlines():
                info = line.strip().split()
                cls = int(info[0])
                coords = list(map(float, info[1:]))
                if cls != self.class_to_index['lines']:
                    continue
                
                # coords is a list of points [x1, y1, x2, y2, ...]
                # draw a polygon on the mask
                points_normalized = np.array(coords).reshape(-1, 2)
                points_scaled_float = np.copy(points_normalized)
                points_scaled_float[:, 0] *= w
                points_scaled_float[:, 1] *= h
                points_int32 = points_scaled_float.astype(np.int32)
                if points_int32.shape[0] >= 3:
                    cv2.fillConvexPoly(mask, points_int32, color=(255,))

        return mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, img_id + '.txt')
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file {img_path} does not exist")
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file {label_path} does not exist for image {img_path}")
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        mask = self.yolo_to_mask(label_path, h, w).astype(np.float32) / 255.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)  # add channel dim
        return image, mask


# ============ 2. Transforms ============
def get_transforms(img_size=512, flip_prob=0.5):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=flip_prob),
        A.VerticalFlip(p=flip_prob),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])


# ============ 3. Training Function ============
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader):
        images, masks = images.to(device), masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ============ 4. Validation ============
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    iou_total = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = torch.sigmoid(model(images))
        preds = (preds > 0.5).float()

        intersection = (preds * masks).sum()
        union = preds.sum() + masks.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_total += iou.item()
    return iou_total / len(loader)


# ============ 5. Visualization ============
def visualize_best_model(model, dataset, device='cuda'):

    os.makedirs("../predictions", exist_ok=True)
    with tqdm(total=len(dataset), desc="Visualizing Predictions") as pbar:
        for i in range(len(dataset)):
            img_filename = os.path.basename(dataset.image_paths[i])
            output_dir = '../predictions'
            img_id = os.path.splitext(img_filename)[0]
            visualize_prediction(model, dataset, idx=i, device=device, save_path=os.path.join(output_dir, 'compare', f"{img_id}_compare.png"), compare=True)
            visualize_prediction(model, dataset, idx=i, device=device, save_path=os.path.join(output_dir, 'predication', f"{img_id}.png"), compare=False)
            pbar.update(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a UNet model for line segmentation")
    parser.add_argument('--dataset_dir', type=str, default='datasets/PGDP5K_yolo_seg', help='Path to the dataset directory')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size for the model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--model_path', type=str, default='models/best-unet.pth', help='Path to the pre-trained model weights')
    parser.add_argument('--model_save_path', type=str, default='models/best-unet-new.pth', help='Path to save the best model weights')
    return parser.parse_args()

# ============ 7. Main ============
def main():
    args = parse_args()
    model_path = args.model_path
    model_save_path = args.model_save_path
    dataset_dir = args.dataset_dir
    img_size = args.img_size
    epochs = args.epochs
    batch_size = args.batch_size

    print(f"Using model path: {model_path} from current directory {os.getcwd()}")

    print(f"Training UNet for line segmentation on dataset: {dataset_dir}")
    print(f"Image size: {img_size}, Batch size: {batch_size}, Epochs: {epochs}")

    # --- Dataset & Dataloader ---
    train_dataset = YoloLineSegDataset(dataset_dir, 'train', transform=get_transforms(img_size=512, flip_prob=0.5))
    val_dataset = YoloLineSegDataset(dataset_dir, 'val', transform=get_transforms(img_size=512, flip_prob=0.0))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    # --- Model ---
    model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights="imagenet", in_channels=3, classes=1)
    model = model.cuda()

    # Load pre-trained weights if available
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        model.load_state_dict(torch.load(model_path))
        print(f"✅ Loaded model weights from {model_path}")
    else:
        print(f"Pre-trained model not found at {model_path}. Training from scratch.")
        

    # --- Loss & Optimizer ---
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()

    def loss_fn(preds, targets):
        return dice_loss(preds, targets) + bce_loss(preds, targets)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_iou = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device='cuda')
        val_iou = evaluate(model, val_loader, device='cuda')
        
        print(f"Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Saved new best model (IoU: {val_iou:.4f})")

    
    # # Load best model for testing
    # model.load_state_dict(torch.load(model_path))
    # test_dataset = YoloLineSegDataset(dataset_dir, 'test', transform=get_transforms(512, flip_prob=0.0))
    # visualize_best_model(model, test_dataset, device='cuda')


if __name__ == "__main__":
    main()
