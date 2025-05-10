import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from model import get_model

class PreprocessedRetinopathyDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_id = self.dataframe.iloc[idx]['id_code']
        img_path = os.path.join(self.img_dir, img_id + '.png')
        label = self.dataframe.iloc[idx]['diagnosis']
        
        # Load preprocessed image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to range [0, 1] first
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor with channel-first format
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # Apply additional transforms if provided
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(is_training=True):
    """Get data transforms based on training mode."""
    if is_training:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ])
    else:
        return None

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=25, model_dir='../models'):
    since = time.time()
    best_model_wts = model.state_dict()
    best_kappa = 0.0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_kappa': [],
        'val_kappa': []
    }
    
    os.makedirs(model_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Use tqdm for progress tracking
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
            
            # Accumulate batch loss
            train_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate average batch loss
        train_loss = train_loss / len(train_loader)
        epoch_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        
        history['train_loss'].append(train_loss)
        history['train_kappa'].append(epoch_kappa)
        
        print(f'Train Loss: {train_loss:.4f} Kappa: {epoch_kappa:.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                # Calculate the loss
                loss = criterion(outputs, labels)
                
                # Accumulate the average batch loss
                val_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate average loss per batch
        val_loss = val_loss / len(val_loader)
        epoch_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        
        history['val_loss'].append(val_loss)
        history['val_kappa'].append(epoch_kappa)
        
        print(f'Val Loss: {val_loss:.4f} Kappa: {epoch_kappa:.4f}')
        
        # Deep copy the model if best performance
        if epoch_kappa > best_kappa:
            best_kappa = epoch_kappa
            best_model_wts = model.state_dict()
            # Save the best model
            torch.save(best_model_wts, os.path.join(model_dir, 'best_model.pth'))
            print(f"Saved new best model with Kappa: {epoch_kappa:.4f}")
        
        # Step the scheduler
        if scheduler:
            scheduler.step(epoch_kappa)
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Kappa: {best_kappa:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_kappa'], label='Train')
    plt.plot(history['val_kappa'], label='Validation')
    plt.title('Quadratic Weighted Kappa')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    
    return model, history

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data paths
    data_dir = "../datasets/aptos2019-blindness-detection"
    processed_dir = os.path.join(data_dir, "processed_train_images")
    train_split_csv = os.path.join(data_dir, "train_split.csv")
    val_split_csv = os.path.join(data_dir, "val_split.csv")
    
    # Check a sample image to understand its value range
    train_df = pd.read_csv(train_split_csv)
    sample_img_path = os.path.join(processed_dir, train_df.iloc[0]['id_code'] + '.png')
    sample_img = cv2.imread(sample_img_path)
    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    print(f"Sample image value range: min={sample_img.min()}, max={sample_img.max()}")
    print(f"Sample image shape: {sample_img.shape}")
    
    # Load the pre-calculated class weights
    class_weights_path = os.path.join(data_dir, "info", "class_weights.npy")
    if os.path.exists(class_weights_path):
        class_weights_dict = np.load(class_weights_path, allow_pickle=True).item()
        print("Loaded class weights:", class_weights_dict)
        class_weights = [class_weights_dict[i] for i in range(5)]
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    else:
        print("Class weights not found. Using equal weights.")
        class_weights_tensor = None
    
    # Load train and validation splits
    val_df = pd.read_csv(val_split_csv)
    
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    
    # Check class distribution
    print("\nTraining set class distribution:")
    train_class_counts = train_df['diagnosis'].value_counts().sort_index()
    for i, count in enumerate(train_class_counts):
        print(f"Class {i}: {count} images ({count/len(train_df)*100:.2f}%)")
    
    print("\nValidation set class distribution:")
    val_class_counts = val_df['diagnosis'].value_counts().sort_index()
    for i, count in enumerate(val_class_counts):
        print(f"Class {i}: {count} images ({count/len(val_df)*100:.2f}%)")
    
    # Create datasets and dataloaders
    train_transform = get_transforms(is_training=True)
    train_dataset = PreprocessedRetinopathyDataset(train_df, processed_dir, transform=train_transform)
    val_dataset = PreprocessedRetinopathyDataset(val_df, processed_dir, transform=None)
    
    # Create weighted sampler for training to address class imbalance
    if os.path.exists(class_weights_path):
        # Get class labels
        train_labels = train_df['diagnosis'].values
        # Calculate sample weights
        class_counts = np.bincount(train_labels)
        weight = 1. / class_counts
        samples_weight = weight[train_labels]
        samples_weight = torch.from_numpy(samples_weight).float()
        # Create sampler
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    else:
        sampler = None
    
    # Define batch size
    batch_size = 16
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,  # Use weighted sampler if available, otherwise shuffle
        shuffle=sampler is None,  # Only shuffle if not using sampler
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nBatch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model
    model = get_model(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Train the model
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=30  # Increased from 20 to 30 for better convergence
    )
    
    # Save the final model
    torch.save(model.state_dict(), '../models/final_model.pth')
    print("Final model saved successfully!")
    
    # Save history as CSV for later analysis
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_kappa': history['train_kappa'],
        'val_kappa': history['val_kappa']
    })
    history_df.to_csv('../models/training_history.csv', index=False)
    print("Training history saved to CSV.")

if __name__ == "__main__":
    main()