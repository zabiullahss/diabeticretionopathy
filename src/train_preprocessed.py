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
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import argparse
import logging
import seaborn as sns
import json
from datetime import datetime

from model import get_model

def setup_logger(model_type, model_version):
    """Set up and return a logger for the specified model training."""
    # Create logs directory if it doesn't exist
    os.makedirs("../logs", exist_ok=True)
    
    logger = logging.getLogger(f"{model_type.lower()}_{model_version}_training")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(f"../logs/{model_type.lower()}_{model_version}_training.log")
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=25, model_dir='../models', model_type='EfficientNet', model_version='b4', logger=None):
    """
    Train the model and save the best weights based on validation kappa score.
    
    Parameters:
    model: The model to train
    train_loader: DataLoader for training data
    val_loader: DataLoader for validation data
    criterion: Loss function
    optimizer: Optimizer
    scheduler: Learning rate scheduler
    device: Device to train on (cuda/cpu)
    num_epochs: Number of epochs to train for
    model_dir: Directory to save model checkpoints
    model_type: Type of model (EfficientNet, ResNet, DenseNet)
    model_version: Version of the model
    logger: Logger object
    
    Returns:
    model: The trained model
    history: Training history dictionary
    """
    since = time.time()
    best_model_wts = model.state_dict()
    best_kappa = 0.0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_kappa': [],
        'val_kappa': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'learning_rates': []
    }
    
    # Create model directory path
    model_dir = os.path.join(model_dir, model_type, model_version)
    os.makedirs(model_dir, exist_ok=True)
    
    logger.info(f"Starting training of {model_type}-{model_version} for {num_epochs} epochs")
    logger.info(f"Model checkpoints will be saved to {model_dir}")
    
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        logger.info('-' * 30)
        
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
            train_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate average epoch loss
        train_loss = train_loss / len(train_loader.dataset)
        epoch_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        epoch_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        history['train_loss'].append(train_loss)
        history['train_kappa'].append(epoch_kappa)
        history['train_accuracy'].append(epoch_accuracy)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        logger.info(f'Train Loss: {train_loss:.4f} Kappa: {epoch_kappa:.4f} Acc: {epoch_accuracy:.4f}')
        
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
                
                # Accumulate the loss
                val_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate average epoch loss
        val_loss = val_loss / len(val_loader.dataset)
        epoch_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        epoch_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        history['val_loss'].append(val_loss)
        history['val_kappa'].append(epoch_kappa)
        history['val_accuracy'].append(epoch_accuracy)
        
        logger.info(f'Val Loss: {val_loss:.4f} Kappa: {epoch_kappa:.4f} Acc: {epoch_accuracy:.4f}')
        
        # Deep copy the model if best performance
        if epoch_kappa > best_kappa:
            best_kappa = epoch_kappa
            best_model_wts = model.state_dict()
            # Save the best model
            torch.save(best_model_wts, os.path.join(model_dir, 'best_model.pth'))
            logger.info(f"Saved new best model with Kappa: {epoch_kappa:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_kappa': epoch_kappa,
                'val_kappa': epoch_kappa,
                'history': history
            }, checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch+1}")
        
        # Step the scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_kappa)
            else:
                scheduler.step()
        
        logger.info("")
    
    time_elapsed = time.time() - since
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best Val Kappa: {best_kappa:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pth'))
    logger.info(f"Final model saved to {os.path.join(model_dir, 'final_model.pth')}")
    
    # Plot training history
    plot_training_history(history, model_dir, model_type, model_version)
    
    # Save training history as JSON
    with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, cls=NumpyEncoder)
    
    # Generate final evaluation report
    generate_evaluation_report(model, val_loader, device, model_dir, model_type, model_version, logger)
    
    return model, history

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                           np.int16, np.int32, np.int64, np.uint8,
                           np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.void):
            return None
        return json.JSONEncoder.default(self, obj)

def plot_training_history(history, model_dir, model_type, model_version):
    """
    Plot training history metrics.
    
    Parameters:
    history: Dictionary containing training history
    model_dir: Directory to save plots
    model_type: Type of model
    model_version: Version of the model
    """
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f'{model_type}-{model_version} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Kappa
    plt.subplot(2, 2, 2)
    plt.plot(history['train_kappa'], label='Train')
    plt.plot(history['val_kappa'], label='Validation')
    plt.title(f'{model_type}-{model_version} Quadratic Weighted Kappa')
    plt.xlabel('Epoch')
    plt.ylabel('Kappa')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(2, 2, 3)
    plt.plot(history['train_accuracy'], label='Train')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.title(f'{model_type}-{model_version} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot learning rate
    plt.subplot(2, 2, 4)
    plt.plot(history['learning_rates'])
    plt.title(f'{model_type}-{model_version} Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    plt.close()

def generate_evaluation_report(model, val_loader, device, model_dir, model_type, model_version, logger):
    """
    Generate a comprehensive evaluation report for the model.
    
    Parameters:
    model: Trained model
    val_loader: Validation data loader
    device: Device to run evaluation on
    model_dir: Directory to save report
    model_type: Type of model
    model_version: Version of the model
    logger: Logger object
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    logger.info("Generating final evaluation report...")
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Log metrics
    logger.info(f"Final model evaluation:")
    logger.info(f"Quadratic Weighted Kappa: {kappa:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'{model_type}-{model_version} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Save metrics to JSON
    evaluation = {
        'model_type': model_type,
        'model_version': model_version,
        'kappa': float(kappa),
        'accuracy': float(accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(model_dir, 'evaluation.json'), 'w') as f:
        json.dump(evaluation, f, cls=NumpyEncoder)
    
    # Create a text report
    with open(os.path.join(model_dir, 'evaluation_report.txt'), 'w') as f:
        f.write(f"{model_type}-{model_version} Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"Quadratic Weighted Kappa: {kappa:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names))
        
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    
    logger.info(f"Evaluation report saved to {model_dir}")
    
    return evaluation

def main():
    """Main function to train the diabetic retinopathy detection model."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a model for diabetic retinopathy detection.")
    parser.add_argument('--model_type', type=str, default='EfficientNet', choices=['EfficientNet', 'ResNet', 'DenseNet'],
                        help='Type of model to train (EfficientNet, ResNet, DenseNet)')
    parser.add_argument('--model_version', type=str, default=None,
                        help='Version of the model (e.g., b4 for EfficientNet, 50 for ResNet, 121 for DenseNet)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for regularization')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained model weights')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set default model versions if none provided
    if args.model_version is None:
        if args.model_type == 'EfficientNet':
            args.model_version = 'b4'
        elif args.model_type == 'ResNet':
            args.model_version = '50'
        elif args.model_type == 'DenseNet':
            args.model_version = '121'
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up logging
    logger = setup_logger(args.model_type, args.model_version)
    logger.info(f"Starting training script for {args.model_type}-{args.model_version}")
    logger.info(f"Using device: {device}")
    logger.info(f"Training parameters: {vars(args)}")
    
    # Data paths
    data_dir = "../datasets/aptos2019-blindness-detection"
    processed_dir = os.path.join(data_dir, "processed_train_images")
    train_split_csv = os.path.join(data_dir, "train_split.csv")
    val_split_csv = os.path.join(data_dir, "val_split.csv")
    
    # Check if preprocessed data exists
    if not os.path.exists(processed_dir) or not os.path.exists(train_split_csv) or not os.path.exists(val_split_csv):
        logger.error("Preprocessed data not found. Please run data_prep.py first.")
        return
    
    # Create model directory
    model_dir = os.path.join("../models", args.model_type, args.model_version)
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Model will be saved to {model_dir}")
    
    # Save run configuration
    with open(os.path.join(model_dir, 'training_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Load training and validation data
    train_df = pd.read_csv(train_split_csv)
    val_df = pd.read_csv(val_split_csv)
    
    logger.info(f"Training set: {len(train_df)} images")
    logger.info(f"Validation set: {len(val_df)} images")
    
    # Log class distribution
    train_class_dist = train_df['diagnosis'].value_counts().sort_index()
    val_class_dist = val_df['diagnosis'].value_counts().sort_index()
    logger.info("\nClass distribution:")
    for i in range(5):
        train_count = train_class_dist.get(i, 0)
        val_count = val_class_dist.get(i, 0)
        train_pct = (train_count / len(train_df)) * 100 if len(train_df) > 0 else 0
        val_pct = (val_count / len(val_df)) * 100 if len(val_df) > 0 else 0
        logger.info(f"Class {i}: Train={train_count} ({train_pct:.2f}%), Val={val_count} ({val_pct:.2f}%)")
    
    # Create datasets
    train_transform = get_transforms(is_training=True)
    train_dataset = PreprocessedRetinopathyDataset(train_df, processed_dir, transform=train_transform)
    val_dataset = PreprocessedRetinopathyDataset(val_df, processed_dir, transform=None)
    
    # Load class weights
    class_weights_path = os.path.join(data_dir, "info", "class_weights.npy")
    if os.path.exists(class_weights_path):
        class_weights_dict = np.load(class_weights_path, allow_pickle=True).item()
        logger.info("Loaded class weights: " + str(class_weights_dict))
        class_weights = [class_weights_dict[i] for i in range(5)]
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    else:
        logger.warning("Class weights file not found. Using equal weights.")
        class_weights_tensor = None
    
    # Create samplers for handling class imbalance
    if os.path.exists(class_weights_path):
        train_labels = train_df['diagnosis'].values
        class_counts = np.bincount(train_labels)
        weight = 1. / class_counts
        samples_weight = weight[train_labels]
        samples_weight = torch.from_numpy(samples_weight).float()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        logger.info("Using weighted sampler to address class imbalance")
    else:
        sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders with batch size {args.batch_size}")
    logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Initialize model
    model = get_model(
        device=device,
        model_type=args.model_type,
        model_version=args.model_version,
        num_classes=5,
        pretrained=args.pretrained
    )
    
    # Set up loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    logger.info(f"Using CrossEntropyLoss with class weights: {class_weights_tensor}")
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    logger.info(f"Using Adam optimizer with learning rate {args.learning_rate} and weight decay {args.weight_decay}")
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    logger.info("Using ReduceLROnPlateau scheduler")
    
    # Train the model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        model_dir=model_dir,
        model_type=args.model_type,
        model_version=args.model_version,
        logger=logger
    )
    
    logger.info("Training completed!")

    # Create directory if it doesn't exist
    os.makedirs(os.path.join("../results", args.model_type, args.model_version), exist_ok=True)

    
    # Save final results to a results.txt file for easy reference
    with open(os.path.join("../results", args.model_type, args.model_version, "results.txt"), 'w') as f:
        f.write(f"{args.model_type}-{args.model_version} Training Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Model Version: {args.model_version}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Epochs: {args.num_epochs}\n")
        f.write(f"Initial Learning Rate: {args.learning_rate}\n\n")
        f.write("Final Results:\n")
        f.write(f"Best Validation Kappa: {max(history['val_kappa']):.4f}\n")
        f.write(f"Best Validation Accuracy: {max(history['val_accuracy']):.4f}\n")
        f.write(f"Final Training Kappa: {history['train_kappa'][-1]:.4f}\n")
        f.write(f"Final Validation Kappa: {history['val_kappa'][-1]:.4f}\n")
    
    logger.info(f"Results summary saved to ../results/{args.model_type}/{args.model_version}/results.txt")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../logs", exist_ok=True)
    os.makedirs("../results", exist_ok=True)
    
    for model_type in ['EfficientNet', 'ResNet', 'DenseNet']:
        os.makedirs(f"../models/{model_type}", exist_ok=True)
        os.makedirs(f"../results/{model_type}", exist_ok=True)
    
    main()