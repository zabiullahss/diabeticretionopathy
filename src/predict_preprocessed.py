import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import seaborn as sns
import argparse
import logging
import json
from datetime import datetime

from model import get_model, setup_logger

class PreprocessedTestDataset(Dataset):
    def __init__(self, img_dir, img_list):
        self.img_dir = img_dir
        self.img_list = img_list
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_id = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_id + '.png')
        
        # Load preprocessed image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image, img_id

def predict(model, test_loader, device, logger):
    """Make predictions on test set."""
    model.eval()
    predictions = []
    img_ids = []
    probabilities = []
    
    logger.info("Starting inference on test dataset...")
    
    with torch.no_grad():
        for inputs, ids in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            img_ids.extend(ids)
    
    logger.info(f"Inference completed on {len(predictions)} images")
    
    return img_ids, predictions, probabilities

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, logger=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        if logger:
            logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.close()

def plot_prediction_samples(img_dir, img_ids, predictions, true_labels=None, num_samples=5, save_path=None, logger=None):
    """
    Plot sample predictions.
    
    Parameters:
    img_dir (str): Directory containing test images
    img_ids (list): List of image IDs
    predictions (list): List of predicted classes
    true_labels (list, optional): List of true labels (if available)
    num_samples (int): Number of samples to plot
    save_path (str): Path to save the figure
    logger (logging.Logger): Logger object
    """
    # Create a figure
    if true_labels is not None:
        # If true labels available, focus on mistakes
        samples = []
        for idx in range(len(img_ids)):
            if true_labels[idx] != predictions[idx]:
                samples.append(idx)
                if len(samples) >= num_samples:
                    break
        
        # If not enough mistakes, add correct predictions
        if len(samples) < num_samples:
            correct_predictions = [idx for idx in range(len(img_ids)) 
                                if true_labels[idx] == predictions[idx]]
            samples.extend(correct_predictions[:num_samples-len(samples)])
    else:
        # If no true labels, randomly sample
        samples = np.random.choice(range(len(img_ids)), min(num_samples, len(img_ids)), replace=False)
    
    # Class names
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    # Create figure
    fig, axes = plt.subplots(len(samples), 2, figsize=(12, 3*len(samples)))
    
    # For a single row case
    if len(samples) == 1:
        axes = [axes]
    
    # Plot samples
    for i, idx in enumerate(samples):
        img_path = os.path.join(img_dir, img_ids[idx] + '.png')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Original image
        axes[i][0].imshow(img)
        
        if true_labels is not None:
            true_class = class_names[true_labels[idx]]
            pred_class = class_names[predictions[idx]]
            title = f"True: {true_class}, Pred: {pred_class}"
            
            if true_labels[idx] == predictions[idx]:
                color = 'green'
            else:
                color = 'red'
                
            axes[i][0].set_title(title, color=color)
        else:
            pred_class = class_names[predictions[idx]]
            axes[i][0].set_title(f"Prediction: {pred_class}")
            
        axes[i][0].axis('off')
        
        # Apply CLAHE for better feature visibility
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        axes[i][1].imshow(enhanced)
        axes[i][1].set_title("Enhanced")
        axes[i][1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        if logger:
            logger.info(f"Prediction samples saved to {save_path}")
    
    plt.close()

def main():
    """Main function to make predictions using a trained model."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Make predictions using a trained model.")
    parser.add_argument('--model_type', type=str, default='EfficientNet', choices=['EfficientNet', 'ResNet', 'DenseNet'],
                        help='Type of model to use (EfficientNet, ResNet, DenseNet)')
    parser.add_argument('--model_version', type=str, default=None,
                        help='Version of the model (e.g., b4 for EfficientNet, 50 for ResNet, 121 for DenseNet)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    
    args = parser.parse_args()
    
    # Set default model versions if none provided
    if args.model_version is None:
        if args.model_type == 'EfficientNet':
            args.model_version = 'b4'
        elif args.model_type == 'ResNet':
            args.model_version = '50'
        elif args.model_type == 'DenseNet':
            args.model_version = '121'
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up logging
    logger = setup_logger(f"{args.model_type.lower()}_{args.model_version}", "predict")
    logger.info(f"Starting prediction script for {args.model_type}-{args.model_version}")
    logger.info(f"Using device: {device}")
    
    # Data paths
    data_dir = "../datasets/aptos2019-blindness-detection"
    processed_test_dir = os.path.join(data_dir, "processed_test_images")
    test_csv = os.path.join(data_dir, "test.csv")
    
    # Model and results paths
    model_path = os.path.join("../models", args.model_type, args.model_version, "best_model.pth")
    results_dir = os.path.join("../results", args.model_type, args.model_version)
    
    # Create directories if they don't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Please train the model first.")
        return
    
    # Load test data
    if not os.path.exists(test_csv):
        logger.error(f"Test CSV not found at {test_csv}.")
        return
    
    test_df = pd.read_csv(test_csv)
    test_imgs = test_df['id_code'].values
    
    logger.info(f"Test set: {len(test_imgs)} images")
    
    # Create test dataset and dataloader
    try:
        test_dataset = PreprocessedTestDataset(processed_test_dir, test_imgs)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True
        )
        logger.info(f"Created test dataloader with batch size {args.batch_size}")
    except Exception as e:
        logger.error(f"Error creating test dataloader: {e}")
        return
    
    # Load model
    try:
        model = get_model(
            device=device,
            model_type=args.model_type,
            model_version=args.model_version,
            num_classes=5,
            pretrained=False
        )
        
        model.load_state_dict(torch.load(model_path))
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Make predictions
    img_ids, predictions, probabilities = predict(model, test_loader, device, logger)
    
    # Create submission file
    submission = pd.DataFrame({
        'id_code': img_ids,
        'diagnosis': predictions
    })
    
    # Save submission
    submission_path = os.path.join(results_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    logger.info(f"Prediction complete! Submission file saved to {submission_path}")
    
    # Save prediction details with probabilities
    detailed_predictions = pd.DataFrame({
        'id_code': img_ids,
        'diagnosis': predictions,
        'prob_class0': [prob[0] for prob in probabilities],
        'prob_class1': [prob[1] for prob in probabilities],
        'prob_class2': [prob[2] for prob in probabilities],
        'prob_class3': [prob[3] for prob in probabilities],
        'prob_class4': [prob[4] for prob in probabilities]
    })
    
    detailed_predictions_path = os.path.join(results_dir, 'detailed_predictions.csv')
    detailed_predictions.to_csv(detailed_predictions_path, index=False)
    logger.info(f"Detailed predictions saved to {detailed_predictions_path}")
    
    # Print class distribution in predictions
    logger.info("\nPredicted class distribution:")
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    for i in range(5):
        count = pred_counts.get(i, 0)
        logger.info(f"Class {i}: {count} images ({count/len(predictions)*100:.2f}%)")
    
    # Visualize sample predictions
    try:
        plot_prediction_samples(
            processed_test_dir,
            img_ids,
            predictions,
            true_labels=None,
            num_samples=8,
            save_path=os.path.join(results_dir, 'prediction_samples.png'),
            logger=logger
        )
    except Exception as e:
        logger.error(f"Error generating prediction sample visualization: {e}")
    
    # If test labels are available (for validation purposes)
    if 'diagnosis' in test_df.columns:
        logger.info("Ground truth labels found in test set. Generating evaluation metrics...")
        true_labels = test_df['diagnosis'].values
        
        # Calculate evaluation metrics
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        kappa = cohen_kappa_score(true_labels, predictions, weights='quadratic')
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        
        logger.info(f"Quadratic Weighted Kappa: {kappa:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        # Print classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(true_labels, predictions, target_names=class_names))
        
        # Plot confusion matrix
        try:
            plot_confusion_matrix(
                true_labels,
                predictions,
                class_names,
                save_path=os.path.join(results_dir, 'confusion_matrix.png'),
                logger=logger
            )
        except Exception as e:
            logger.error(f"Error generating confusion matrix: {e}")
        
        # Plot sample predictions with ground truth
        try:
            plot_prediction_samples(
                processed_test_dir,
                img_ids,
                predictions,
                true_labels=true_labels,
                num_samples=10,
                save_path=os.path.join(results_dir, 'prediction_with_ground_truth.png'),
                logger=logger
            )
        except Exception as e:
            logger.error(f"Error generating prediction samples with ground truth: {e}")
        
        # Save evaluation metrics to JSON
        evaluation = {
            'model_type': args.model_type,
            'model_version': args.model_version,
            'kappa': float(kappa),
            'accuracy': float(accuracy),
            'prediction_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(results_dir, 'test_evaluation.json'), 'w') as f:
            json.dump(evaluation, f, indent=4)
    
    logger.info("Prediction process completed successfully.")
    
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("../logs", exist_ok=True)
    
    for model_type in ['EfficientNet', 'ResNet', 'DenseNet']:
        os.makedirs(f"../results/{model_type}", exist_ok=True)
    
    main()