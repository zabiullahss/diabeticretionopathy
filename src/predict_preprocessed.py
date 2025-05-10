import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from model import get_model

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image, img_id

def predict(model, test_loader, device):
    """Make predictions on test set."""
    model.eval()
    predictions = []
    img_ids = []
    probabilities = []
    
    with torch.no_grad():
        for inputs, ids in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            img_ids.extend(ids)
    
    return img_ids, predictions, probabilities

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
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
    
    plt.show()

def plot_prediction_samples(img_dir, img_ids, true_labels, predictions, num_samples=5):
    """Plot sample predictions."""
    # Create a figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
    
    # Class names
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    # Get random samples with focus on mistakes
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
    
    # Plot samples
    for i, idx in enumerate(samples):
        img_path = os.path.join(img_dir, img_ids[idx] + '.png')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Original image
        axes[i, 0].imshow(img)
        true_class = class_names[true_labels[idx]]
        pred_class = class_names[predictions[idx]]
        title = f"True: {true_class}, Pred: {pred_class}"
        
        if true_labels[idx] == predictions[idx]:
            color = 'green'
        else:
            color = 'red'
            
        axes[i, 0].set_title(title, color=color)
        axes[i, 0].axis('off')
        
        # Highlight features
        # This is a simplified version - real feature highlighting would require more sophisticated techniques
        # Apply CLAHE for better visibility
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        axes[i, 1].imshow(enhanced)
        axes[i, 1].set_title(f"Enhanced", color=color)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('../results/prediction_samples.png')
    plt.show()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data paths
    data_dir = "../datasets/aptos2019-blindness-detection"
    processed_dir = os.path.join(data_dir, "processed_test_images")
    test_csv = os.path.join(data_dir, "test.csv")
    model_path = "../models/best_model.pth"
    
    # Create results directory
    os.makedirs('../results', exist_ok=True)
    
    # Load test data
    test_df = pd.read_csv(test_csv)
    test_imgs = test_df['id_code'].values
    
    print(f"Test set: {len(test_imgs)} images")
    
    # Create test dataset and dataloader
    test_dataset = PreprocessedTestDataset(processed_dir, test_imgs)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    model = get_model(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    
    # Make predictions
    img_ids, predictions, probabilities = predict(model, test_loader, device)
    
    # Create submission file
    submission = pd.DataFrame({
        'id_code': img_ids,
        'diagnosis': predictions
    })
    
    # Save submission
    submission_path = '../results/submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Prediction complete! Submission file saved to {submission_path}")
    
    # Print class distribution in predictions
    print("\nPredicted class distribution:")
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    for i, count in enumerate(pred_counts):
        print(f"Class {i}: {count} images ({count/len(predictions)*100:.2f}%)")
    
    # If test labels are available (only for validation purposes)
    if 'diagnosis' in test_df.columns:
        true_labels = test_df['diagnosis'].values
        
        # Plot confusion matrix
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        plot_confusion_matrix(true_labels, predictions, class_names, '../results/confusion_matrix.png')
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, target_names=class_names))
        
        # Plot sample predictions
        plot_prediction_samples(processed_dir, img_ids, true_labels, predictions)
    
if __name__ == "__main__":
    main()