import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from model import get_model

def visualize_predictions(model, img_dir, submission_df, num_samples=20, save_path='../results/prediction_visualization.png'):
    """Visualize model predictions on test images."""
    # Class names
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    # Sample from each predicted class
    samples = []
    for class_idx in range(5):
        class_samples = submission_df[submission_df['diagnosis'] == class_idx].sample(
            min(num_samples // 5, sum(submission_df['diagnosis'] == class_idx))
        )
        samples.append(class_samples)
    
    # Combine samples
    sampled_df = pd.concat(samples).reset_index(drop=True)
    
    # Create figure
    rows = len(sampled_df)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3*rows))
    
    # For a single row case
    if rows == 1:
        axes = [axes]
    
    # Plot each sample
    for i, (_, row) in enumerate(sampled_df.iterrows()):
        img_id = row['id_code']
        pred_class = row['diagnosis']
        
        img_path = os.path.join(img_dir, img_id + '.png')
        
        # Load and process image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Original image
        axes[i][0].imshow(img)
        axes[i][0].set_title(f"Prediction: {class_names[pred_class]}")
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
    
    # Save figure
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    
    plt.show()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data paths
    data_dir = "../datasets/aptos2019-blindness-detection"
    processed_dir = os.path.join(data_dir, "processed_test_images")
    submission_path = "../results/submission.csv"
    
    # Ensure results directory exists
    os.makedirs('../results', exist_ok=True)
    
    # Load submission
    if not os.path.exists(submission_path):
        print("Submission file not found. Please run prediction script first.")
        return
    
    submission_df = pd.read_csv(submission_path)
    
    # Visualize predictions
    visualize_predictions(None, processed_dir, submission_df)

if __name__ == "__main__":
    main()