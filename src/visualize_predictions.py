import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from model import get_model, setup_logger
import argparse
import logging
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.gridspec import GridSpec

def setup_visualization_logger():
    """Set up and return a logger for the visualization script."""
    # Create logs directory if it doesn't exist
    os.makedirs("../logs", exist_ok=True)
    
    logger = logging.getLogger("visualization")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler("../logs/visualization.log")
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

def visualize_predictions(model_type, model_version, img_dir, submission_df, num_samples=20, 
                          save_path=None, true_labels_df=None, logger=None):
    """
    Visualize model predictions on test images.
    
    Parameters:
    model_type (str): Type of model used
    model_version (str): Version of the model used
    img_dir (str): Directory containing preprocessed test images
    submission_df (pd.DataFrame): DataFrame containing predictions
    num_samples (int): Number of samples to visualize (per class if possible)
    save_path (str): Path to save the visualization
    true_labels_df (pd.DataFrame): DataFrame containing ground truth labels (if available)
    logger (logging.Logger): Logger object
    """
    if logger:
        logger.info(f"Visualizing predictions for {model_type}-{model_version}")
        logger.info(f"Directory: {img_dir}")
        logger.info(f"Number of predictions: {len(submission_df)}")
    
    # Class names
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    # Create output directory if needed
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Check if ground truth is available
    has_ground_truth = (true_labels_df is not None)
    
    # If true labels available, merge with predictions
    if has_ground_truth:
        merged_df = pd.merge(
            submission_df, 
            true_labels_df, 
            on='id_code', 
            suffixes=('_pred', '')
        )
        if logger:
            logger.info(f"Ground truth available. Merged DataFrame size: {len(merged_df)}")
    else:
        merged_df = submission_df.rename(columns={'diagnosis': 'diagnosis_pred'})
    
    # Create confusion matrix if ground truth available
    if has_ground_truth:
        cm = confusion_matrix(merged_df['diagnosis'], merged_df['diagnosis_pred'], labels=range(5))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'{model_type}-{model_version} Confusion Matrix')
        plt.tight_layout()
        
        # Save confusion matrix if save_path provided
        if save_path:
            confusion_path = os.path.join(os.path.dirname(save_path), f'{model_type}_{model_version}_confusion_matrix.png')
            plt.savefig(confusion_path)
            if logger:
                logger.info(f"Confusion matrix saved to {confusion_path}")
        plt.close()
    
    # Sample from each predicted class
    samples = []
    for class_idx in range(5):
        class_df = merged_df[merged_df['diagnosis_pred'] == class_idx]
        if len(class_df) == 0:
            continue
            
        # If ground truth available, include both correct and incorrect predictions
        if has_ground_truth:
            # Get some correct predictions
            correct_df = class_df[class_df['diagnosis'] == class_idx]
            correct_samples = min(num_samples // 2, len(correct_df))
            if correct_samples > 0:
                samples.append(correct_df.sample(correct_samples))
            
            # Get some incorrect predictions
            incorrect_df = class_df[class_df['diagnosis'] != class_idx]
            incorrect_samples = min(num_samples // 2, len(incorrect_df))
            if incorrect_samples > 0:
                samples.append(incorrect_df.sample(incorrect_samples))
        else:
            # Without ground truth, just sample from predicted class
            class_samples = min(num_samples // 5, len(class_df))
            if class_samples > 0:
                samples.append(class_df.sample(class_samples))
    
    # Combine sampled rows
    sampled_df = pd.concat(samples).reset_index(drop=True)
    if logger:
        logger.info(f"Selected {len(sampled_df)} samples for visualization")
    
    # Calculate grid dimensions
    num_rows = min(len(sampled_df), 20)  # Limit to 20 rows for readability
    
    # Create figure
    fig = plt.figure(figsize=(15, num_rows * 2.5))
    gs = GridSpec(num_rows, 3, figure=fig, width_ratios=[1, 1, 0.5])
    
    # Plot each sample
    for i, (_, row) in enumerate(sampled_df.iloc[:num_rows].iterrows()):
        if i >= num_rows:
            break
            
        img_id = row['id_code']
        pred_class = row['diagnosis_pred']
        
        img_path = os.path.join(img_dir, img_id + '.png')
        
        # Load and process image
        try:
            img = cv2.imread(img_path)
            if img is None:
                if logger:
                    logger.warning(f"Could not read image: {img_path}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            if logger:
                logger.error(f"Error loading image {img_path}: {e}")
            continue
        
        # Original image
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(img)
        
        title = f"Pred: {class_names[pred_class]}"
        if has_ground_truth:
            true_class = row['diagnosis']
            title = f"True: {class_names[true_class]} | {title}"
            
            # Color code based on correctness
            if true_class == pred_class:
                title_color = 'green'
            else:
                title_color = 'red'
        else:
            title_color = 'black'
            
        ax1.set_title(title, color=title_color)
        ax1.axis('off')
        
        # Apply CLAHE for better feature visibility
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(enhanced)
        ax2.set_title("Enhanced")
        ax2.axis('off')
        
        # Class probabilities if available
        if 'prob_class0' in row.index:
            ax3 = fig.add_subplot(gs[i, 2])
            probs = [row[f'prob_class{j}'] for j in range(5)]
            ax3.barh(range(5), probs, color='skyblue')
            ax3.set_yticks(range(5))
            ax3.set_yticklabels(class_names)
            ax3.set_xlim(0, 1)
            ax3.set_title("Class Probabilities")
            
            # Highlight the predicted class
            ax3.get_children()[pred_class].set_color('navy')
            
            # Highlight the true class if available
            if has_ground_truth and true_class != pred_class:
                # Add a red edge to the true class bar
                ax3.get_children()[true_class].set_edgecolor('red')
                ax3.get_children()[true_class].set_linewidth(2)
    
    plt.suptitle(f"{model_type}-{model_version} Prediction Visualization", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    
    # Save visualization
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if logger:
            logger.info(f"Visualization saved to {save_path}")
    
    plt.close()
    
    return sampled_df

def visualize_model_comparison(model_types, model_versions, img_dir, num_samples=10, save_path=None, logger=None):
    """
    Create a side-by-side comparison of predictions from different models.
    
    Parameters:
    model_types (list): List of model types
    model_versions (list): List of model versions corresponding to model_types
    img_dir (str): Directory containing preprocessed test images
    num_samples (int): Number of samples to visualize
    save_path (str): Path to save the visualization
    logger (logging.Logger): Logger object
    """
    if logger:
        logger.info(f"Creating model comparison visualization")
        logger.info(f"Models: {list(zip(model_types, model_versions))}")
    
    # Load prediction files for each model
    predictions_dfs = []
    for model_type, model_version in zip(model_types, model_versions):
        submission_path = os.path.join("../results", model_type, model_version, "detailed_predictions.csv")
        
        if not os.path.exists(submission_path):
            if logger:
                logger.warning(f"Predictions not found for {model_type}-{model_version}: {submission_path}")
            continue
            
        df = pd.read_csv(submission_path)
        df['model'] = f"{model_type}-{model_version}"
        predictions_dfs.append(df)
    
    if not predictions_dfs:
        if logger:
            logger.error("No prediction files found. Cannot create comparison.")
        return
    
    # Merge all predictions
    base_df = predictions_dfs[0][['id_code', 'diagnosis']]
    base_df = base_df.rename(columns={'diagnosis': 'diagnosis_base'})
    
    all_df = base_df.copy()
    for idx, df in enumerate(predictions_dfs):
        model_name = df['model'].iloc[0]
        model_df = df[['id_code', 'diagnosis']].rename(columns={'diagnosis': f'diagnosis_{idx}'})
        all_df = pd.merge(all_df, model_df, on='id_code')
    
    # Find images with disagreement between models
    disagreement_mask = False
    for i in range(len(predictions_dfs)):
        disagreement_mask |= (all_df['diagnosis_base'] != all_df[f'diagnosis_{i}'])
    
    disagreement_df = all_df[disagreement_mask]
    
    if logger:
        logger.info(f"Found {len(disagreement_df)} images with disagreement between models")
    
    # If no disagreements, sample random images
    if len(disagreement_df) == 0:
        sampled_ids = np.random.choice(all_df['id_code'].values, min(num_samples, len(all_df)), replace=False)
    else:
        # Sample from disagreement set
        sampled_ids = disagreement_df.sample(min(num_samples, len(disagreement_df)))['id_code'].values
    
    # Create figure
    num_models = len(predictions_dfs)
    fig, axes = plt.subplots(len(sampled_ids), num_models, figsize=(4*num_models, 3*len(sampled_ids)))
    
    # Handle case of single sample
    if len(sampled_ids) == 1:
        axes = np.array([axes])
    
    # Class names
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    # Plot each sample
    for i, img_id in enumerate(sampled_ids):
        img_path = os.path.join(img_dir, img_id + '.png')
        
        # Load image
        try:
            img = cv2.imread(img_path)
            if img is None:
                if logger:
                    logger.warning(f"Could not read image: {img_path}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply CLAHE for better visibility
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[..., 0] = clahe.apply(lab[..., 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        except Exception as e:
            if logger:
                logger.error(f"Error loading image {img_path}: {e}")
            continue
        
        # Show prediction from each model
        for j, df in enumerate(predictions_dfs):
            model_df = df[df['id_code'] == img_id]
            
            if len(model_df) == 0:
                continue
                
            pred_class = model_df['diagnosis'].iloc[0]
            
            axes[i, j].imshow(enhanced)
            axes[i, j].set_title(f"{model_df['model'].iloc[0]}\nPred: {class_names[pred_class]}")
            axes[i, j].axis('off')
            
            # If there's disagreement between models, add a colored border
            if j > 0 and pred_class != predictions_dfs[0][predictions_dfs[0]['id_code'] == img_id]['diagnosis'].iloc[0]:
                for spine in axes[i, j].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
    
    plt.suptitle(f"Model Comparison Visualization", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    
    # Save visualization
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if logger:
            logger.info(f"Comparison visualization saved to {save_path}")
    
    plt.close()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize diabetic retinopathy predictions.")
    parser.add_argument('--model_type', type=str, default='EfficientNet', choices=['EfficientNet', 'ResNet', 'DenseNet', 'all'],
                        help='Type of model to visualize (EfficientNet, ResNet, DenseNet, or all)')
    parser.add_argument('--model_version', type=str, default=None,
                        help='Version of the model (e.g., b4 for EfficientNet, 50 for ResNet, 121 for DenseNet)')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to visualize')
    parser.add_argument('--comparison', action='store_true', help='Create model comparison visualization')
    
    args = parser.parse_args()
    
    # Set default model versions if none provided
    if args.model_version is None:
        if args.model_type == 'EfficientNet':
            args.model_version = 'b4'
        elif args.model_type == 'ResNet':
            args.model_version = '50'
        elif args.model_type == 'DenseNet':
            args.model_version = '121'
    
    # Set up logging
    logger = setup_visualization_logger()
    logger.info(f"Starting visualization script with args: {vars(args)}")
    
    # Data paths
    data_dir = "../datasets/aptos2019-blindness-detection"
    processed_test_dir = os.path.join(data_dir, "processed_test_images")
    test_csv = os.path.join(data_dir, "test.csv")
    
    # Check if test data exists
    if not os.path.exists(processed_test_dir) or not os.path.exists(test_csv):
        logger.error(f"Test data not found. Please run data_prep.py first.")
        return
    
    # Load test labels if available (for ground truth comparison)
    test_df = pd.read_csv(test_csv)
    true_labels_df = test_df if 'diagnosis' in test_df.columns else None
    
    if true_labels_df is not None:
        logger.info("Ground truth labels found in test set and will be used for visualization.")
    else:
        logger.info("No ground truth labels found in test set.")
    
    # Create results directory
    os.makedirs("../results", exist_ok=True)
    
    # If comparison mode, visualize all models side-by-side
    if args.comparison:
        logger.info("Creating model comparison visualization")
        
        # Define models to compare
        model_types = ['EfficientNet', 'ResNet', 'DenseNet']
        model_versions = ['b4', '50', '121']
        
        # Check which models have predictions available
        available_models = []
        available_versions = []
        
        for model_type, model_version in zip(model_types, model_versions):
            submission_path = os.path.join("../results", model_type, model_version, "submission.csv")
            if os.path.exists(submission_path):
                available_models.append(model_type)
                available_versions.append(model_version)
        
        if not available_models:
            logger.error("No prediction files found for any model. Please run predictions first.")
            return
        
        logger.info(f"Found predictions for: {list(zip(available_models, available_versions))}")
        
        # Create comparison visualization
        visualize_model_comparison(
            available_models,
            available_versions, 
            processed_test_dir, 
            num_samples=args.num_samples,
            save_path="../results/model_comparison.png",
            logger=logger
        )
        
        logger.info("Model comparison visualization complete.")
        return
    
    # Visualize single model or all models individually
    if args.model_type == 'all':
        # Visualize all models with available predictions
        model_types = ['EfficientNet', 'ResNet', 'DenseNet']
        model_versions = ['b4', '50', '121']
        
        for model_type, model_version in zip(model_types, model_versions):
            submission_path = os.path.join("../results", model_type, model_version, "detailed_predictions.csv")
            
            if not os.path.exists(submission_path):
                logger.warning(f"Predictions not found for {model_type}-{model_version}: {submission_path}")
                continue
                
            logger.info(f"Visualizing predictions for {model_type}-{model_version}")
            
            # Load predictions
            submission_df = pd.read_csv(submission_path)
            
            # Create visualization
            visualize_predictions(
                model_type,
                model_version,
                processed_test_dir,
                submission_df,
                num_samples=args.num_samples,
                save_path=os.path.join("../results", model_type, model_version, "prediction_visualization.png"),
                true_labels_df=true_labels_df,
                logger=logger
            )
    else:
        # Visualize specific model
        submission_path = os.path.join("../results", args.model_type, args.model_version, "detailed_predictions.csv")
        
        if not os.path.exists(submission_path):
            logger.error(f"Predictions not found for {args.model_type}-{args.model_version}: {submission_path}")
            return
        
        # Load predictions
        submission_df = pd.read_csv(submission_path)
        
        # Create visualization
        visualize_predictions(
            args.model_type,
            args.model_version,
            processed_test_dir,
            submission_df,
            num_samples=args.num_samples,
            save_path=os.path.join("../results", args.model_type, args.model_version, "prediction_visualization.png"),
            true_labels_df=true_labels_df,
            logger=logger
        )
    
    logger.info("Visualization complete.")

if __name__ == "__main__":
    main()