import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler
import time
from tqdm import tqdm

# Define paths
DATA_DIR = "../datasets/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")
PROCESSED_TRAIN_DIR = os.path.join(DATA_DIR, "processed_train_images")
PROCESSED_TEST_DIR = os.path.join(DATA_DIR, "processed_test_images")
os.makedirs(PROCESSED_TRAIN_DIR, exist_ok=True)
os.makedirs(PROCESSED_TEST_DIR, exist_ok=True)

def crop_black_borders(img):
    """
    Crop black borders around the retinal area.
    
    Parameters:
    img (numpy.ndarray): Input image
    
    Returns:
    numpy.ndarray: Cropped image containing only the retinal area
    """
    # Convert to grayscale if image is RGB
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Apply binary threshold to identify the retinal area
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours and len(contours) > 0:
        # Find the largest contour (retinal area)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add a small margin (5% of width/height)
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(img.shape[1] - x, w + 2*margin_x)
        h = min(img.shape[0] - y, h + 2*margin_y)
        
        # Return cropped image
        return img[y:y+h, x:x+w]
    
    return img  # Return original if no contours found

def apply_clahe(img):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance image contrast.
    
    Parameters:
    img (numpy.ndarray): Input RGB image
    
    Returns:
    numpy.ndarray: Enhanced RGB image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    
    # Convert back to RGB
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def resize_image(img, target_size=(512, 512)):
    """
    Resize image to target size.
    
    Parameters:
    img (numpy.ndarray): Input image
    target_size (tuple): Target dimensions (width, height)
    
    Returns:
    numpy.ndarray: Resized image
    """
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def normalize_image(img):
    """
    Normalize pixel values to range [-1, 1].
    
    Parameters:
    img (numpy.ndarray): Input image with values in range [0, 255]
    
    Returns:
    numpy.ndarray: Normalized image with values in range [-1, 1]
    """
    return (img / 127.5) - 1.0

def preprocess_image(img_path, target_size=(512, 512), save_path=None):
    """
    Apply full preprocessing pipeline to an image.
    
    Parameters:
    img_path (str): Path to the input image
    target_size (tuple): Target dimensions for resizing
    save_path (str, optional): Path to save the preprocessed image
    
    Returns:
    dict: Dictionary containing original image and intermediate results
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to read image at {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Store original for visualization
    original = img.copy()
    
    # Apply preprocessing steps
    cropped = crop_black_borders(img)
    enhanced = apply_clahe(cropped)
    resized = resize_image(enhanced, target_size)
    normalized = normalize_image(resized)
    
    # Save preprocessed image if path provided
    if save_path:
        # Convert normalized image back to [0, 255] range for saving
        save_img = ((normalized + 1.0) * 127.5).astype(np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))
    
    # For visualization and analysis purposes
    results = {
        'original': original,
        'cropped': cropped,
        'enhanced': enhanced,
        'resized': resized,
        'normalized': normalized
    }
    
    return results

def preprocess_and_save_all_images(csv_path, input_dir, output_dir, target_size=(512, 512)):
    """
    Preprocess all images in the dataset and save them to disk.
    
    Parameters:
    csv_path (str): Path to CSV file with image IDs
    input_dir (str): Directory containing original images
    output_dir (str): Directory to save preprocessed images
    target_size (tuple): Target size for preprocessed images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    total_images = len(df)
    
    print(f"Starting preprocessing of {total_images} images...")
    start_time = time.time()
    
    # Process each image with progress bar
    for idx, row in tqdm(df.iterrows(), total=total_images, desc="Preprocessing images"):
        img_id = row['id_code']
        input_path = os.path.join(input_dir, img_id + '.png')
        output_path = os.path.join(output_dir, img_id + '.png')
        
        # Skip if already processed
        if os.path.exists(output_path):
            continue
        
        # Preprocess and save
        try:
            preprocess_image(input_path, target_size, save_path=output_path)
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Preprocessing complete. {total_images} images processed in {elapsed_time:.2f} seconds")
    print(f"Preprocessed images saved to {output_dir}")

def visualize_preprocessing_samples(train_df, input_dir, output_dir, num_samples=3):
    """
    Visualize original and preprocessed images from each class.
    
    Parameters:
    train_df (pandas.DataFrame): DataFrame with image IDs and labels
    input_dir (str): Directory containing original images
    output_dir (str): Directory containing preprocessed images
    num_samples (int): Number of samples to visualize from each class
    """
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    num_classes = len(class_names)
    
    # Create figure
    fig, axes = plt.subplots(num_classes, num_samples*2, figsize=(num_samples*6, num_classes*3))
    
    # Visualize samples from each class
    for class_idx in range(num_classes):
        # Get samples from this class
        class_samples = train_df[train_df['diagnosis'] == class_idx].sample(
            min(num_samples, sum(train_df['diagnosis'] == class_idx))
        )
        
        # Plot each sample
        for i, (_, row) in enumerate(class_samples.iterrows()):
            img_id = row['id_code']
            orig_path = os.path.join(input_dir, img_id + '.png')
            proc_path = os.path.join(output_dir, img_id + '.png')
            
            # Original image
            orig_img = cv2.imread(orig_path)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
            # Preprocessed image
            proc_img = cv2.imread(proc_path)
            proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
            
            # Plot original
            ax = axes[class_idx, i*2]
            ax.imshow(orig_img)
            if i == 0:
                ax.set_ylabel(f"Class {class_idx}: {class_names[class_idx]}", fontsize=12)
            if class_idx == 0:
                ax.set_title("Original", fontsize=12)
            ax.axis('off')
            
            # Plot preprocessed
            ax = axes[class_idx, i*2+1]
            ax.imshow(proc_img)
            if class_idx == 0:
                ax.set_title("Preprocessed", fontsize=12)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'preprocessing_samples.png'))
    plt.show()

def load_and_split_data(csv_path, test_size=0.2, random_state=42):
    """
    Load data from CSV and split into training and validation sets with stratification.
    
    Parameters:
    csv_path (str): Path to CSV file with image IDs and labels
    test_size (float): Proportion of data to use for validation
    random_state (int): Random seed for reproducibility
    
    Returns:
    tuple: Training and validation DataFrames
    """
    # Load the data
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df)} images")
    
    # Display class distribution
    class_dist = df['diagnosis'].value_counts().sort_index()
    print("Class distribution:")
    for i, count in enumerate(class_dist):
        print(f"Class {i}: {count} images ({count/len(df)*100:.2f}%)")
    
    # Split with stratification to maintain class balance
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['diagnosis']
    )
    
    print(f"\nAfter splitting:")
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    
    # Save splits to CSV for reference
    train_df.to_csv(os.path.join(DATA_DIR, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, 'val_split.csv'), index=False)
    
    return train_df, val_df

def calculate_class_weights(labels):
    """
    Calculate class weights inversely proportional to class frequencies.
    
    Parameters:
    labels (numpy.ndarray): Array of class labels
    
    Returns:
    dict: Dictionary mapping class indices to weights
    """
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(labels), 
        y=labels
    )
    weights_dict = dict(zip(np.unique(labels), class_weights))
    
    print("Class weights for balanced training:")
    for class_idx, weight in weights_dict.items():
        print(f"Class {class_idx}: {weight:.4f}")
    
    return weights_dict

def generate_dataset_info(train_df, val_df, processed_dir):
    """
    Generate and save dataset information for training phase.
    
    Parameters:
    train_df (pandas.DataFrame): Training DataFrame
    val_df (pandas.DataFrame): Validation DataFrame
    processed_dir (str): Directory with processed images
    """
    # Create info directory
    info_dir = os.path.join(DATA_DIR, 'info')
    os.makedirs(info_dir, exist_ok=True)
    
    # Calculate and save class weights
    train_labels = train_df['diagnosis'].values
    class_weights = calculate_class_weights(train_labels)
    
    # Save class weights
    np.save(os.path.join(info_dir, 'class_weights.npy'), class_weights)
    
    # Save dataset stats
    dataset_stats = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'class_distribution': train_df['diagnosis'].value_counts().sort_index().to_dict(),
        'class_weights': class_weights,
        'image_size': (512, 512),
        'preprocessing': 'crop_borders + CLAHE + resize + normalize'
    }
    
    # Save as text file
    with open(os.path.join(info_dir, 'dataset_info.txt'), 'w') as f:
        f.write("Diabetic Retinopathy Dataset Information\n")
        f.write("======================================\n\n")
        f.write(f"Training set: {dataset_stats['train_size']} images\n")
        f.write(f"Validation set: {dataset_stats['val_size']} images\n\n")
        
        f.write("Class distribution (training set):\n")
        for class_idx, count in dataset_stats['class_distribution'].items():
            f.write(f"Class {class_idx}: {count} images ({count/dataset_stats['train_size']*100:.2f}%)\n")
        
        f.write("\nClass weights for balanced training:\n")
        for class_idx, weight in dataset_stats['class_weights'].items():
            f.write(f"Class {class_idx}: {weight:.4f}\n")
        
        f.write(f"\nImage size: {dataset_stats['image_size'][0]}x{dataset_stats['image_size'][1]}\n")
        f.write(f"Preprocessing: {dataset_stats['preprocessing']}\n")
    
    print(f"Dataset information saved to {info_dir}")

def prepare_test_set():
    """
    Preprocess the test set images for later inference.
    """
    # Check if test CSV exists
    if not os.path.exists(TEST_CSV):
        print("Test CSV not found. Skipping test set preprocessing.")
        return
    
    # Load test CSV
    test_df = pd.read_csv(TEST_CSV)
    
    print(f"Preprocessing {len(test_df)} test images...")
    
    # Process and save test images
    preprocess_and_save_all_images(
        TEST_CSV,
        TEST_IMAGES_DIR,
        PROCESSED_TEST_DIR,
        target_size=(512, 512)
    )
    
    print("Test set preprocessing complete.")

def main():
    """Main function to preprocess all images and prepare the dataset for training."""
    print("Starting diabetic retinopathy image preprocessing pipeline...")
    
    # Step 1: Load and split the data
    train_df, val_df = load_and_split_data(TRAIN_CSV)
    
    # Step 2: Preprocess training images
    print("\nPreprocessing training images...")
    preprocess_and_save_all_images(
        os.path.join(DATA_DIR, 'train_split.csv'),
        TRAIN_IMAGES_DIR,
        PROCESSED_TRAIN_DIR,
        target_size=(512, 512)
    )
    
    # Step 3: Preprocess validation images
    print("\nPreprocessing validation images...")
    preprocess_and_save_all_images(
        os.path.join(DATA_DIR, 'val_split.csv'),
        TRAIN_IMAGES_DIR,
        PROCESSED_TRAIN_DIR,
        target_size=(512, 512)
    )
    
    # Step 4: Preprocess test images (if available)
    prepare_test_set()
    
    # Step 5: Visualize some preprocessing examples
    print("\nGenerating visualization of preprocessing results...")
    visualize_preprocessing_samples(train_df, TRAIN_IMAGES_DIR, PROCESSED_TRAIN_DIR)
    
    # Step 6: Generate dataset information for training phase
    print("\nGenerating dataset information...")
    generate_dataset_info(train_df, val_df, PROCESSED_TRAIN_DIR)
    
    print("\nAll preprocessing tasks completed successfully!")
    print(f"Preprocessed training/validation images: {PROCESSED_TRAIN_DIR}")
    print(f"Preprocessed test images: {PROCESSED_TEST_DIR}")
    print(f"Dataset information: {os.path.join(DATA_DIR, 'info')}")
    print("\nYou can now proceed to the model training phase.")

if __name__ == "__main__":
    main()