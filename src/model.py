import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

def setup_logger(model_name):
    """Set up and return a logger for the specified model."""
    logger = logging.getLogger(f"{model_name}_model")
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(f"../logs/{model_name}_model.log")
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

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, version='b4'):
        super(EfficientNetModel, self).__init__()
        
        # Choose EfficientNet version
        if version == 'b0':
            self.base_model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        elif version == 'b1':
            self.base_model = models.efficientnet_b1(weights='DEFAULT' if pretrained else None)
        elif version == 'b2':
            self.base_model = models.efficientnet_b2(weights='DEFAULT' if pretrained else None)
        elif version == 'b3':
            self.base_model = models.efficientnet_b3(weights='DEFAULT' if pretrained else None)
        elif version == 'b4':
            self.base_model = models.efficientnet_b4(weights='DEFAULT' if pretrained else None)
        elif version == 'b5':
            self.base_model = models.efficientnet_b5(weights='DEFAULT' if pretrained else None)
        elif version == 'b6':
            self.base_model = models.efficientnet_b6(weights='DEFAULT' if pretrained else None)
        elif version == 'b7':
            self.base_model = models.efficientnet_b7(weights='DEFAULT' if pretrained else None)
        else:
            raise ValueError(f"Unsupported EfficientNet version: {version}")
        
        # Replace the classifier
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

class ResNetModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, version='50'):
        super(ResNetModel, self).__init__()
        
        # Choose ResNet version
        if version == '18':
            self.base_model = models.resnet18(weights='DEFAULT' if pretrained else None)
        elif version == '34':
            self.base_model = models.resnet34(weights='DEFAULT' if pretrained else None)
        elif version == '50':
            self.base_model = models.resnet50(weights='DEFAULT' if pretrained else None)
        elif version == '101':
            self.base_model = models.resnet101(weights='DEFAULT' if pretrained else None)
        elif version == '152':
            self.base_model = models.resnet152(weights='DEFAULT' if pretrained else None)
        else:
            raise ValueError(f"Unsupported ResNet version: {version}")
        
        # Replace the classifier
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

class DenseNetModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, version='121'):
        super(DenseNetModel, self).__init__()
        
        # Choose DenseNet version
        if version == '121':
            self.base_model = models.densenet121(weights='DEFAULT' if pretrained else None)
        elif version == '161':
            self.base_model = models.densenet161(weights='DEFAULT' if pretrained else None)
        elif version == '169':
            self.base_model = models.densenet169(weights='DEFAULT' if pretrained else None)
        elif version == '201':
            self.base_model = models.densenet201(weights='DEFAULT' if pretrained else None)
        else:
            raise ValueError(f"Unsupported DenseNet version: {version}")
        
        # Replace the classifier
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

def get_model(device, model_type='EfficientNet', model_version=None, num_classes=5, pretrained=True):
    """
    Initialize the model and move it to the specified device.
    
    Parameters:
    device (torch.device): Device to move the model to
    model_type (str): Type of model to use - 'EfficientNet', 'ResNet', or 'DenseNet'
    model_version (str): Version of the model (e.g., 'b4' for EfficientNet, '50' for ResNet)
    num_classes (int): Number of output classes
    pretrained (bool): Whether to use pre-trained weights
    
    Returns:
    model: The initialized model on the specified device
    """
    # Create logs directory if it doesn't exist
    import os
    os.makedirs("../logs", exist_ok=True)
    
    # Set up logger
    logger = setup_logger(model_type.lower())
    
    # Set default model versions if none provided
    if model_version is None:
        if model_type == 'EfficientNet':
            model_version = 'b4'
        elif model_type == 'ResNet':
            model_version = '50'
        elif model_type == 'DenseNet':
            model_version = '121'
    
    # Log model initialization
    logger.info(f"Initializing {model_type} model (version: {model_version}, pretrained: {pretrained})")
    
    # Initialize the appropriate model
    if model_type == 'EfficientNet':
        model = EfficientNetModel(num_classes=num_classes, pretrained=pretrained, version=model_version)
        logger.info(f"EfficientNet-{model_version} model created")
    elif model_type == 'ResNet':
        model = ResNetModel(num_classes=num_classes, pretrained=pretrained, version=model_version)
        logger.info(f"ResNet-{model_version} model created")
    elif model_type == 'DenseNet':
        model = DenseNetModel(num_classes=num_classes, pretrained=pretrained, version=model_version)
        logger.info(f"DenseNet-{model_version} model created")
    else:
        error_msg = f"Unsupported model type: {model_type}. Choose from 'EfficientNet', 'ResNet', or 'DenseNet'."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Move to device
    model = model.to(device)
    logger.info(f"Model moved to device: {device}")
    
    return model

if __name__ == "__main__":
    # Test all model types
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create logs directory if it doesn't exist
    import os
    os.makedirs("../logs", exist_ok=True)
    
    for model_type in ['EfficientNet', 'ResNet', 'DenseNet']:
        model = get_model(device, model_type=model_type)
        
        # Print model summary
        print(f"\n{model_type} Model Summary:")
        print(model)
        
        # Test with a random input
        batch_size = 4
        channels = 3
        height, width = 512, 512
        x = torch.randn(batch_size, channels, height, width).to(device)
        output = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")