import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DiabeticRetinopathyModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(DiabeticRetinopathyModel, self).__init__()
        
        # Use a pre-trained model like EfficientNet
        self.base_model = models.efficientnet_b4(weights='DEFAULT' if pretrained else None)
        
        # Replace the classifier
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

def get_model(device, num_classes=5, pretrained=True):
    """Initialize the model and move it to the specified device."""
    model = DiabeticRetinopathyModel(num_classes, pretrained)
    model = model.to(device)
    return model

# Alternative model options
def get_resnet_model(device, num_classes=5, pretrained=True):
    """Get a ResNet50 model."""
    model = models.resnet50(weights='DEFAULT' if pretrained else None)
    # Replace the classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)

def get_densenet_model(device, num_classes=5, pretrained=True):
    """Get a DenseNet121 model."""
    model = models.densenet121(weights='DEFAULT' if pretrained else None)
    # Replace the classifier
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model.to(device)

if __name__ == "__main__":
    # Test model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    
    # Print model summary
    print(model)
    
    # Test with a random input
    batch_size = 4
    channels = 3
    height, width = 512, 512
    x = torch.randn(batch_size, channels, height, width).to(device)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")