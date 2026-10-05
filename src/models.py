"""Three understandable CNN architectures; logits for stable cross entropy."""
import torch
from torch import nn
from torchvision.models import mobilenet_v2, resnet50, MobileNet_V2_Weights, ResNet50_Weights

class CustomCNN(nn.Module):
    """Four convolutional blocks and global pooling avoid a huge dense layer."""
    def __init__(self, num_classes=6):
        super().__init__()
        layers = []
        channels = 3
        for width in (32, 64, 128, 192):
            layers += [nn.Conv2d(channels, width, 3, padding=1, bias=False),
                       nn.BatchNorm2d(width), nn.ReLU(inplace=False),
                       nn.MaxPool2d(2), nn.Dropout2d(0.1)]
            channels = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Dropout(0.35), nn.Linear(192, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))

def build_model(name, pretrained=True, num_classes=6):
    """Use official torchvision ImageNet weights or reconstruct from checkpoint."""
    if name == 'custom_cnn':
        return CustomCNN(num_classes)
    if name == 'mobilenetv2':
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.last_channel, num_classes))
        return model
    if name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
        return model
    raise ValueError(f'Unknown architecture: {name}')

def configure_phase(model, name, finetune=False):
    """Freeze backbone, or unfreeze MobileNet final blocks / ResNet layer4."""
    for parameter in model.parameters():
        parameter.requires_grad = name == 'custom_cnn'
    if name == 'custom_cnn':
        return
    head = model.classifier if name == 'mobilenetv2' else model.fc
    for parameter in head.parameters():
        parameter.requires_grad = True
    if finetune:
        later = model.features[14:] if name == 'mobilenetv2' else model.layer4
        for parameter in later.parameters():
            parameter.requires_grad = True

def training_mode(model, name):
    """Keep pretrained batch-norm statistics fixed even during fine-tuning."""
    model.train()
    if name != 'custom_cnn':
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

def gradcam_layer(model, name):
    """Choose final spatial feature activations before global pooling."""
    if name == 'custom_cnn':
        return model.features[-4]
    if name == 'mobilenetv2':
        return model.features[-1]
    return model.layer4[-1]
