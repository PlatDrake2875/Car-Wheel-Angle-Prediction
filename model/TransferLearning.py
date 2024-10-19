import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


class TLearning(nn.Module):
    def __init__(self):
        super(TLearning, self).__init__()
        self.base_model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
        self.base_model = nn.Sequential(*list(self.base_model.features))

        # Fully convolutional layers
        self.conv1 = nn.Conv2d(2560, 1024, kernel_size=1)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=1)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, Input):
        features = self.base_model(Input)  # Shape [batch_size, 2048, 7, 7]

        # Forward pass through the convolutional layers
        x = F.relu(self.conv1(features))  # Shape [batch_size, 1024, 7, 7]
        x = F.relu(self.conv2(x))  # Shape [batch_size, 512, 7, 7]
        x = F.relu(self.conv3(x))  # Shape [batch_size, 256, 7, 7]
        x = F.relu(self.conv4(x))  # Shape [batch_size, 64, 7, 7]
        x = self.conv5(x)  # Shape [batch_size, 1, 7, 7]

        # Feature map -> Single value
        angle = self.global_avg_pool(x)  # Shape [batch_size, 1, 1, 1]
        angle = torch.flatten(angle, 1)  # Flatten -> [batch_size, 1]

        return angle


if __name__ == "__main__":
    model = TLearning()
    dummy_input = torch.randn(5, 3, 224, 224)
    output = model(dummy_input)
    print("Output:", output)
