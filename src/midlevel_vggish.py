"""
PyTorch MidLevelVGGish Deep Neural Network Architecture
Extracts 7 Mid-Level Perceptual Features from Audio Spectrograms/MFCCs:
1. Melodiousness
2. Articulation
3. Rhythmic Stability
4. Rhythmic Complexity
5. Dissonance
6. Tonal Stability
7. Minorness
Ref: Aljanaki & Soleymani (ISMIR 2018), Lyberatos et al. (IEEE Access 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MidLevelVGGish(nn.Module):
    """
    VGG-ish Convolutional Neural Network for predicting 7 mid-level perceptual features.
    Input: Spectrogram / MFCC matrix (Batch, 1, 40, 649) or (Batch, 1, Mel_Bins, Time_Frames)
    Output: 7-dimensional perceptual vector bounded in [0, 1]
    """
    def __init__(self, num_targets=7):
        super(MidLevelVGGish, self).__init__()
        
        # Block 1
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.3)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.3)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(384)
        
        # Block 4
        self.conv4_1 = nn.Conv2d(384, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.bn4_2 = nn.BatchNorm2d(256)
        
        # Global Adaptive Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Output Linear Head for 7 Perceptual Qualities
        self.fc = nn.Linear(256, num_targets)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, 1, F, T)
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.drop1(self.pool1(x))
        
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.drop2(self.pool2(x))
        
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        out = self.sigmoid(self.fc(x))
        return out

def get_midlevel_feature_names():
    return [
        "Melodiousness", "Articulation", "Rhythmic_Stability",
        "Rhythmic_Complexity", "Dissonance", "Tonal_Stability", "Minorness"
    ]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MidLevelVGGish().to(device)
    dummy_input = torch.randn(4, 1, 40, 649).to(device)
    output = model(dummy_input)
    print(f"MidLevelVGGish initialized on {device}.")
    print(f"Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
    print(f"Predicted Perceptual Qualities: {get_midlevel_feature_names()}")
