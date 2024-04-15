import torch.nn as nn
import torch.nn.functional as F
class SiameseThreeDim(nn.Module):
    def __init__(self):
        super(SiameseThreeDim, self).__init__()
        # Define the architecture for the Siamese network
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(10, 128)  # Adjust input size based on input dimensions

    def forward(self, input1, input2):
        # Forward pass through the Siamese network
        output1 = F.relu(self.bn1(self.conv1(input1)))
        output1 = F.max_pool3d(output1, kernel_size=2, stride=2)
        output1 = F.relu(self.bn2(self.conv2(output1)))
        output1 = F.max_pool3d(output1, kernel_size=2, stride=2)
        output1 = F.relu(self.bn3(self.conv3(output1)))
        output1 = F.max_pool3d(output1, kernel_size=2, stride=2)

        output2 = F.relu(self.bn1(self.conv1(input2)))
        output2 = F.max_pool3d(output2, kernel_size=2, stride=2)
        output2 = F.relu(self.bn2(self.conv2(output2)))
        output2 = F.max_pool3d(output2, kernel_size=2, stride=2)
        output2 = F.relu(self.bn3(self.conv3(output2)))
        output2 = F.max_pool3d(output2, kernel_size=2, stride=2)

        return output1, output2