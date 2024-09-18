import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSiamese(nn.Module):
    def __init__(self):
        super(SimpleSiamese, self).__init__()
        # Define the architecture for the Siamese network
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(p=0.5)
        # self.fc1 = nn.Linear(131072, 128)  # Adjust input size based on input dimensions
        self.fc1 = nn.Linear(128 * 32 * 32, 128)  # Output size is 128, adjust if needed

    def forward(self, input1, input2):
        # Forward pass through the Siamese network
        output1 = F.relu(self.bn1(self.conv1(input1)))
        output1 = F.max_pool2d(output1, kernel_size=2, stride=2)
        output1 = F.relu(self.bn2(self.conv2(output1)))
        output1 = F.max_pool2d(output1, kernel_size=2, stride=2)
        output1 = F.relu(self.bn3(self.conv3(output1)))
        output1 = F.max_pool2d(output1, kernel_size=2, stride=2)
        # output1 = output1.view(output1.size(0), -1)  # Flatten to (batch_size, 128*32*32)
        # # output1 = self.dropout(output1)
        # output1 = self.fc1(output1)
        
        output2 = F.relu(self.bn1(self.conv1(input2)))
        output2 = F.max_pool2d(output2, kernel_size=2, stride=2)
        output2 = F.relu(self.bn2(self.conv2(output2)))
        output2 = F.max_pool2d(output2, kernel_size=2, stride=2)
        output2 = F.relu(self.bn3(self.conv3(output2)))
        output2 = F.max_pool2d(output2, kernel_size=2, stride=2)
        # output2 = output2.view(output2.size(0), -1)  # Flatten to (batch_size, 128*32*32)
        # output2 = self.dropout(output2)
        # output2 = self.fc1(output2)

        return output1, output2


class SiameseMLO(nn.Module):
    def __init__(self):
        super(SiameseMLO, self).__init__()
        # Define the architecture for the Siamese network
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(p=0.5)
        # self.fc1 = nn.Linear(131072, 128)  # Adjust input size based on input dimensions
        self.fc1 = nn.Linear(128 * 32 * 32, 128)  # Output size is 128, adjust if needed

    def forward(self, input1, input2):
        # Forward pass through the Siamese network
        output1 = F.relu(self.bn1(self.conv1(input1)))
        output1 = F.max_pool2d(output1, kernel_size=2, stride=2)
        output1 = F.relu(self.bn2(self.conv2(output1)))
        output1 = F.max_pool2d(output1, kernel_size=2, stride=2)
        output1 = F.relu(self.bn3(self.conv3(output1)))
        output1 = F.max_pool2d(output1, kernel_size=2, stride=2)
        # output1 = output1.view(output1.size(0), -1)  # Flatten to (batch_size, 128*32*32)
        # # output1 = self.dropout(output1)
        # output1 = self.fc1(output1)
        
        output2 = F.relu(self.bn1(self.conv1(input2)))
        output2 = F.max_pool2d(output2, kernel_size=2, stride=2)
        output2 = F.relu(self.bn2(self.conv2(output2)))
        output2 = F.max_pool2d(output2, kernel_size=2, stride=2)
        output2 = F.relu(self.bn3(self.conv3(output2)))
        output2 = F.max_pool2d(output2, kernel_size=2, stride=2)
        # output2 = output2.view(output2.size(0), -1)  # Flatten to (batch_size, 128*32*32)
        # output2 = self.dropout(output2)
        # output2 = self.fc1(output2)

        return output1, output2