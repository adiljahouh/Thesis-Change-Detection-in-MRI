import torch
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
        self.conv3 = nn.Conv3d(64, 128, kernel_size=5, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=5, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.dropout = nn.Dropout3d(p=0.5)
        #self.fc1 = nn.Linear(68921, 128)  # Adjust input size based on input dimensions

    def forward(self, input1, input2):
        # Forward pass through the Siamese network
        output1 = F.relu(self.bn1(self.conv1(input1)))
        output1 = F.max_pool3d(output1, kernel_size=2, stride=2)
        
        output1 = F.relu(self.bn2(self.conv2(output1)))
        output1 = F.max_pool3d(output1, kernel_size=2, stride=2)
        
        output1 = F.relu(self.bn3(self.conv3(output1)))
        output1 = F.max_pool3d(output1, kernel_size=2, stride=2)
        
        output1 = F.relu(self.bn4(self.conv4(output1)))
        output1 = F.max_pool3d(output1, kernel_size=2, stride=2)

        output2 = F.relu(self.bn1(self.conv1(input2)))
        output2 = F.max_pool3d(output2, kernel_size=2, stride=2)
       
        output2 = F.relu(self.bn2(self.conv2(output2)))
        output2 = F.max_pool3d(output2, kernel_size=2, stride=2)
       
        output2 = F.relu(self.bn3(self.conv3(output2)))
        output2 = F.max_pool3d(output2, kernel_size=2, stride=2)

        output2 = F.relu(self.bn4(self.conv4(output2)))
        output2 = F.max_pool3d(output2, kernel_size=2, stride=2)

        return output1, output2


class SiameseVGG3D(nn.Module):
    def __init__(self):
        super(SiameseVGG3D, self).__init__()
        # Define 3D convolutional blocks
        self.conv1_1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv3d(512, 512, kernel_size=3, padding=1)

    def forward(self, input1, input2):
        # Process the first input through all layers
        output1 = F.relu(self.conv1_1(input1))
        output1 = F.relu(self.conv1_2(output1))
        output1 = F.max_pool3d(output1, kernel_size=2, stride=2)

        output1 = F.relu(self.conv2_1(output1))
        output1 = F.relu(self.conv2_2(output1))
        output1 = F.max_pool3d(output1, kernel_size=2, stride=2)

        output1 = F.relu(self.conv3_1(output1))
        output1 = F.relu(self.conv3_2(output1))
        output1 = F.relu(self.conv3_3(output1))
        output1 = F.max_pool3d(output1, kernel_size=2, stride=2)

        output1 = F.relu(self.conv4_1(output1))
        output1 = F.relu(self.conv4_2(output1))
        output1 = F.relu(self.conv4_3(output1))
        output1 = F.max_pool3d(output1, kernel_size=2, stride=2)

        output1 = F.relu(self.conv5_1(output1))
        output1 = F.relu(self.conv5_2(output1))
        output1 = F.relu(self.conv5_3(output1))
        output1 = F.max_pool3d(output1, kernel_size=2, stride=2)
        output1 = torch.flatten(output1, 1)  # Flatten the output

        # Process the second input through all layers
        output2 = F.relu(self.conv1_1(input2))
        output2 = F.relu(self.conv1_2(output2))
        output2 = F.max_pool3d(output2, kernel_size=2, stride=2)

        output2 = F.relu(self.conv2_1(output2))
        output2 = F.relu(self.conv2_2(output2))
        output2 = F.max_pool3d(output2, kernel_size=2, stride=2)

        output2 = F.relu(self.conv3_1(output2))
        output2 = F.relu(self.conv3_2(output2))
        output2 = F.relu(self.conv3_3(output2))
        output2 = F.max_pool3d(output2, kernel_size=2, stride=2)

        output2 = F.relu(self.conv4_1(output2))
        output2 = F.relu(self.conv4_2(output2))
        output2 = F.relu(self.conv4_3(output2))
        output2 = F.max_pool3d(output2, kernel_size=2, stride=2)

        output2 = F.relu(self.conv5_1(output2))
        output2 = F.relu(self.conv5_2(output2))
        output2 = F.relu(self.conv5_3(output2))
        output2 = F.max_pool3d(output2, kernel_size=2, stride=2)
        output2 = torch.flatten(output2, 1)  # Flatten the output

        return output1, output2
