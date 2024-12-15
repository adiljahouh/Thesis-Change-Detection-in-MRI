import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSiamese(nn.Module):
    def __init__(self):
        """
            Siamese network with a simple architecture
            used for Single layer output so will not be accurate for complex tasks
            also can be inefficient for noise and 
        """
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
    

class complexSiameseExt(nn.Module):
    def __init__(self):
        super(complexSiameseExt, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)        
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)    
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        ## same output channels, just to refine the features later on
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    
        self.bn5 = nn.BatchNorm2d(512)
        
        self.conv8 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(1024)
        
        # self.fc1 = nn.Linear(131072, 128)  # Adjust input size based on input dimensions
        # self.fc1 = nn.Linear(128 * 32 * 32, 128)  # Output size is 128, adjust if needed

    def forward(self, input1, input2, mode='train'):
        # Forward pass through the Siamese network
        output1_conv1 = F.relu(self.bn1(self.conv1(input1)))
        output1_pool1 = F.max_pool2d(output1_conv1, kernel_size=2, stride=2)
        output1_conv2 = F.relu(self.bn2(self.conv2(output1_pool1)))
        output1_pool2 = F.max_pool2d(output1_conv2, kernel_size=2, stride=2)
        output1_conv3 = F.relu(self.bn3(self.conv3(output1_pool2)))
        output1_pool3 = F.max_pool2d(output1_conv3, kernel_size=2, stride=2)
        
        output1_conv4 = F.relu(self.conv4(output1_pool3))
        output1_conv5 = F.relu(self.bn4(self.conv5(output1_conv4)))
        output1_pool4 = F.max_pool2d(output1_conv5, kernel_size=2, stride=2)
        
        output1_conv6 = F.relu(self.conv6(output1_pool4))
        output1_conv7 = F.relu(self.bn5(self.conv7(output1_conv6)))
        output1_pool5 = F.max_pool2d(output1_conv7, kernel_size=2, stride=2)
        output1_conv8 = F.relu(self.conv8(output1_pool5))
        output1_conv9 = F.relu(self.bn6(self.conv9(output1_conv8)))
        output1_pool6 = F.max_pool2d(output1_conv9, kernel_size=2, stride=2)
        
        #######
        
        output2_conv1 = F.relu(self.bn1(self.conv1(input2)))
        output2_pool1 = F.max_pool2d(output2_conv1, kernel_size=2, stride=2)
        output2_conv2 = F.relu(self.bn2(self.conv2(output2_pool1)))
        output2_pool2 = F.max_pool2d(output2_conv2, kernel_size=2, stride=2)
        output2_conv3 = F.relu(self.bn3(self.conv3(output2_pool2)))
        output2_pool3 = F.max_pool2d(output2_conv3, kernel_size=2, stride=2)
        
        output2_conv4 = F.relu(self.conv4(output2_pool3))
        output2_conv5 = F.relu(self.bn4(self.conv5(output2_conv4)))
        output2_pool4 = F.max_pool2d(output2_conv5, kernel_size=2, stride=2)
    
        output2_conv6 = F.relu(self.conv6(output2_pool4))
        output2_conv7 = F.relu(self.bn5(self.conv7(output2_conv6)))
        output2_pool5 = F.max_pool2d(output2_conv7, kernel_size=2, stride=2)
        output2_conv8 = F.relu(self.conv8(output2_pool5))
        output2_conv9 = F.relu(self.bn6(self.conv9(output2_conv8)))
        output2_pool6 = F.max_pool2d(output2_conv9, kernel_size=2, stride=2)
        if mode == 'train':
            return [output1_pool4, output2_pool4], [output1_pool5, output2_pool5], [output1_pool6, output2_pool6]
        elif mode == 'test':
            # return before the pooling layer to visualize them
            return [output1_conv3, output2_conv3], [output1_conv4, output2_conv4], [output1_pool6, output2_pool6]


class testDeepSiamese(nn.Module):
    def __init__(self):
        super(testDeepSiamese, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)        
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)    
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        ## same output channels, just to refine the features later on
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    
        self.bn5 = nn.BatchNorm2d(512)
        
        self.conv8 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(1024)
    def forward(self, input1, input2, mode='train'):
        # Forward pass through the Siamese network
        output1_conv1 = F.relu(self.bn1(self.conv1(input1)))
        # output1_pool1 = F.max_pool2d(output1_conv1, kernel_size=2, stride=2)
        output1_conv2 = F.relu(self.bn2(self.conv2(output1_conv1)))
        # output1_pool2 = F.max_pool2d(output1_conv2, kernel_size=2, stride=2)
        output1_conv3 = F.relu(self.bn3(self.conv3(output1_conv2)))
        output1_pool3 = F.max_pool2d(output1_conv3, kernel_size=2, stride=2)
        
        output1_conv4 = F.relu(self.conv4(output1_pool3))
        output1_conv5 = F.relu(self.bn4(self.conv5(output1_conv4)))
        output1_pool4 = F.max_pool2d(output1_conv5, kernel_size=2, stride=2)
        
        output1_conv6 = F.relu(self.conv6(output1_pool4))
        output1_conv7 = F.relu(self.bn5(self.conv7(output1_conv6)))
        output1_pool5 = F.max_pool2d(output1_conv7, kernel_size=2, stride=2)
        output1_conv8 = F.relu(self.conv8(output1_pool5))
        output1_conv9 = F.relu(self.bn6(self.conv9(output1_conv8)))
        output1_pool6 = F.max_pool2d(output1_conv9, kernel_size=2, stride=2)
        
        #######
        
        output2_conv1 = F.relu(self.bn1(self.conv1(input2)))
        # output2_pool1 = F.max_pool2d(output2_conv1, kernel_size=2, stride=2)
        output2_conv2 = F.relu(self.bn2(self.conv2(output2_conv1)))
        # output2_pool2 = F.max_pool2d(output2_conv2, kernel_size=2, stride=2)
        output2_conv3 = F.relu(self.bn3(self.conv3(output2_conv2)))
        output2_pool3 = F.max_pool2d(output2_conv3, kernel_size=2, stride=2)
        
        output2_conv4 = F.relu(self.conv4(output2_pool3))
        output2_conv5 = F.relu(self.bn4(self.conv5(output2_conv4)))
        output2_pool4 = F.max_pool2d(output2_conv5, kernel_size=2, stride=2)
    
        output2_conv6 = F.relu(self.conv6(output2_pool4))
        output2_conv7 = F.relu(self.bn5(self.conv7(output2_conv6)))
        output2_pool5 = F.max_pool2d(output2_conv7, kernel_size=2, stride=2)
        output2_conv8 = F.relu(self.conv8(output2_pool5))
        output2_conv9 = F.relu(self.bn6(self.conv9(output2_conv8)))
        output2_pool6 = F.max_pool2d(output2_conv9, kernel_size=2, stride=2)
        if mode == 'train':
            return [output1_pool4, output2_pool4], [output1_pool5, output2_pool5], [output1_pool6, output2_pool6]
        elif mode == 'test':
            # return before the pooling layer to visualize them
            return [output1_conv5, output2_conv5], [output1_conv7, output2_conv7], [output1_pool6, output2_pool6]

class l2normalization(nn.Module):
    def __init__(self,scale):

        super(l2normalization, self).__init__()
        self.scale = scale

    def forward(self, x, dim=1):
        '''out = scale * x / sqrt(\sum x_i^2)'''
        # return self.scale * x * x.pow(2).sum(dim).clamp(min=1e-12).rsqrt().expand_as(x)
        return (x - x.min()) / (x.max() - x.min())
