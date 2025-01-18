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
            #return [output1_conv3, output2_conv3], [output1_pool4, output2_pool4], [output1_conv4, output2_conv4]
            return [output1_pool4, output2_pool4], [output1_pool5, output2_pool5], [output1_pool6, output2_pool6]

class DeepLab(nn.Module):
    def __init__(self):
        super(DeepLab, self).__init__()
        
        # Initial layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Dilated convolution layers
        self.dilated_conv4_1 = nn.Conv2d(256, 512, kernel_size=3, dilation=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.dilated_conv5_1 = nn.Conv2d(512, 1024, kernel_size=3, dilation=4, padding=4)
        self.bn5 = nn.BatchNorm2d(1024)
        
        # Final embedding layers
        self.embedding_conv = nn.Conv2d(1024, 1024, kernel_size=1)
        self.bn_embedding = nn.BatchNorm2d(1024)

    def forward(self, input1, input2, mode='train'):
        def siamese_branch(input):
            # Initial layers
            layer1_output = F.relu(self.bn1(self.conv1(input)))
            pool1_output = F.max_pool2d(layer1_output, kernel_size=2, stride=2)
            
            layer2_output = F.relu(self.bn2(self.conv2(pool1_output)))
            pool2_output = F.max_pool2d(layer2_output, kernel_size=2, stride=2)
            
            layer3_output = F.relu(self.bn3(self.conv3(pool2_output)))
            
            # Dilated convolutions
            dilated4_output = F.relu(self.bn4(self.dilated_conv4_1(layer3_output)))
            
            dilated5_output = F.relu(self.bn5(self.dilated_conv5_1(dilated4_output)))
            
            # Embedding layer
            embedding_output = F.relu(self.bn_embedding(self.embedding_conv(dilated5_output)))
            
            return [layer1_output, dilated4_output, embedding_output]
        
        # Siamese branches
        output1 = siamese_branch(input1)
        output2 = siamese_branch(input2)
        
        if mode == 'train':
            return [output1[0], output2[0]], [output1[1], output2[1]], [output1[2], output2[2]]
        elif mode == 'test':
            return [output1[0], output2[0]], [output1[1], output2[1]], [output1[2], output2[2]]


class DeepLabExtended(nn.Module):
    def __init__(self):
        super(DeepLabExtended, self).__init__()
        
        # Initial layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Downscale 256x256 -> 128x128
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Downscale 128x128 -> 64x64
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Downscale 64x64 -> 32x32
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)  # Keep 32x32
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
        )

        # Multi-scale contexts
        self.fc6_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, dilation=3, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc6_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, dilation=6, padding=6),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc6_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, dilation=9, padding=9),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc6_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, dilation=12, padding=12),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.embedding_layer = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)

    def forward_branch(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        conv3_feature = self.conv3(x)
        conv4_feature = self.conv4(conv3_feature)
        conv5_feature = self.conv5(conv4_feature)
        fc6_1 = self.fc6_1(conv5_feature)
        fc7_1 = self.fc7_1(fc6_1)
        fc6_2 = self.fc6_2(conv5_feature)
        fc7_2 = self.fc7_2(fc6_2)
        fc6_3 = self.fc6_3(conv5_feature)
        fc7_3 = self.fc7_3(fc6_3)
        fc6_4 = self.fc6_4(conv5_feature)
        fc7_4 = self.fc7_4(fc6_4)
        # print(fc7_1.shape, fc7_2.shape, fc7_3.shape, fc7_4.shape)
        fc_feature = fc7_1 + fc7_2 + fc7_3 + fc7_4
        embedding_feature = self.embedding_layer(fc_feature)
        return conv4_feature, conv5_feature, fc_feature
    # def normalize(self, x, scale = 1.0, dim = 1):
    #     return scale * x * x.pow(2).sum(dim).clamp(min=1e-12).rsqrt().expand_as(x)
    def normalize(self, x, scale=1.0, dim=1):
        norm = x.pow(2).sum(dim=dim, keepdim=True).clamp(min=1e-12).rsqrt()
        return scale * x * norm
    def forward(self, x1, x2, mode='train'):
        out1 = self.forward_branch(x1)
        out2 = self.forward_branch(x2)
        if mode == 'train':
            return [self.normalize(out1[0]), self.normalize(out2[0])], [self.normalize(out1[1]), self.normalize(out2[1])], [self.normalize(out1[2]), self.normalize(out2[2])]
        elif mode == 'test':
            # return before the pooling layer to visualize them
            #return [output1_conv3, output2_conv3], [output1_pool4, output2_pool4], [output1_conv4, output2_conv4]
            return [self.normalize(out1[0]), self.normalize(out2[0])], [self.normalize(out1[1]), self.normalize(out2[1])], [self.normalize(out1[2]), self.normalize(out2[2])]
