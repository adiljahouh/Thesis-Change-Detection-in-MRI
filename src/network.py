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
    
class complexSiamese(nn.Module):
    def __init__(self):
        super(complexSiamese, self).__init__()
        # Define the architecture for the Siamese network
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)        
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)    
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # self.fc1 = nn.Linear(131072, 128)  # Adjust input size based on input dimensions
        # self.fc1 = nn.Linear(128 * 32 * 32, 128)  # Output size is 128, adjust if needed

    def forward(self, input1, input2):
        # Forward pass through the Siamese network
        output1_conv1 = F.relu(self.bn1(self.conv1(input1)))
        output1_pool1 = F.max_pool2d(output1_conv1, kernel_size=2, stride=2)
        output1_conv2 = F.relu(self.bn2(self.conv2(output1_pool1)))
        output1_pool2 = F.max_pool2d(output1_conv2, kernel_size=2, stride=2)
        output1_conv3 = F.relu(self.bn3(self.conv3(output1_pool2)))
        output1_pool3 = F.max_pool2d(output1_conv3, kernel_size=2, stride=2)
        output1_conv4 = F.relu(self.bn4(self.conv4(output1_pool3)))
        output1_pool4 = F.max_pool2d(output1_conv4, kernel_size=2, stride=2)


        output2_conv1 = F.relu(self.bn1(self.conv1(input2)))
        output2_pool1 = F.max_pool2d(output2_conv1, kernel_size=2, stride=2)
        output2_conv2 = F.relu(self.bn2(self.conv2(output2_pool1)))
        output2_pool2 = F.max_pool2d(output2_conv2, kernel_size=2, stride=2)
        output2_conv3 = F.relu(self.bn3(self.conv3(output2_pool2)))
        output2_pool3 = F.max_pool2d(output2_conv3, kernel_size=2, stride=2)
        output2_conv4 = F.relu(self.bn4(self.conv4(output2_pool3)))
        output2_pool4 = F.max_pool2d(output2_conv4, kernel_size=2, stride=2)
        # output2 = output2.view(output2.size(0), -1)  # Flatten to (batch_size, 128*32*32)
        # output2 = self.dropout(output2)
        # output2 = self.fc1(output2)

        return [output1_pool2, output2_pool2], [output1_pool3, output2_pool3], [output1_pool4, output2_pool4]
class deeplab_V2(nn.Module):
    def __init__(self):
        super(deeplab_V2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1 ,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),  
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,ceil_mode=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,ceil_mode=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,ceil_mode=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,dilation=2,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
        )

        ####### multi-scale contexts #######
        ####### dialtion = 6 ##########
        self.fc6_1 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,dilation=6,padding=6),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        ####### dialtion = 12 ##########
        self.fc6_2 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,dilation=12,padding=12),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        ####### dialtion = 18 ##########
        self.fc6_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=18, padding=18),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        ####### dialtion = 24 ##########
        self.fc6_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=24, padding=24),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.embedding_layer = nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1)
        #self.fc8 = nn.Softmax2d()
        #self.fc8 = fun.l2normalization(scale=1)

    def forward(self,x):

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
        fc_feature = fc7_1 + fc7_2 + fc7_3 + fc7_4
        #conv5_feature = self.fc8(x)
        #fc7_feature = self.fc8(fc)
        embedding_feature = self.embedding_layer(fc_feature)
        #score_final_up = F.upsample_bilinear(score_final,size[2:])
        #return conv4_feature,conv5_feature,fc_feature,embedding_feature
        return conv5_feature, fc_feature,embedding_feature
        #return fc_feature, embedding_feature
        #return embedding_feature


class l2normalization(nn.Module):
    def __init__(self,scale):

        super(l2normalization, self).__init__()
        self.scale = scale

    def forward(self, x, dim=1):
        '''out = scale * x / sqrt(\sum x_i^2)'''
        # return self.scale * x * x.pow(2).sum(dim).clamp(min=1e-12).rsqrt().expand_as(x)
        return (x - x.min()) / (x.max() - x.min())

class SiameseMLO(nn.Module):
    def __init__(self,norm_flag = 'l2'):
        super(SiameseMLO, self).__init__()
        self.CNN = deeplab_V2()
        if norm_flag == 'l2':
           self.norm = l2normalization(scale=1)
        if norm_flag == 'exp':
            self.norm = nn.Softmax2d()

    def forward(self,input1,input2):

        out_t0_conv5,out_t0_fc7,out_t0_embedding = self.CNN(input1)
        out_t1_conv5,out_t1_fc7,out_t1_embedding = self.CNN(input2)
        out_t0_conv5_norm,out_t1_conv5_norm = self.norm(out_t0_conv5),self.norm(out_t1_conv5)
        out_t0_fc7_norm,out_t1_fc7_norm = self.norm(out_t0_fc7),self.norm(out_t1_fc7)
        out_t0_embedding_norm,out_t1_embedding_norm = self.norm(out_t0_embedding),self.norm(out_t1_embedding)
        return [out_t0_conv5_norm,out_t1_conv5_norm],[out_t0_fc7_norm,out_t1_fc7_norm],[out_t0_embedding_norm,out_t1_embedding_norm]
