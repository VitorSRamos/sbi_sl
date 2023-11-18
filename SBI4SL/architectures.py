import torch.nn as nn
import torch
from torchsummary import summary
from torchview import draw_graph
import graphviz

# Criando Classe de RN
class CNN1(nn.Module):
    # CNN baseada em Poh et al
    def __init__(self, input_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size[0], 8, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.mp1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.mp2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        
        self.conv6 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.mp3 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        
        self.linear = nn.Linear(3200, 16)

    def forward(self, image):
        image = self.conv1(image)
        image = self.bn1(image)
        
        image = self.conv2(image)
        image = self.bn2(image)
        image = self.mp1(image)
        
        image = self.conv3(image)
        image = self.bn3(image)
        
        image = self.conv4(image)
        image = self.bn4(image)
        image = self.mp2(image)
        
        image = self.conv5(image)
        image = self.bn5(image)
        
        image = self.conv6(image)
        image = self.bn6(image)
        image = self.mp3(image)
        
        image = self.flatten(image)
        
        image = self.linear(image)
        return image
    
class CNN2(nn.Module):
    # CNN muito mais agressiva, larga mas não mais profunda
    def __init__(self, input_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size[0], 64, 4, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        #self.mp1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.mp2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        #self.mp3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(256, 512, 3, 1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.mp4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(512, 1024, 3, 1, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        #self.mp5 = nn.MaxPool2d(2, 2)

        self.conv6 = nn.Conv2d(1024, 1024, 3, 1, padding=1)
        self.bn6 = nn.BatchNorm2d(1024)
        self.mp6 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        
        self.linear = nn.Linear(25600, 64)

    def forward(self, image):
        image = self.conv1(image)
        image = self.bn1(image)
        #image = self.mp1(image)

        image = self.conv2(image)
        image = self.bn2(image)
        image = self.mp2(image)
        
        image = self.conv3(image)
        image = self.bn3(image)
        #image = self.mp3(image)

        image = self.conv4(image)
        image = self.bn4(image)
        image = self.mp4(image)
        
        image = self.conv5(image)
        image = self.bn5(image)
        #image = self.mp5(image)

        image = self.conv6(image)
        image = self.bn6(image)
        image = self.mp6(image)
        
        image = self.flatten(image)
        
        image = self.linear(image)
        return image
    
class CNN3(nn.Module):
    # CNN mais progressiva, mas tbm mais profunda
    def __init__(self, input_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size[0], 8, 2, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        #self.mp1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(8, 16, 2, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        #self.mp2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(16, 32, 2, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.mp3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        #self.mp4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        #self.mp5 = nn.MaxPool2d(2, 2)

        self.conv6 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.mp6 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(256, 512, 4, 1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        #self.mp7 = nn.MaxPool2d(2, 2)

        self.conv8 = nn.Conv2d(512, 1024, 4, 1, padding=1)
        self.bn8 = nn.BatchNorm2d(1024)
        #self.mp8 = nn.MaxPool2d(2, 2)
        
        self.conv9 = nn.Conv2d(1024, 2048, 4, 1, padding=1)
        self.bn9 = nn.BatchNorm2d(2048)
        self.mp9 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        
        self.linear = nn.Linear(32768, 64)

    def forward(self, image):
        image = self.conv1(image)
        image = self.bn1(image)
        #image = self.mp1(image)

        image = self.conv2(image)
        image = self.bn2(image)
        #image = self.mp2(image)
        
        image = self.conv3(image)
        image = self.bn3(image)
        image = self.mp3(image)

        image = self.conv4(image)
        image = self.bn4(image)
        #image = self.mp4(image)
        
        image = self.conv5(image)
        image = self.bn5(image)
        #image = self.mp5(image)

        image = self.conv6(image)
        image = self.bn6(image)
        image = self.mp6(image)

        image = self.conv7(image)
        image = self.bn7(image)
        #image = self.mp7(image)
        
        image = self.conv8(image)
        image = self.bn8(image)
        #image = self.mp8(image)

        image = self.conv9(image)
        image = self.bn9(image)
        image = self.mp9(image)
        
        image = self.flatten(image)
        
        #image = self.linear(image)
        return image
        

def inception_block(input_channels, out_channels):
    
    output_channels = int(out_channels/4) # out = 4K, output=K (clecio 2019)
    
    branch1 = nn.Sequential(
        nn.Conv2d(input_channels,  output_channels, 1,1, padding=0), # (1,1) convolution
        nn.BatchNorm2d( output_channels),
        nn.ReLU(),
    )
    
    branch2 = nn.Sequential(
        nn.Conv2d(input_channels,  output_channels, 1,1, padding=0), # (1,1) convolution
        nn.Conv2d( output_channels,   output_channels, 3,1, padding=1), # (3,3) cconvoution
        nn.BatchNorm2d( output_channels),
        nn.ReLU(),
    )
    
    branch3 = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 1,1, padding=0), # (1,1) convolution
        nn.Conv2d(output_channels, output_channels, 3,1, padding=1), # (3,3) cconvoution
        nn.Conv2d(output_channels, output_channels, 3,1, padding=1), # (3,3) cconvoution
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),
    )

    branch4 = nn.Sequential(
        nn.MaxPool2d(3,1, padding=1),
        nn.Conv2d(input_channels, output_channels, 1,1, padding=0), # (1,1) convolution
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),
    )
    return branch1, branch2, branch3, branch4

def output_block(input_neurons):
    output_block = nn.Sequential(
        nn.Flatten(1, -1), #starts at dimension 1 and ends at dimension -1
        nn.Linear(input_neurons, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    return output_block

class inception_full_76(nn.Module):
    # taken from de Bom 2018
    def __init__(self, input_size, n_out_features):
        super().__init__()
        self.n_out_features = n_out_features # number of output blocks in output (same as number of parameters to estimate)

        # input stream
        self.input_stream = nn.Sequential(
            nn.Conv2d(input_size[0], 64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 5, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # inception block 1
        self.inception1_branch1, self.inception1_branch2, self.inception1_branch3, self.inception1_branch4  = inception_block(128, 128)
        
        # inception block 2
        self.inception2_branch1, self.inception2_branch2, self.inception2_branch3, self.inception2_branch4  = inception_block(128, 256)

        # inception block 3
        self.inception3_branch1, self.inception3_branch2, self.inception3_branch3, self.inception3_branch4  = inception_block(256, 512)

        # inception block 4
        self.inception4_branch1, self.inception4_branch2, self.inception4_branch3, self.inception4_branch4  = inception_block(512, 1024)

        # output stream
        self.output_block1 = output_block(262144)
        self.output_block2 = output_block(262144) # uncomment this line to infer a second parameter

    def forward(self, image):
        # input stream
        image = self.input_stream(image)
        
        # core stream
        # inception block 1
        i1_b1 = self.inception1_branch1(image)
        i1_b2 = self.inception1_branch2(image)
        i1_b3 = self.inception1_branch3(image)
        i1_b4 = self.inception1_branch4(image)
        image = torch.cat([i1_b1, i1_b2, i1_b3, i1_b4], dim=1) # inception block 1 output

        # inception block 2
        i2_b1 = self.inception2_branch1(image)
        i2_b2 = self.inception2_branch2(image)
        i2_b3 = self.inception2_branch3(image)
        i2_b4 = self.inception2_branch4(image)
        image = torch.cat([i2_b1, i2_b2, i2_b3, i2_b4], dim=1) # inception block 2 output

        # inception block 3
        i3_b1 = self.inception3_branch1(image)
        i3_b2 = self.inception3_branch2(image)
        i3_b3 = self.inception3_branch3(image)
        i3_b4 = self.inception3_branch4(image)
        image = torch.cat([i3_b1, i3_b2, i3_b3, i3_b4], dim=1) # inception block 3 output

        # inception block 4
        i4_b1 = self.inception4_branch1(image)
        i4_b2 = self.inception4_branch2(image)
        i4_b3 = self.inception4_branch3(image)
        i4_b4 = self.inception4_branch4(image)
        image = torch.cat([i4_b1, i4_b2, i4_b3, i4_b4], dim=1) # inception block 4 output
        
        # output stream
        out1 = self.output_block1(image)
        out2 = self.output_block2(image)
        return [out1, out2]

class inception_feature_extractor_76(nn.Module):
    # taken from de Bom 2018
    def __init__(self, input_size, n_out_features):
        super().__init__()
        self.n_out_features = n_out_features # number of features extracted from image

        # input stream
        self.input_stream = nn.Sequential(
            nn.Conv2d(input_size[0], 64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 5, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # inception block 1
        self.inception1_branch1, self.inception1_branch2, self.inception1_branch3, self.inception1_branch4  = inception_block(128, 128)
        
        # inception block 2
        self.inception2_branch1, self.inception2_branch2, self.inception2_branch3, self.inception2_branch4  = inception_block(128, 256)

        # inception block 3
        self.inception3_branch1, self.inception3_branch2, self.inception3_branch3, self.inception3_branch4  = inception_block(256, 512)

        # inception block 4
        self.inception4_branch1, self.inception4_branch2, self.inception4_branch3, self.inception4_branch4  = inception_block(512, 1024)

        # output stream
        self.output_stream = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(262144, self.n_out_features)
        )

    def forward(self, image):
        # input stream
        image = self.input_stream(image)
        
        # core stream
        # inception block 1
        i1_b1 = self.inception1_branch1(image)
        i1_b2 = self.inception1_branch2(image)
        i1_b3 = self.inception1_branch3(image)
        i1_b4 = self.inception1_branch4(image)
        image = torch.cat([i1_b1, i1_b2, i1_b3, i1_b4], dim=1) # inception block 1 output

        # inception block 2
        i2_b1 = self.inception2_branch1(image)
        i2_b2 = self.inception2_branch2(image)
        i2_b3 = self.inception2_branch3(image)
        i2_b4 = self.inception2_branch4(image)
        image = torch.cat([i2_b1, i2_b2, i2_b3, i2_b4], dim=1) # inception block 2 output

        # inception block 3
        i3_b1 = self.inception3_branch1(image)
        i3_b2 = self.inception3_branch2(image)
        i3_b3 = self.inception3_branch3(image)
        i3_b4 = self.inception3_branch4(image)
        image = torch.cat([i3_b1, i3_b2, i3_b3, i3_b4], dim=1) # inception block 3 output

        # inception block 4
        i4_b1 = self.inception4_branch1(image)
        i4_b2 = self.inception4_branch2(image)
        i4_b3 = self.inception4_branch3(image)
        i4_b4 = self.inception4_branch4(image)
        image = torch.cat([i4_b1, i4_b2, i4_b3, i4_b4], dim=1) # inception block 4 output
        
        # output stream
        out = self.output_stream(image)
        return out

class inception_full_45(nn.Module):
    # taken from de Bom 2018
    def __init__(self, input_size, n_out_features):
        super().__init__()
        self.n_out_features = n_out_features # number of output blocks in output (same as number of parameters to estimate)

        # input stream
        self.input_stream = nn.Sequential(
            nn.Conv2d(input_size[0], 64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 5, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # inception block 1
        self.inception1_branch1, self.inception1_branch2, self.inception1_branch3, self.inception1_branch4  = inception_block(128, 128)
        
        # inception block 2
        self.inception2_branch1, self.inception2_branch2, self.inception2_branch3, self.inception2_branch4  = inception_block(128, 256)

        # inception block 3
        self.inception3_branch1, self.inception3_branch2, self.inception3_branch3, self.inception3_branch4  = inception_block(256, 512)

        # inception block 4
        self.inception4_branch1, self.inception4_branch2, self.inception4_branch3, self.inception4_branch4  = inception_block(512, 1024)

        # output stream
        self.output_block1 = output_block(65536)
        #self.output_block2 = output_block(262144) # uncomment this line to infer a second parameter

    def forward(self, image):
        # input stream
        image = self.input_stream(image)
        
        # core stream
        # inception block 1
        i1_b1 = self.inception1_branch1(image)
        i1_b2 = self.inception1_branch2(image)
        i1_b3 = self.inception1_branch3(image)
        i1_b4 = self.inception1_branch4(image)
        image = torch.cat([i1_b1, i1_b2, i1_b3, i1_b4], dim=1) # inception block 1 output

        # inception block 2
        i2_b1 = self.inception2_branch1(image)
        i2_b2 = self.inception2_branch2(image)
        i2_b3 = self.inception2_branch3(image)
        i2_b4 = self.inception2_branch4(image)
        image = torch.cat([i2_b1, i2_b2, i2_b3, i2_b4], dim=1) # inception block 2 output

        # inception block 3
        i3_b1 = self.inception3_branch1(image)
        i3_b2 = self.inception3_branch2(image)
        i3_b3 = self.inception3_branch3(image)
        i3_b4 = self.inception3_branch4(image)
        image = torch.cat([i3_b1, i3_b2, i3_b3, i3_b4], dim=1) # inception block 3 output

        # inception block 4
        i4_b1 = self.inception4_branch1(image)
        i4_b2 = self.inception4_branch2(image)
        i4_b3 = self.inception4_branch3(image)
        i4_b4 = self.inception4_branch4(image)
        image = torch.cat([i4_b1, i4_b2, i4_b3, i4_b4], dim=1) # inception block 4 output
        
        # output stream
        out1 = self.output_block1(image)
        #out2 = self.output_block2(image)
        #return [out1, out2]
        return out1

class inception_feature_extractor_45(nn.Module):
    # taken from de Bom 2018
    def __init__(self, input_size, n_out_features):
        super().__init__()
        self.n_out_features = n_out_features # number of features extracted from image

        # input stream
        self.input_stream = nn.Sequential(
            nn.Conv2d(input_size[0], 64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 5, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # inception block 1
        self.inception1_branch1, self.inception1_branch2, self.inception1_branch3, self.inception1_branch4  = inception_block(128, 128)
        
        # inception block 2
        self.inception2_branch1, self.inception2_branch2, self.inception2_branch3, self.inception2_branch4  = inception_block(128, 256)

        # inception block 3
        self.inception3_branch1, self.inception3_branch2, self.inception3_branch3, self.inception3_branch4  = inception_block(256, 512)

        # inception block 4
        self.inception4_branch1, self.inception4_branch2, self.inception4_branch3, self.inception4_branch4  = inception_block(512, 1024)

        # output stream
        self.output_stream = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(65536, self.n_out_features)
        )

    def forward(self, image):
        # input stream
        image = self.input_stream(image)
        
        # core stream
        # inception block 1
        i1_b1 = self.inception1_branch1(image)
        i1_b2 = self.inception1_branch2(image)
        i1_b3 = self.inception1_branch3(image)
        i1_b4 = self.inception1_branch4(image)
        image = torch.cat([i1_b1, i1_b2, i1_b3, i1_b4], dim=1) # inception block 1 output

        # inception block 2
        i2_b1 = self.inception2_branch1(image)
        i2_b2 = self.inception2_branch2(image)
        i2_b3 = self.inception2_branch3(image)
        i2_b4 = self.inception2_branch4(image)
        image = torch.cat([i2_b1, i2_b2, i2_b3, i2_b4], dim=1) # inception block 2 output

        # inception block 3
        i3_b1 = self.inception3_branch1(image)
        i3_b2 = self.inception3_branch2(image)
        i3_b3 = self.inception3_branch3(image)
        i3_b4 = self.inception3_branch4(image)
        image = torch.cat([i3_b1, i3_b2, i3_b3, i3_b4], dim=1) # inception block 3 output

        # inception block 4
        i4_b1 = self.inception4_branch1(image)
        i4_b2 = self.inception4_branch2(image)
        i4_b3 = self.inception4_branch3(image)
        i4_b4 = self.inception4_branch4(image)
        image = torch.cat([i4_b1, i4_b2, i4_b3, i4_b4], dim=1) # inception block 4 output
        
        # output stream
        out = self.output_stream(image)
        return out

class inception_feature_extractor_87(nn.Module):
    # taken from de Bom 2018
    def __init__(self, input_size, n_out_features):
        super().__init__()
        self.n_out_features = n_out_features # number of features extracted from image

        # input stream
        self.input_stream = nn.Sequential(
            nn.Conv2d(input_size[0], 64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 5, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # inception block 1
        self.inception1_branch1, self.inception1_branch2, self.inception1_branch3, self.inception1_branch4  = inception_block(128, 128)
        
        # inception block 2
        self.inception2_branch1, self.inception2_branch2, self.inception2_branch3, self.inception2_branch4  = inception_block(128, 256)

        # inception block 3
        self.inception3_branch1, self.inception3_branch2, self.inception3_branch3, self.inception3_branch4  = inception_block(256, 512)

        # inception block 4
        self.inception4_branch1, self.inception4_branch2, self.inception4_branch3, self.inception4_branch4  = inception_block(512, 1024)

        # output stream
        self.output_stream = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(331776, self.n_out_features)
        )

    def forward(self, image):
        # input stream
        image = self.input_stream(image)
        
        # core stream
        # inception block 1
        i1_b1 = self.inception1_branch1(image)
        i1_b2 = self.inception1_branch2(image)
        i1_b3 = self.inception1_branch3(image)
        i1_b4 = self.inception1_branch4(image)
        image = torch.cat([i1_b1, i1_b2, i1_b3, i1_b4], dim=1) # inception block 1 output

        # inception block 2
        i2_b1 = self.inception2_branch1(image)
        i2_b2 = self.inception2_branch2(image)
        i2_b3 = self.inception2_branch3(image)
        i2_b4 = self.inception2_branch4(image)
        image = torch.cat([i2_b1, i2_b2, i2_b3, i2_b4], dim=1) # inception block 2 output

        # inception block 3
        i3_b1 = self.inception3_branch1(image)
        i3_b2 = self.inception3_branch2(image)
        i3_b3 = self.inception3_branch3(image)
        i3_b4 = self.inception3_branch4(image)
        image = torch.cat([i3_b1, i3_b2, i3_b3, i3_b4], dim=1) # inception block 3 output

        # inception block 4
        i4_b1 = self.inception4_branch1(image)
        i4_b2 = self.inception4_branch2(image)
        i4_b3 = self.inception4_branch3(image)
        i4_b4 = self.inception4_branch4(image)
        image = torch.cat([i4_b1, i4_b2, i4_b3, i4_b4], dim=1) # inception block 4 output
        
        # output stream
        out = self.output_stream(image)
        return out