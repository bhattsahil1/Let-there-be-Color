import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def Conv2d(in_ch, out_ch, stride, kernel_size=3, padding=1):
    return nn.Conv2d(in_channels=in_ch, out_channels=out_ch,stride=stride, kernel_size=kernel_size, padding=padding)


class LowLevelFeatures(nn.Module):
    def __init__(self):
        super(LowLevelFeatures, self).__init__()
        self.conv1 = Conv2d(1, 64, 2)
        self.conv2 = Conv2d(64, 128, 1)
        self.conv3 = Conv2d(128, 128, 2)
        self.conv4 = Conv2d(128, 256, 1)
        self.conv5 = Conv2d(256, 256, 2)
        self.conv6 = Conv2d(256, 512, 1)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        return out


class MidLevelFeatures(nn.Module):
    def __init__(self):
        super(MidLevelFeatures, self).__init__()
        self.conv7 = Conv2d(512, 512, 1)
        self.conv8 = Conv2d(512, 256, 1)

    def forward(self, x):
        out = F.relu(self.conv7(x))
        out = F.relu(self.conv8(out))
        return out


class GlobalFeatures(nn.Module):
    def __init__(self):
        super(GlobalFeatures, self).__init__()
        self.conv1 = Conv2d(512, 512, 2)
        self.conv2 = Conv2d(512, 512, 1)
        self.conv3 = Conv2d(512, 512, 2)
        self.conv4 = Conv2d(512, 512, 1)

        self.fc1 = nn.Linear(7*7*512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = F.relu(self.conv4(y))
        y = y.view(-1, 7*7*512)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))        
        out = F.relu(self.fc3(y))
        return out

class ColorizationNetwork(nn.Module):
    def __init__(self):
        super(ColorizationNetwork, self).__init__()
        self.conv9 = Conv2d(256, 128, 1)

        self.conv10 = Conv2d(128, 64, 1)
        self.conv11 = Conv2d(64, 64, 1)

        self.conv12 = Conv2d(64, 32, 1)
        self.conv13 = Conv2d(32, 2, 1)
    
    def forward(self, x):
        out = F.relu(self.conv9(x))

        # Upsample #1        
        out = nn.functional.interpolate(input=out, scale_factor=2)

        out = F.relu(self.conv10(out))
        out = F.relu(self.conv11(out))
        
        # Upsample #2
        out = nn.functional.interpolate(input=out, scale_factor=2)

        out = F.relu(self.conv12(out))
        out = torch.sigmoid(self.conv13(out))
        
        # Upsample #3
        out = nn.functional.interpolate(input=out, scale_factor=2)
        
        return out


class ColNet(nn.Module):
    def __init__(self, num_classes):
        super(ColNet, self).__init__()
        self.conv_fuse = Conv2d(512, 256, 1, kernel_size=1, padding=0)
        self.low = LowLevelFeatures()
        self.mid = MidLevelFeatures()
        self.glob = GlobalFeatures()
        self.col = ColorizationNetwork()


    def fusion_layer(self, mid_out, glob_out):
        h = mid_out.shape[2]   
        w = mid_out.shape[3]  
        
        glob_stack2d = torch.stack(tuple(glob_out for _ in range(w)), 1)
        glob_stack3d = torch.stack(tuple(glob_stack2d for _ in range(h)), 1)
        glob_stack3d = glob_stack3d.permute(0, 3, 1, 2)

        stack_volume = torch.cat((mid_out, glob_stack3d), 1)

        out = F.relu(self.conv_fuse(stack_volume))
        return out


    def forward(self, x):
        low_out = self.low(x)                               #Low-level features
        mid_out = low_out                                   #Sharing weights with mid level
        glob_out = low_out                                  #Sharing weights with global features
        mid_out = self.mid(mid_out)                         #Mid-level features
        glob_out = self.glob(glob_out)                      #Global features
        out = self.fusion_layer(mid_out, glob_out)          #Fusion layer
        out = self.col(out)                                 #Colorization network                
        return out
        