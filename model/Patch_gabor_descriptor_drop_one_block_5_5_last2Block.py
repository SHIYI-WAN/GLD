from __future__ import division, print_function

from pyexpat import features

import torch
import torch.nn.init
import torch.nn as nn
from GaborEncoder_patch_drop_one_block_5_5 import gabor_encoder_v2


def center_crop_and_align(feature_map: torch.Tensor, target_h: int = 48, target_w: int = 32) -> torch.Tensor:
   
    _, _, H, W = feature_map.shape

    if H < target_h or W < target_w:
        raise ValueError("Feature map size is smaller than target crop size.")

 
    h_start = (H - target_h) // 2
    h_end = h_start + target_h
    w_start = (W - target_w) // 2
    w_end = w_start + target_w

    cropped_feature = feature_map[:, :, h_start:h_end, w_start:w_end]

    return cropped_feature


class FeatureFusion(nn.Module):
    def __init__(self, total_in_channels: int, fuse_channels: int = 256):
        super().__init__()


        self.conv1x1 = nn.Sequential(
            nn.Conv2d(total_in_channels, fuse_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(fuse_channels),
            nn.ReLU(inplace=True)
        )


        self.conv3x3 = nn.Sequential(
            nn.Conv2d(fuse_channels, fuse_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(fuse_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features) :


        fused_concat = torch.cat(features, dim=1)


        fused_1x1 = self.conv1x1(fused_concat)


        # output = self.conv3x3(fused_1x1)

        return fused_1x1

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x


class L2Net(nn.Module):
    def __init__(self):
        super(L2Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=(4,3), stride=2, padding=1, bias = False),#3
            nn.BatchNorm2d(160, affine=False),
            nn.ReLU(),
            nn.Conv2d(160, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(4,3), stride=2,padding=1, bias = False),#3
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            # nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            # nn.BatchNorm2d(128, affine=False),
            # nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=(12,8), bias = False),#8
            nn.BatchNorm2d(128, affine=False),

        )
        self.features.apply(weights_init)
        return

    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)

        return L2Norm()(x)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant_(m.bias.data, 0.01)

        except:
            pass
    return

def get_net():
    return L2Net()


class Gabor_descriptor(nn.Module):

    def __init__(self):
        super(Gabor_descriptor, self).__init__()
        self.GDblocks = gabor_encoder_v2()
        self.fusion =FeatureFusion(83,160)
        self.L2net_part = L2Net()

    def forward(self,input_patch):
        gray_image = input_patch[:, 0] * 0.2989 + input_patch[:, 1] * 0.5870 + input_patch[:, 2] * 0.1140
        gray_image = gray_image.unsqueeze(1)
        gray_image= gray_image.repeat(1, 8, 1, 1)
        x1,x2 = self.GDblocks(gray_image)
        x0=center_crop_and_align(input_patch)
        x1= center_crop_and_align(x1)
        # x2 = center_crop_and_align(x2)
        features = [x0,x1,x2]
        x = self.fusion(features)
        x = self.L2net_part(x)

        return x


    # from torchvision import transforms
    # from PIL import Image
    #
    #
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    #
    #
    # image_path = '/project/project/shiyi/lyft_training_data/Adapted_Data_out2/patch/Cluster_1/host-a004_cam0_1232815252451064006/1.png'
    # image = Image.open(image_path).convert('RGB')
    # image_tensor = transform(image)
    # image_tensor = image_tensor.unsqueeze(0)
