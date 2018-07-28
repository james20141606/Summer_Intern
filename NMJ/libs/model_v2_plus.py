import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# -- 0.2 u-net+ --
# u-net with dilated convolution and ELU

class res_unet_plus(nn.Module):
    # unet architecture with residual blocks
    def __init__(self, in_num=1, out_num=1, filters=[32,64,64,96,96]):
        super(res_unet_plus, self).__init__()
        self.filters = filters 
        self.layer_num = len(filters) # 5
        self.aniso_num = 3 # the number of anisotropic conv layers

        self.downC = nn.ModuleList(
                  [res_unet_AnisoBlock_dilation(in_num, filters[0])]
                + [res_unet_AnisoBlock_dilation(filters[x], filters[x+1])
                      for x in range(self.aniso_num-1)] 
                + [res_unet_IsoBlock(filters[x], filters[x+1])
                      for x in range(self.aniso_num-1, self.layer_num-2)]) 

        self.downS = nn.ModuleList(
                [nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
                    for x in range(self.aniso_num)]
              + [nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
                    for x in range(self.aniso_num, self.layer_num-1)])

        self.center = res_unet_IsoBlock(filters[-2], filters[-1])

        self.upS = nn.ModuleList(
            [nn.Sequential(
                nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=False),
                nn.Conv3d(filters[self.layer_num-1-x], filters[self.layer_num-2-x], kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=True))
                for x in range(self.layer_num-self.aniso_num-1)]
          + [nn.Sequential(
                nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False),
                nn.Conv3d(filters[self.layer_num-1-x], filters[self.layer_num-2-x], kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=True))
                for x in range(1, self.aniso_num+1)])

        self.upC = nn.ModuleList(
            [res_unet_IsoBlock(filters[self.layer_num-2-x], filters[self.layer_num-2-x])
                for x in range(self.layer_num-self.aniso_num-1)]
          + [res_unet_AnisoBlock_dilation(filters[self.layer_num-2-x], filters[self.layer_num-2-x])
                for x in range(1, self.aniso_num)]
          + [nn.Sequential(
                  res_unet_AnisoBlock_dilation(filters[0], filters[0]),
                  nn.Conv3d(filters[0], out_num, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=True))])

        self.elu = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()      

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 

    def forward(self, x):
        down_u = [None]*(self.layer_num-1)
        for i in range(self.layer_num-1):
            down_u[i] = self.downC[i](x)
            x = self.downS[i](down_u[i])

        x = self.center(x)

        for i in range(self.layer_num-1):
            x = down_u[self.layer_num-2-i] + self.upS[i](x)
            x = self.elu(x)
            x = self.upC[i](x)
        # convert to probability    
        x = self.sigmoid(x)
        return x

# -- 0.3 building blocks--
class res_unet_IsoBlock(nn.Module):
    # Basic residual module of unet
    def __init__(self, in_planes, out_planes):
        super(res_unet_IsoBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(in_planes,  out_planes, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ELU(inplace=True))
        self.block2 = nn.Sequential(
            nn.Conv3d(out_planes, out_planes, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ELU(inplace=True),
            nn.Conv3d(out_planes, out_planes, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=False),
            nn.BatchNorm3d(out_planes))
        self.block3 = nn.ELU(inplace=True)    

    def forward(self, x):
        residual  = self.block1(x)
        out = residual + self.block2(residual)
        out = self.block3(out)
        return out 

# class res_unet_AnisoBlock(nn.Module):
#     # Basic residual module of unet
#     def __init__(self, in_planes, out_planes):
#         super(res_unet_AnisoBlock, self).__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv3d(in_planes,  out_planes, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
#             nn.BatchNorm3d(out_planes),
#             nn.ELU(inplace=True))
#         self.block2 = nn.Sequential(
#             nn.Conv3d(out_planes, out_planes, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
#             nn.BatchNorm3d(out_planes),
#             nn.ELU(inplace=True),
#             nn.Conv3d(out_planes, out_planes, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
#             nn.BatchNorm3d(out_planes))
#         self.block3 = nn.ELU(inplace=True)    

#     def forward(self, x):
#         residual  = self.block1(x)
#         out = residual + self.block2(residual)
#         out = self.block3(out)
#         return out 

class res_unet_AnisoBlock_dilation(nn.Module):
    # Basic residual module of unet
    def __init__(self, in_planes, out_planes):
        super(res_unet_AnisoBlock_dilation, self).__init__() 
        self.inconv = nn.Sequential(
            nn.Conv3d(in_planes,  out_planes, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=True),
            nn.BatchNorm3d(out_planes),
            nn.ELU(inplace=True))

        self.block1 = nn.Sequential(
            nn.Conv3d(out_planes, out_planes, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.BatchNorm3d(out_planes))

        self.block2 = nn.Sequential(
            nn.Conv3d(out_planes, out_planes, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ELU(inplace=True),
            nn.Conv3d(out_planes, out_planes, kernel_size=(1,3,3), stride=1, dilation=(1,2,2), padding=(0,2,2), bias=False),
            nn.BatchNorm3d(out_planes))

        self.block3 = nn.Sequential(
            nn.Conv3d(out_planes, out_planes, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ELU(inplace=True),
            nn.Conv3d(out_planes, out_planes, kernel_size=(1,3,3), stride=1, dilation=(1,2,2), padding=(0,2,2), bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ELU(inplace=True),
            nn.Conv3d(out_planes, out_planes, kernel_size=(1,3,3), stride=1, dilation=(1,4,4), padding=(0,4,4), bias=False),
            nn.BatchNorm3d(out_planes))

        self.block4 = nn.Sequential(
            nn.Conv3d(out_planes, out_planes, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ELU(inplace=True),
            nn.Conv3d(out_planes, out_planes, kernel_size=(1,3,3), stride=1, dilation=(1,2,2), padding=(0,2,2), bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ELU(inplace=True),
            nn.Conv3d(out_planes, out_planes, kernel_size=(1,3,3), stride=1, dilation=(1,4,4), padding=(0,4,4), bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ELU(inplace=True),
            nn.Conv3d(out_planes, out_planes, kernel_size=(1,3,3), stride=1, dilation=(1,8,8), padding=(0,8,8), bias=False),
            nn.BatchNorm3d(out_planes))

        self.activation = nn.ELU(inplace=True)    

    def forward(self, x):
        residual  = self.inconv(x)

        x1 = self.block1(residual)
        x2 = self.block2(residual)
        x3 = self.block3(residual)
        x4 = self.block4(residual)

        out = residual + x1 + x2 + x3 + x4
        out = self.activation(out)
        return out       