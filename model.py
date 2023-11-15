import torch
import torch.nn as nn
import torch.nn.functional as F

#basic block for unet++
class ConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.bt1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1)
        self.bt2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bt1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bt2(out)
        out = self.relu(out)
        return out

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPPModule, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.atrous_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0])
        self.atrous_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1])
        self.atrous_conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2])
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        self.image_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.atrous_conv1(x)
        x3 = self.atrous_conv2(x)
        x4 = self.atrous_conv3(x)
        x5 = self.image_pool(x)
        x5 = self.image_conv(x5)
        x5 = nn.functional.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=False)
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return out

class UNet_ASPP(nn.Module):
    def __init__(self, num_classes, in_channels, deep_supervision, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        # out_channels = [32, 64, 128, 256, 512]
        out_channels = [288, 288*2, 288*4, 288*8, 288*16]

        #backbone
        self.conv0_0 = ConvBlock(in_channels, out_channels[0], out_channels[0])
        self.conv1_0 = ConvBlock(out_channels[0], out_channels[1], out_channels[1])
        self.conv2_0 = ConvBlock(out_channels[1], out_channels[2], out_channels[2])
        self.conv3_0 = ConvBlock(out_channels[2], out_channels[3], out_channels[3])
        self.conv4_0 = ConvBlock(out_channels[3], out_channels[4], out_channels[4])

        #skip pathways
        #each conv: input = previous conv + lower conv, output = previous conv
        self.conv0_1 = ConvBlock(out_channels[0]+out_channels[1], out_channels[0], out_channels[0])
        self.conv1_1 = ConvBlock(out_channels[1]+out_channels[2], out_channels[1], out_channels[1])
        self.conv2_1 = ConvBlock(out_channels[2]+out_channels[3], out_channels[2], out_channels[2])
        self.conv3_1 = ConvBlock(out_channels[3]+out_channels[4], out_channels[3], out_channels[3])

        self.conv0_2 = ConvBlock(out_channels[0]*2+out_channels[1], out_channels[0], out_channels[0])
        self.conv1_2 = ConvBlock(out_channels[1]*2+out_channels[2], out_channels[1], out_channels[1])
        self.conv2_2 = ConvBlock(out_channels[2]*2+out_channels[3], out_channels[2], out_channels[2])

        self.conv0_3 = ConvBlock(out_channels[0]*3+out_channels[1], out_channels[0], out_channels[0])
        self.conv1_3 = ConvBlock(out_channels[1]*3+out_channels[2], out_channels[1], out_channels[1])

        self.conv0_4 = ConvBlock(out_channels[0]*4+out_channels[1], out_channels[0], out_channels[0])

        #pooling and up_sampling
        self.pool = nn.MaxPool2d(2, 2)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #ASPP module
        aspp_in_channels = out_channels[4]
        aspp_out_channels = out_channels[4]
        aspp_rates = [6, 12, 18]
        self.aspp = ASPPModule(aspp_in_channels, aspp_out_channels, aspp_rates)

        #deep_supervision=True: accurate mode
        #False: fast mode
        if self.deep_supervision:
            self.ds1 = nn.Conv2d(out_channels[0], num_classes, kernel_size=1)
            self.ds2 = nn.Conv2d(out_channels[0], num_classes, kernel_size=1)
            self.ds3 = nn.Conv2d(out_channels[0], num_classes, kernel_size=1)
            self.ds4 = nn.Conv2d(out_channels[0], num_classes, kernel_size=1)
        else:
            self.ds = nn.Conv2d(out_channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        #backbone
        x = x.float()
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # input = previous conv + up_sample lower conv
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up_sample(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up_sample(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up_sample(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up_sample(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up_sample(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up_sample(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up_sample(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up_sample(x1_2)], 1))
        x1_3 = self.conv0_3(torch.cat([x1_0, x1_1, x1_2, self.up_sample(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up_sample(x1_3)], 1))

        # x0_1 = self.conv0_1(x0_0)
        # x1_1 = self.conv1_1(x1_0)
        # x2_1 = self.conv2_1(x2_0)
        # x3_1 = self.conv3_1(torch.cat([x3_0, self.up_sample(x4_0)], 1))

        # x0_2 = self.conv0_2(x0_1)
        # x1_2 = self.conv1_2(x1_1)
        # x2_2 = self.conv2_2(torch.cat([x2_1, self.up_sample(x3_1)], 1))

        # x0_3 = self.conv0_3(x0_2)
        # x1_3 = self.conv0_3(torch.cat([x1_2, self.up_sample(x2_2)], 1))

        # x0_4 = self.conv0_4(torch.cat([x0_3, self.up_sample(x1_3)], 1))

        #ASPP module
        aspp_out = self.aspp(x4_0)

        if self.deep_supervision:
            out1 = self.ds1(x0_1)
            out2 = self.ds2(x0_2)
            out3 = self.ds3(x0_3)
            out4 = self.ds4(x0_4)
            return [out1, out2, out3, out4, aspp_out]

        else:
            out = self.ds(x0_4)
            return out