import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
from nets.mobilenetv3 import mobilenetv3

class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=16, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]  # 特征保存到interverted_residual_setting最后一层

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]  # 压缩图片长宽的卷积位置

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    # 这部分内容决定后面class DeepLab(nn.Module)中backbone中特征层的确定
    # def forward(self, x):
    #     low_level_features = self.features[:4](x)  # 浅层语义信息（开始的1x1，加上interverted_residual_setting的三层，共4层）
    #     x = self.features[4:](low_level_features)  # 深层语义信息（从interverted_residual_setting的第四层到最后）
    #     return low_level_features, x

    def forward(self, x):
        low_level_features1 = self.features[:4](x)  # 浅层语义信息（开始的1x1，加上interverted_residual_setting的三层，共4层）
        low_level_features2 = self.features[:7](x)
        # low_level_features3 = self.features[:11](x)
        x = self.features[7:](low_level_features2)  # 深层语义信息（从interverted_residual_setting的第四层到最后）
        return low_level_features1, low_level_features2, x
        # return low_level_features1, low_level_features2, low_level_features3, x


class MobileNetV3(nn.Module):
    def __init__(self, downsample_factor=16, pretrained=True):
        super(MobileNetV3, self).__init__()
        from functools import partial

        model = mobilenetv3(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 13]  # 压缩图片长宽的卷积位置

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__  # 得到网络层的名字
        if classname.find('Conv') != -1:  # 使用find函数，如果不存在返回值为-1，所以让其不等于-1
            if m.stride == (2, 2):  # 两个方向步长都为2
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    # 这部分内容决定后面class DeepLab(nn.Module)中backbone中特征层的确定
    def forward(self, x):
        low_level_features = self.features[:4](x)  # 浅层语义信息（开始的1x1，加上cfgs的三层，共4层）
        x = self.features[4:](low_level_features)  # 深层语义信息（从cfgs的第四层到最后）
        return low_level_features, x

    #  # 提取3个低级特征信息，共进行16倍下采样(改进之处)
    # def forward(self, x):
    #     low_level_features1 = self.features[:4](x)   # 浅层语义信息（开始的1x1，加上cfgs的三层，共4层）
    #     low_level_features2 = self.features[:7](x)
    #     low_level_features3 = self.features[:13](x)
    #     x = self.features[13:](low_level_features3)    # 深层语义信息（从cfgs的第四层到最后）
    #     return low_level_features1, low_level_features2, low_level_features3, x
    #     # return low_level_features1, low_level_features2, x


# -----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# -----------------------------------------#
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True), )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True), )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True), )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True), )
        # self.branch5 = nn.Sequential(
        # 		nn.Conv2d(dim_in, dim_out, 3, 1, padding=24*rate, dilation=24*rate, bias=True),
        # 		nn.BatchNorm2d(dim_out, momentum=bn_mom),
        # 		nn.ReLU(inplace=True),)

        # 图像池化过程
        # self.branch6_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
        # self.branch6_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        # self.branch6_relu = nn.ReLU(inplace=True)

        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        # 6个特征叠加
        # self.conv_cat = nn.Sequential(
        # 		nn.Conv2d(dim_out*6, dim_out, 1, 1, padding=0,bias=True),
        # 		nn.BatchNorm2d(dim_out, momentum=bn_mom),
        # 		nn.ReLU(inplace=True),)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True), )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共六个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # conv3x3_4 = self.branch5(x)
        # -----------------------------------------#
        #   第六个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        # global_feature = self.branch6_conv(global_feature)
        # global_feature = self.branch6_bn(global_feature)
        # global_feature = self.branch6_relu(global_feature)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将六个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class DeepLab(nn.Module):
    def __init__(self, num_classes = 6, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone == "xception":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            # ----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256

        # backbone为mobilenetv2
        elif backbone == "mobilenet":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24],[64, 64, 32],[32, 32, 64]
            #   主干部分    [16,16,320]
            # ----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            # low_level_channels = 24
            low_level_channels1 = 24
            low_level_channels2 = 32
            # low_level_channels3 = 64

        # # backbone为mobilenetv3
        # elif backbone == "mobilenet":
        #     # ----------------------------------#
        #     #   获得三个特征层
        #     #   浅层特征1    [128,128,24]
        #     #   浅层特征2    [64,64,40]
        #     #   浅层特征3    [32,32,112]
        #     #   主干部分    [16,16,160]
        #     # ----------------------------------#
        #     self.backbone = MobileNetV3(downsample_factor=downsample_factor, pretrained=pretrained)
        #     in_channels = 160
        #     low_level_channels = 24
        #     # low_level_channels1 = 24
        #     # low_level_channels2 = 40
        #     # low_level_channels3 = 112
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        # -----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)

        # ----------------------------------#
        #   浅层特征处理,以第一个浅层特征为标准进行处理
        # ----------------------------------#
        # self.shortcut_conv = nn.Sequential(
        #     nn.Conv2d(low_level_channels, 48, 1),
        #     nn.BatchNorm2d(48),
        #     nn.ReLU(inplace=True))

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels1, 24, 1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True))

        # 3x3卷积进行特征融合与提取
        self.cat_conv = nn.Sequential(
            nn.Conv2d(24 + 24 + 256, 256, 3, stride=1, padding=1),  # 提取几个特征值就要写几个通道
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

        # self.cat_conv = nn.Sequential(
        #     nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),  # 提取几个特征值就要写几个通道
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        # 最后进行通道数调整
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # -----------------------------------------#
        #   获得3（2个浅层+backbone出口的特征层）个特征层
        #   浅层特征-进行卷积处理
        #   主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        # low_level_features1, low_level_features2, low_level_features3, x = self.backbone(x)
        low_level_features1, low_level_features2, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features1 = self.shortcut_conv(low_level_features1)
        low_level_features2 = self.shortcut_conv(low_level_features1)
        # low_level_features3 = self.shortcut_conv(low_level_features1)

        # low_level_features, x = self.backbone(x)
        # x = self.aspp(x)
        # low_level_features = self.shortcut_conv(low_level_features)

        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = F.interpolate(x, size=(low_level_features1.size(2), low_level_features1.size(3)), mode='bilinear', align_corners=True)
        low_level_features2 = F.interpolate(low_level_features2, size=(low_level_features1.size(2), low_level_features1.size(3)), mode='bilinear', align_corners=True)
        # low_level_features3 = F.interpolate(low_level_features3, size=(low_level_features1.size(2), low_level_features1.size(3)), mode='bilinear', align_corners=True)
        # x = self.cat_conv(torch.cat((x, low_level_features1, low_level_features2, low_level_features3), dim=1))
        x = self.cat_conv(torch.cat((x, low_level_features1, low_level_features2), dim=1))
        # 最后进行上采样
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

        # x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
        #                   align_corners=True)
        # x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        # # 最后进行上采样
        # x = self.cls_conv(x)
        # x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        # return x
