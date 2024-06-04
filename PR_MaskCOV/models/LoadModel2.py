import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import pretrainedmodels
from models.Asoftmax_linear import AngleLinear

from config import pretrained_model

import pdb
# 58.33
class MainModel2(nn.Module):
    def __init__(self, config,args):
        super(MainModel2, self).__init__()
        self.use_cdrm = config.use_cdrm
        self.num_classes = config.numcls
        self.backbone_arch = config.backbone
        self.use_Asoftmax = config.use_Asoftmax
        print(self.backbone_arch)

        if self.backbone_arch in dir(models):
            self.model = getattr(models, self.backbone_arch)()

            if self.backbone_arch in pretrained_model:
                self.model.load_state_dict(torch.load(pretrained_model[self.backbone_arch]))
        else:
            if self.backbone_arch in pretrained_model:
                self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000, pretrained=None)
            else:
                self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000)

        if self.backbone_arch == 'resnet50' or self.backbone_arch == 'se_resnet50':
            self.model = nn.Sequential(*list(self.model.children())[:-2])  # remove avgpool and fc
        if self.backbone_arch == 'senet154':
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if self.backbone_arch == 'se_resnext101_32x4d':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'se_resnet101':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.conv_dim = nn.Conv2d(2048, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)

        if self.use_cdrm:
            if config.cls_2:
                self.classifier_swap = nn.Linear(2048, 2, bias=False)
            if config.cls_2xmul:
                self.classifier_swap = nn.Linear(2048, 2*self.num_classes, bias=False)

            self.blockN = config.swap_num[0]*config.swap_num[1]
            self.classifier_cova = nn.Linear(2048, self.blockN*9, bias=False)

        if self.use_Asoftmax:
            self.Aclassifier = AngleLinear(2048, self.num_classes, bias=False)

    def forward(self, x, last_cont=None):
        x = self.model(x)  # resnet50 backbone
        
        f = x
        # print(f.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        out = []
        out.append(self.classifier(x))

        if self.use_cdrm:
            out.append(self.classifier_swap(x))
            f1 = self.conv_dim(f)
            # print(f1.shape)
            f2 = self.crop_feature(f1)
            # print(f2.shape)
            # import pdb; pdb.set_trace()
            f_cov = self.cal_covariance(f2)
            # print(f_cov.shape)
            # import pdb; pdb.set_trace()
            # out.append(self.classifier_cova(x))
            f_cov = f_cov.contiguous()
            # print(f_cov.shape)
            # 使用 view 重新形状为 [6, 36]
            f_cov = f_cov.view(x.shape[0], 36)
            # print(f_cov.shape)
            out.append(f_cov)
            # import pdb; pdb.set_trace()
        if self.use_Asoftmax:
            if last_cont is None:
                x_size = x.size(0)
                out.append(self.Aclassifier(x[0:x_size:2]))
            else:
                last_x = self.model(last_cont)
                last_x = self.avgpool(last_x)
                last_x = last_x.view(last_x.size(0), -1)
                out.append(self.Aclassifier(last_x))

        return out
    

    def cal_covariance(self, input):
    # 输入形状: [l, p, c, h, w] = [6, 4, 3, 6, 6]
        # print("Input shape:", input.shape)
        l, p, c, h, w = input.shape
        # 初始化协方差矩阵列表
        covariance_matrices_list = []
        # 遍历每个图像
        for i in range(l):
            # 针对当前图像，存储所有patch的协方巧矩阵
            image_covariances = []
            # 遍历每个patch
            for j in range(p):
                # 获取当前patch
                img = input[i, j]  # Shape: [c, h, w]
                # 调整形状以便计算协方巧
                img = img.reshape(c, -1)  # Shape: [c, h*w]
                # 计算均值并去中心化
                mean = img.mean(dim=1, keepdim=True)  # Shape: [c, 1]
                img = img - mean
                # 计算协方巧矩阵
                covariance_matrix = torch.mm(img, img.t())  # Shape: [c, c]
                covariance_matrix /= (h * w - 1)
                # 添加到当前图像的patch列表中
                image_covariances.append(covariance_matrix)
            # 将当前图像的所有patch协方巧矩阵转换为张量并添加到总列表中
            covariance_matrices_list.append(torch.stack(image_covariances))
        # 将所有图像的协方巧矩阵堆叠成一个张量
        return torch.stack(covariance_matrices_list)  # 最终形状: [6, 4, 3, 3]
    
    def crop_feature(self, image_tensor, cropnum = [2,2]):
        L, C, H, W = image_tensor.shape
        crop_h = H // cropnum[1]
        crop_w = W // cropnum[0]
        
        # 生成裁剪后的图像块
        cropped_images = []
        for j in range(cropnum[1]):  # 垂直方向
            for i in range(cropnum[0]):  # 水平方向
                # 计算当前块的起始和结束索引
                start_h = j * crop_h
                end_h = start_h + crop_h
                start_w = i * crop_w
                end_w = start_w + crop_w
                
                # 切片提取图像块
                # 保证不超出原图边界，使用 min 函数
                cropped = image_tensor[:, :, start_h:min(end_h, H), start_w:min(end_w, W)]
                cropped_images.append(cropped)
                cropped_images_tensor = torch.stack(cropped_images).transpose(0, 1)
        return cropped_images_tensor


