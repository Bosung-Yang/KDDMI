# -*- coding: utf-8 -*-
import time
import torch
import numpy as np
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import math, evolve

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class VIB(nn.Module):
    def __init__(self, latent_dim):
        super(VIB, self).__init__()
        self.mu = nn.Linear(1000, latent_dim)
        self.log_var = nn.Linear(1000, latent_dim)
        
    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)
        std = torch.exp(0.5 + log_var)
        eps = torch.randn_like(std)
        z = mu + std*eps
        kl_div = -0.5 + torch.sum(\
            1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return z, kl_div

class MY_VGGVIB(nn.Module) :
    def __init__(self, n_classes):
        super(MY_VGGVIB, self).__init__()
        self.model = torchvision.models.vgg16_bn(pretrained=True)
        self.fc1 = nn.Linear(1000,1000)
        self.fc2 = nn.Linear(128,1000)
        self.vib = VIB(128)
        
    
    def forward(self,x):
        x_latent = self.model(x)
        #fc1 = self.fc1(x)
        #z, kl_div = self.vib(fc1)
        #x_latent = self.fc2(z)
        return x_latent

class VGG16_vanilla(nn.Module):
    def __init__(self, n_classes):
        super(VGG16_vanilla, self).__init__()
        model = torchvision.models.vgg16(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        
        return [feature, res]
     
class VGG16(nn.Module):
    def __init__(self, n_classes):
        super(VGG16, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        
        return [feature, res]
    
class VGG16_ReLU(nn.Module):
    def __init__(self, n_classes):
        super(VGG16_ReLU, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        res = F.relu(res)
        
        return [feature, res]


class Resnet50_simple(nn.Module):
    def __init__(self, n_classes):
        super(Resnet50_simple, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(1000)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(1000, self.n_classes)
        self.output_layer = nn.Sequential(nn.BatchNorm1d(1000),
                        nn.Dropout(),
                        Flatten(),
                        nn.Linear(1000, 1000),
                        nn.BatchNorm1d(1000)) 
        
            
    def forward(self, x):
        feature = self.model(x)
        res = self.fc_layer(feature)

        return [feature, res]
      
class Resnet50(nn.Module):
    def __init__(self, n_classes):
        super(Resnet50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(1000)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(1000, self.n_classes)
        self.output_layer = nn.Sequential(nn.BatchNorm1d(1000),
                        nn.Dropout(),
                        Flatten(),
                        nn.Linear(1000, 1000),
                        nn.BatchNorm1d(1000)) 
        
            
    def forward(self, x):
        feature = self.model(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.output_layer(feature)
        feature = self.bn(feature)
        res = self.fc_layer(feature)

        return [feature, res]

class Resnet50_softmax(nn.Module):
    def __init__(self, n_classes):
        super(Resnet50_softmax, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(1000)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(1000, self.n_classes)
        self.output_layer = nn.Sequential(nn.BatchNorm1d(1000),
                        nn.Dropout(),
                        Flatten(),
                        nn.Linear(1000, 1000),
                        nn.BatchNorm1d(1000)) 
        
            
    def forward(self, x):
        feature = self.model(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.output_layer(feature)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        res = F.softmax(res,dim=1)

        return [feature, res]


    
class VGG16_Sigmoid(nn.Module):
    def __init__(self, n_classes):
        super(VGG16_Sigmoid, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        res = F.sigmoid(res)
        
        return [feature, res]

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return out
    
    def deactivate(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        
        return [feature, res]
    
class VGG16_VirtualSoftmax_2000(nn.Module):
    def __init__(self, n_classes):
        super(VGG16_VirtualSoftmax_2000, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, 2000)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        res = F.softmax(res,dim=1)
        
        return [feature, res]

    def deactivate(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)

        return [feature, res]
    

class VGG16_VirtualSoftmax_1024(nn.Module):
    def __init__(self, n_classes):
        super(VGG16_VirtualSoftmax_1024, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        res = F.softmax(res,dim=1)
        
        return [feature, res]

    def deactivate(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        return [feature, res]
    
class VGG16_Softmax(nn.Module):
    def __init__(self, n_classes):
        super(VGG16_Softmax, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, 1000)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        res = F.softmax(res,dim=1)
        
        return [feature, res]

    def deactivate(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        
        return [feature, res]
    
class VGG16_vib(nn.Module):
    def __init__(self, n_classes):
        super(VGG16_vib, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.k = 1000
        self.n_classes = n_classes
        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Linear(self.k, self.n_classes)
            
    def forward(self, x, mode="train"):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
       
        return [feature, out, mu, std]
    
    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
       
        return out


    
class VGG16_vib_softmax(nn.Module):
    def __init__(self, n_classes):
        super(VGG16_vib_softmax, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.k = 1000
        self.n_classes = n_classes
        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Linear(self.k, self.n_classes)
            
    def forward(self, x, mode="train"):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1) 
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        #out = F.softmax(out,dim=1)
       
        return [feature, out, mu, std]
    
    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
       
        return out

class CrossEntropyLoss(_Loss):
    def forward(self, out, gt, mode="reg"):
        bs = out.size(0)
        loss = - torch.mul(gt.float(), torch.log(out.float() + 1e-7))
        if mode == "dp":
            loss = torch.sum(loss, dim=1).view(-1)
        else:
            loss = torch.sum(loss) / bs
        return loss

class BinaryLoss(_Loss):
    def forward(self, out, gt):
        bs = out.size(0)
        loss = - (gt * torch.log(out.float()+1e-7) + (1-gt) * torch.log(1-out.float()+1e-7))
        loss = torch.mean(loss)
        return loss


class FaceNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(FaceNet, self).__init__()
        self.feature = evolve.IR_50_112((112, 112))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def predict(self, x):
        feat = self.feature(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return out
            
    def forward(self, x):
        # print("input shape:", x.shape)
        # import pdb; pdb.set_trace()
        
        feat = self.feature(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return [feat, out]

class FaceNet64(nn.Module):
    def __init__(self, num_classes = 1000):
        super(FaceNet64, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)
        return feat, out
    
class FaceNet64_(nn.Module):
    def __init__(self, num_classes = 1000):
        super(FaceNet64_, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        feat = self.feature(x)
        #feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        #out = self.fc_layer(feat)
        #__, iden = torch.max(out, dim=1)
        #iden = iden.view(-1, 1)
        return feat, out
    
class FaceNet64_softmax(nn.Module):
    def __init__(self, num_classes = 1000):
        super(FaceNet64_softmax, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        feat = self.feature(x)
        #feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        out = F.softmax(out,dim=1)
        #__, iden = torch.max(out, dim=1)
        #iden = iden.view(-1, 1)
        
        return feat, out

class IR152(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR152, self).__init__()
        self.feature = evolve.IR_152_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return feat, out

class IR152_softmax(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR152_softmax, self).__init__()
        self.feature = evolve.IR_152_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        out = F.softmax(out,dim=1)
        return feat, out

class IR152_vs1024(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR152_vs1024, self).__init__()
        self.feature = evolve.IR_152_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.fc_layer = nn.Linear(self.feat_dim,self.num_classes)
            
    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        out = F.softmax(out,dim=1)
        return feat, out

class IR152_vib(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR152_vib, self).__init__()
        self.feature = evolve.IR_152_64((64, 64))
        self.feat_dim = 512
        self.k = self.feat_dim // 2
        self.n_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.k, self.n_classes),
            nn.Softmax(dim = 1))

    def forward(self, x, mode='train'):
        feature = self.output_layer(self.feature(x))
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)

        return feature, out, mu, std

class FaceNet64_vib(nn.Module):
    def __init__(self, num_classes = 1000):
        super(FaceNet64_vib, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.k = 1000
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  
        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Linear(self.k, self.num_classes)
            
    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        
        statis = self.st_layer(feat)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)
        return feat, out, mu, std
    
class IR50(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR50, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.k, self.n_classes),
            nn.Softmax(dim = 1))

    def forward(self, x):
        feature = self.output_layer(self.feature(x))
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)

        return feature, out, iden, mu, std

class IR50_vib(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR50_vib, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.n_classes = num_classes
        self.k = self.feat_dim // 2
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.k, self.n_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        feat = self.output_layer(self.feature(x))
        feat = feat.view(feat.size(0), -1)
        statis = self.st_layer(feat)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)

        return feat, out, iden, mu, std


