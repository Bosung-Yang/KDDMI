import torch, os, classify, sys
import numpy as np 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from torchvision import transforms, datasets
import time
import torch.nn.functional as F
import torch
import cv2 as cv
from torchvision import transforms, datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import misc
import scipy
import util
import random

def train_vib(img, label, model, teacher, encoder, optimizer, mode):
    lossfn = torch.nn.CrossEntropyLoss()
    model.train()
    teacher.eval()
    optimizer.zero_grad()
    
    if 'kd' in mode:
        if 'enc' in mode:
            img = encoder(img)
        if 'dct' in mode:
            idct = low_frequency(img,img.shape)
            img = torch.tensor(idct).to('cuda').float()
        soft_label = teacher(img)[1].to(device)
        y_pred = model(img)[-1].to(device)
        loss = distillation(y_pred, label, soft_label, T=8, alpha= 0.2)
    else:
        if 'enc' in mode:
            img = encoder(img)
        if 'dct' in mode:
            idct = low_frequency(img,img.shape)
            img = torch.tensor(idct).to('cuda').float()
        ___, y_pred, mu, std = model(img, "train")
        y_pred = y_pred.cuda()
        info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean() # kl_div = -0.5 + torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        cross_loss = lossfn(y_pred, label)
        loss = cross_loss  + 0.01 * info_loss
        #print(cross_loss)
    loss.backward()
    optimizer.step()
    
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def distillation(y, labels, teacher_scores, T, alpha):
    # distillation loss + classification loss
    # y: student
    # labels: hard label
    # teacher_scores: soft label
    return nn.KLDivLoss()(F.log_softmax(y/T), F.softmax(teacher_scores/T)) * (T*T * 2.0 + alpha)  + 0.01 * F.cross_entropy(y,labels) * (1.-alpha) + 0.01*nn.MSELoss()(y,teacher_scores)
    #return nn.MSELoss()(y,teacher_scores)

# Vanilla - None, DCT, Enc
# KD - None, DCT, Enc
def train(img, label, model, teacher, encoder, optimizer, mode, temp):
    lossfn = torch.nn.CrossEntropyLoss()
    model.train()
    teacher.eval()
    optimizer.zero_grad()
    
    if 'kd' in mode:
        if 'enc' in mode:
            img = encoder(img)
        if 'dct' in mode:
            idct = low_frequency(img,img.shape) 
            img = torch.tensor(idct).to('cuda').float()
        soft_label = teacher(img)[1].to(device)
        y_pred = model(img)[1].to(device)
        loss = distillation(y_pred, label, soft_label, T=temp, alpha= 0.2)
    else:
        if 'enc' in mode:
            img = encoder(img)
        if 'dct' in mode:
            idct = low_frequency(img,img.shape)
            img = torch.tensor(idct).to('cuda').float()
        y_pred = model(img)[1].to(device)
        loss = lossfn(y_pred, label)
    loss.backward()
    optimizer.step()

if __name__ == '__main__':
    parser = ArgumentParser(description='Step1 : Training Face Classifiers')
    parser.add_argument('--gpu')
    parser.add_argument('--epoch',default=100)
    parser.add_argument('--mode',default='none')
    parser.add_argument('--student',default='vgg16')
    parser.add_argument('--teacher',default='vgg16')
    parser.add_argument('--teacher_path',default='./final_pths/vgg16.pth')
    parser.add_argument('--save_name',default='./model.pth')
    parser.add_argument('--temp',default=64,type=float)
    parser.add_argument('--num_class',type=int)
    parser.add_argument('--seed', type=int, default=1)
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataloader
    data_path = './data/'
    batch_size = 64
    train_folder = 'train/'
    test_folder = 'test/'
    image_transforms = {
        'train': transforms.Compose([
            transforms.CenterCrop((128,128)),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((128,128)),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
    }
    set_seed(args.seed)

    train_images = datasets.ImageFolder(data_path+train_folder,image_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images, batch_size = 64 ,num_workers=4,shuffle=True)
    test_images = datasets.ImageFolder(data_path+test_folder,image_transforms['test'])
    test_loader = torch.utils.data.DataLoader(test_images, batch_size = 64 ,num_workers=4,shuffle=True) 

    # model
    student = util.get_model(args.student, args.num_class)
    student = student.cuda()

    teacher = util.get_model(args.teacher, args.num_class)
    #teacher.load_state_dict(torch.load(args.teacher_path))
    teacher = teacher.cuda()
    

    enc=None
    #optim = torch.optim.Adam(student.parameters(), lr = 0.0001)
    lr = 0.0001
    optim = torch.optim.Adam(params=student.parameters(),
							    lr=lr)
    nepoch = int(args.epoch)
    best_score = 0
    
    for e in range(nepoch):
        for i, data in enumerate(train_loader):
            img, label = data
            img = img.to(device)
            label = label.to(device)
            if 'vib' in args.student:
                train_vib(img = img, label = label, model = student, teacher = teacher, encoder = enc, optimizer = optim, mode = args.mode)
            else:
                train(img = img, label = label, model = student, teacher = teacher, encoder = enc, optimizer = optim, mode = args.mode, temp = args.temp)
        answer = 0
        total = 0
        dct_answer = 0
        
        for i, data in enumerate(test_loader):
            student.eval()
            img, label = data
            img = img.to(device)
            label = label.to(device)
            y_pred = student(img)[1].to(device)
            _, p_label = y_pred.max(1)
            total += label.size(0)
            answer += (p_label == label).sum().float()
            
        if answer/total > best_score:
            best_score = answer/total
            save_model_name = args.save_name
            torch.save(student.state_dict(),save_model_name)
    
    print(args.save_name,args.student,' best accuracy : ',best_score)
