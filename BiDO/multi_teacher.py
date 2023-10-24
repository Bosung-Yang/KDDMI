import torch, os, engine, model, utils, sys
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from copy import deepcopy
import numpy as np
import collections
from torchvision import transforms, datasets
import model
from util import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
device = "cuda"
import torch.nn.functional as F
from tqdm import tqdm

def load_my_state_dict(net, state_dict):
    print("load nature model!!!")
    net_state = net.state_dict()
    for ((name, param), (old_name, old_param),) in zip(net_state.items(), state_dict.items()):
        # print(name, '---', old_name)
        net_state[name].copy_(old_param.data)

def KD(args, n_classes, trainloader, testloader):
    n_epochs = 100
    lr = 0.001

    
    if model_name == "VGG16" or model_name == "reg":
        net = model.VGG16_V(n_classes)

    elif model_name == "ResNet":
        net = model.ResNetCls(nclass=n_classes, resnetl=10)
        # net = model.ResNet18(n_classes=n_classes)

    optimizer = torch.optim.Adam(net.parameters(), lr)

    net = torch.nn.DataParallel(net).to(device)

    vanilla_teacher = model.VGG16_V(1000)
    e_path = '../final_tars/eval/VGG16_79.23.tar'
    ckp_E = torch.load(e_path)
    vanilla_teacher.load_state_dict(ckp_E['state_dict'], strict=False)
    vanilla_teacher = vanilla_teacher.cuda()
    
    HSIC_teacher =model.VGG16(n_classes,True)
    e_path = '../final_tars/BiDO_teacher_71.35_0.1_0.1.tar'
    ckp_E = torch.load(e_path)
    HSIC_teacher.load_state_dict(ckp_E['state_dict'], strict=False)
    HSIC_teacher = HSIC_teacher.cuda()

    best_ACC = -1
    for epoch in range(n_epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=100)
        for batch_idx, (inputs, iden) in pbar:
            net.train()
            optimizer.zero_grad()
            inputs, iden = inputs.to(device), iden.to(device)
            iden = iden.view(-1)
            feats, out_logit = net(inputs)
            _, vt_output = vanilla_teacher(inputs)
            _, ht_output = HSIC_teacher(inputs)
            cls_loss = F.cross_entropy(out_logit, iden)
            #print(out_logit)
            #print(vt_output)
            #vt_loss = nn.KLDivLoss(out_logit, vt_output)
            vt_loss = nn.MSELoss()(out_logit, vt_output)
            ht_loss = nn.MSELoss()(out_logit, ht_output)
            loss = cls_loss #+ 0.01 * vt_loss + 0.001 * ht_loss
            loss.backward()
            optimizer.step()
        #train_loss, train_acc = engine.train_kd(net,teacher, criterion, optimizer, trainloader)
        test_acc = engine.test(net, F.cross_entropy, testloader)
        print(test_acc)
        if test_acc > best_ACC:
            best_ACC = test_acc
            best_model = deepcopy(net)
        #scheduler.step()

    print("best acc:", best_ACC)
    
    utils.save_checkpoint({
        'state_dict': best_model.state_dict(),
        }, '../final_tars', "student-BiDO_{:.2f}.tar".format(best_ACC))


        
if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Defense against MI')
    parser.add_argument('--dataset', default='celeba')
    parser.add_argument('--defense', default = 'KD')
    parser.add_argument('--root_path', default='/workspace/data/', help='')
    parser.add_argument('--model_dir', default='./target_model', help='')
    parser.add_argument('--nclass', type=int , default=1000)
    parser.add_argument('--model',type=str, default = 'VGG16')
    parser.add_argument('--teacher', type=str, default='HSIC')

    args = parser.parse_args()
    

    model_name = args.model
    weight_decay = 1e-4
    momentum = 0.9
    n_epochs = 50
    lr = 1e-4
    milestones = 60

    model_path = os.path.join(args.root_path, args.model_dir, args.dataset, args.defense)
    os.makedirs(model_path, exist_ok=True)

    model_name = 'VGG16'

    data_path = '/workspace/data/'
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
    train_images = datasets.ImageFolder(data_path+train_folder,image_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images, batch_size = 64 ,num_workers=4,shuffle=True)
    test_images = datasets.ImageFolder(data_path+test_folder,image_transforms['test'])
    test_loader = torch.utils.data.DataLoader(test_images, batch_size = 64 ,num_workers=4,shuffle=True) 
 
    KD(args, 1000, train_loader, test_loader)
    
