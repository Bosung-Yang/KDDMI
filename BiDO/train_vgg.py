import torch, os, sys
import numpy as np 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from torchvision import transforms, datasets
import time
import torch.nn.functional as F
import torch
from torchvision import transforms, datasets
import torch.nn.functional as F
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
import mlflow.pytorch
from mlflow.models import infer_signature
import mlflow
from torch.optim.lr_scheduler import MultiStepLR
from copy import deepcopy
import collections
from util import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import utils
import model
import engine
device = "cuda"

def load_my_state_dict(net, state_dict):
    print("load nature model!!!")
    net_state = net.state_dict()
    for ((name, param), (old_name, old_param),) in zip(net_state.items(), state_dict.items()):
        # print(name, '---', old_name)
        net_state[name].copy_(old_param.data)


def load_feature_extractor(net, state_dict):
    print("load_pretrained_feature_extractor!!!")
    net_state = net.state_dict()

    new_state_dict = collections.OrderedDict()
    for name, param in state_dict.items():
        if "running_var" in name:
            new_state_dict[name] = param
            new_item = name.replace("running_var", "num_batches_tracked")
            new_state_dict[new_item] = torch.tensor(0)
        else:
            new_state_dict[name] = param

    for ((name, param), (new_name, mew_param)) in zip(net_state.items(), new_state_dict.items()):
        if "classifier" in new_name:
            break
        if "num_batches_tracked" in new_name:
            continue
        # print(name, '---', new_name)
        net_state[name].copy_(mew_param.data)

def main(args, loaded_args, trainloader, testloader):
    n_classes = 1000
    model_name = loaded_args["dataset"]["model_name"]
    weight_decay = loaded_args[model_name]["weight_decay"]
    momentum = loaded_args[model_name]["momentum"]
    n_epochs = loaded_args[model_name]["epochs"]
    lr = 0.0001
    milestones = loaded_args[model_name]["adjust_epochs"]

    hp_list = [
        (0.1, 0.5)
    ]

    criterion = nn.CrossEntropyLoss().cuda()


    if model_name == "VGG16" or model_name == "reg":
        net = model.VGG16_VS(n_classes)

        load_pretrained_feature_extractor = False
        if load_pretrained_feature_extractor:
            pretrained_model_ckpt = "/workspace/data/vgg.pth"
            checkpoint = torch.load(pretrained_model_ckpt)
            load_feature_extractor(net, checkpoint)

    elif model_name == "ResNet":
        net = model.ResNetCls(nclass=n_classes, resnetl=10)
        # net = model.ResNet18(n_classes=n_classes)

    optimizer = torch.optim.Adam(net.parameters(), lr)

    net = torch.nn.DataParallel(net).to(device)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    print('ah?')
    best_ACC = -1
    for epoch in range(n_epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
        train_loss, train_acc = engine.train(net, criterion, optimizer, trainloader)
        test_acc = engine.test(net, criterion, testloader)
        if test_acc > best_ACC:
            best_ACC = test_acc
            best_model = deepcopy(net)
        #scheduler.step()

    print("best acc:", best_ACC)
    mlflow.log_metric("accuracy", best_ACC)
    utils.save_checkpoint({
        'state_dict': best_model.state_dict(),
    }, model_path, "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, 0, 0, best_ACC))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='train with BiDO')
    parser.add_argument('--dataset', default='celeba', help='celeba | mnist | cifar')
    parser.add_argument('--measure', default='Vanilla', help='HSIC | COCO')
    parser.add_argument('--ktype', default='linear', help='gaussian, linear, IMQ')
    parser.add_argument('--hsic_training', default=True, help='multi-layer constraints', type=bool)
    parser.add_argument('--root_path', default='./', help='')
    parser.add_argument('--config_dir', default='./config', help='')
    parser.add_argument('--model_dir', default='./target_model', help='')
    args = parser.parse_args()

    model_path = os.path.join(args.root_path, args.model_dir, args.dataset, 'Vanilla')
    os.makedirs(model_path, exist_ok=True)

    file = os.path.join(args.config_dir, args.dataset + ".json")

    loaded_args = utils.load_json(json_file=file)
    model_name = loaded_args["dataset"]["model_name"]


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
    test_loader = torch.utils.data.DataLoader(test_images, batch_size = 128,num_workers=4,shuffle=True) 
 


    main(args, loaded_args, train_loader, test_loader)
