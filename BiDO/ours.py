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

def distillation(student_scores, labels, teacher_scores):
    # distillation loss + classification loss
    # y: student
    # labels: hard label
    # teacher_scores: soft label
    #teacher_scores = F.softmax(teacher_scores)
    T = 2
    return F.cross_entropy(student_scores,labels) + nn.KLDivLoss()(F.log_softmax(student_scores/T), F.softmax(teacher_scores/T)) * (T*T * 2.0 + 0.7)
    #10* nn.MSELoss()(student_scores,teacher_scores)
    #return nn.MSELoss()(y,teacher_scores)

def mkd(student_scores, labels, vgg_output, hsic_output):
    # distillation loss + classification loss
    # y: student
    # labels: hard label
    # teacher_scores: soft label
    #teacher_scores = F.softmax(teacher_scores)
    T=64
    return F.cross_entropy(student_scores,labels) +100* nn.MSELoss()(student_scores,hsic_output) +  nn.KLDivLoss()(F.log_softmax(student_scores/T), F.softmax(vgg_output/T)) * (T*T * 2.0 + 0.7)
    #return nn.MSELoss()(y,teacher_scores)

def KD(args, n_classes, trainloader, testloader):
    n_epochs = 100
    lr = 0.001

    lossfns = [distillation]
    for loss in lossfns:
        criterion = loss
        if model_name == "VGG16" or model_name == "reg":
            net = model.VGG16_V(n_classes)

        elif model_name == "ResNet":
            net = model.ResNetCls(nclass=n_classes, resnetl=10)
            # net = model.ResNet18(n_classes=n_classes)

        optimizer = torch.optim.Adam(net.parameters(), lr)

        net = torch.nn.DataParallel(net).to(device)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
        if args.teacher=='VGG16':
            teacher = model.VGG16_V(n_classes)
            e_path = '../final_tars/eval/VGG16_79.23.tar'
        elif args.teacher == 'HSIC':
            teacher = model.VGG16(n_classes,True)
            e_path = '../final_tars/BiDO_teacher_71.35_0.1_0.1.tar'
        ckp_E = torch.load(e_path)
        teacher.load_state_dict(ckp_E['state_dict'], strict=False)
        teacher = teacher.cuda()

        best_ACC = -1
        for epoch in range(n_epochs):
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
            train_loss, train_acc = engine.train_kd(net,teacher, criterion, optimizer, trainloader)
            test_acc = engine.test(net, criterion, testloader)
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
 

    if args.defense == 'HSIC':
        HSIC(args, train_loader, test_loader)
    if args.defense == 'VIB':
        VIB(args, train_loader , test_loader)
    if args.defense=='NODEF':
        NODEF(args, 1000, train_loader,test_loader)
    if args.defense=='KD':
        KD(args, args.nclass, train_loader, test_loader)

