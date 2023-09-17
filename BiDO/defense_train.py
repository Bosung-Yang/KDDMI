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


def HSIC(args, trainloader, testloader):
    n_classes = 1000
    hp_list = [
                 (0.05,0.2) ,(0.05,0.5), (0.1,0.1), (0.05,0.3), (0.3,0.05)
    ]

    criterion = nn.CrossEntropyLoss().cuda()

    for i, (a1, a2) in enumerate(hp_list):
        print("a1:", a1, "a2:", a2)

        if model_name == "VGG16" or model_name == "reg":
            net = model.VGG16(n_classes,hsic_training=True)

            load_pretrained_feature_extractor = True
            if load_pretrained_feature_extractor:
                print('use')
                pretrained_model_ckpt = "./model.pth"
                checkpoint = torch.load(pretrained_model_ckpt)
                load_feature_extractor(net,checkpoint)
                #net.load_state_dict(checkpoint['state_dict'],strict=False)

        elif model_name == "ResNet":
            net = model.ResNetCls(nclass=n_classes, resnetl=10)
            # net = model.ResNet18(n_classes=n_classes)

        elif model_name == "MCNN":
            net = model.MCNN(n_classes)
        elif model_name == "LeNet":
            net = model.LeNet3(n_classes)

        elif model_name == "SimpleCNN":
            net = model.Classifier(1, 128, n_classes)

        optimizer = torch.optim.Adam(net.parameters(), lr)

        net = torch.nn.DataParallel(net).to(device)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35], gamma=0.5)

        best_ACC = -1
        for epoch in range(n_epochs):
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
            train_loss, train_acc = engine.train_HSIC(net, criterion, optimizer, trainloader, a1, a2, n_classes,
                                                      ktype='linear',
                                                      hsic_training=True)
            test_loss, test_acc = engine.test_HSIC(net, criterion, testloader, a1, a2, n_classes, ktype='linear',
                                                   hsic_training=True)

            if test_acc > best_ACC:
                best_ACC = test_acc
                best_model = deepcopy(net)
            scheduler.step()

        print("best acc:", best_ACC)
        #mlflow.log_metric("accuracy", best_ACC)
        utils.save_checkpoint({
            'state_dict': best_model.state_dict(),
        }, '../GMI/', "{}_{:.3f}_{:.3f}_{:.2f}.tar".format(model_name, a1, a2, best_ACC))

def VIB(args,trainloader, testloader):

    model_name = "VGG16_vib"
    n_epochs = 50
    weight_decay = 1e-4
    momentum = 0.9
    lr = 0.01
    milestones = [20,35]
    n_classes = 1000
    if model_name == "VGG16_vib":
        net = model.VGG16_vib(n_classes)

    elif model_name == "ResNet":
        net = model.PretrainedResNet(nc=1, nclass=n_classes, imagesize=128)

    elif model_name == "MCNN":
        net = model.MCNN_vib(n_classes)

    optimizer = torch.optim.SGD(params=net.parameters(),
                                lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    # scheduler = MultiStepLR(optimizer, milestones, gamma=0.1)

    criterion = nn.CrossEntropyLoss().cuda()
    net = torch.nn.DataParallel(net).to(device)

    best_ACC = -1

    for epoch in range(n_epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, lr))
        train_loss, train_acc = engine.train_vib(net, criterion, optimizer, trainloader, 0.01)
        test_loss, test_acc = engine.test_vib(net, criterion, testloader, 0.01)

        if test_acc > best_ACC:
            best_ACC = test_acc
            best_model = deepcopy(net)

        # scheduler.step()

    print("best acc:", best_ACC)
    #malflow.log_metric("accuracy", best_ACC)
    utils.save_checkpoint({
        'state_dict': best_model.state_dict(),
    }, '../GMI', "{}_beta{:.3f}_{:.2f}.tar".format(model_name, 0.01, best_ACC))

def NODEF(args, n_classes, trainloader, testloader):
    model_name = 'VGG16'
    n_epochs = 50
    lr = 0.0001

    criterion = nn.CrossEntropyLoss().cuda()


    if model_name == "VGG16" or model_name == "reg":
        net = model.VGG16_V(n_classes)

    elif model_name == "ResNet":
        net = model.ResNetCls(nclass=n_classes, resnetl=10)

    optimizer = torch.optim.Adam(net.parameters(), lr)

    net = torch.nn.DataParallel(net).to(device)

    best_ACC = -1
    for epoch in range(n_epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
        train_loss, train_acc = engine.train(net, criterion, optimizer, trainloader)
        test_acc = engine.test(net, criterion, testloader)
        if test_acc > best_ACC:
            best_ACC = test_acc
            best_model = deepcopy(net)
        print(test_acc)

    print("best acc:", best_ACC)
    mlflow.log_metric("accuracy", best_ACC)
    utils.save_checkpoint({
        'state_dict': best_model.state_dict(),
    }, '../GMI/', "VGG16.tar")


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
            e_path = '../GMI/VGG16.tar'
        elif args.teacher == 'HSIC':
            teacher = model.VGG16(n_classes,True)
            e_path = '../GMI/VGG16_0.050_0.200_68.20.tar'
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
            }, '../GMI', "KD_lastest.tar")

def Multiple_KD(args, n_classes, trainloader, testloader):
    n_epochs = 100
    lr = 0.001

    vgg_teacher = model.VGG16_V(n_classes)
    vgg_path = '../final_tars/VGG16.tar'
    ckp_E = torch.load(vgg_path)
    vgg_teacher.load_state_dict(ckp_E['state_dict'], strict=False)
    vgg_teacher = vgg_teacher.cuda()

    hsic_teacher = model.VGG16(n_classes, True)
    hsic_path = '../final_tars/HSIC.tar'
    ckp_E = torch.load(hsic_path)
    hsic_teacher.load_state_dict(ckp_E['state_dict'], strict=False)
    hsic_teacher = hsic_teacher.cuda()
    lossfns = [mkd]
    for loss in lossfns:
        criterion = loss
        net = model.VGG16_V(n_classes)
        optimizer = torch.optim.Adam(net.parameters(), lr)
        net = torch.nn.DataParallel(net).to(device)
        best_ACC = -1
        for epoch in range(n_epochs):
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
            train_loss, train_acc = engine.train_kd(net,vgg_teacher,hsic_teacher, criterion, optimizer, trainloader)
            test_acc = engine.test(net, criterion, testloader)
            print(test_acc)
            if test_acc > best_ACC:
                best_ACC = test_acc
                best_model = deepcopy(net)
            #scheduler.step()

        print("best acc:", best_ACC)
        
        utils.save_checkpoint({
            'state_dict': best_model.state_dict(),
            }, '../final_tars/', "MKD.tar")
        
if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Defense against MI')
    parser.add_argument('--dataset', default='celeba')
    parser.add_argument('--defense')
    parser.add_argument('--root_path', default='/workspace/data/', help='')
    parser.add_argument('--model_dir', default='./target_model', help='')
    parser.add_argument('--nclass', type=int , default=1000)
    parser.add_argument('--model',type=str, default = 'VGG16')
    parser.add_argument('--teacher', type=str, default='VGG16')

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

