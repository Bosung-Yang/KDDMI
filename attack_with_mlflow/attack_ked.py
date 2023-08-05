from losses import completion_network_loss, noise_loss
from utils import *
from classify import *
from generator import *
from discri import *
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss, DataParallel
from torchvision.utils import save_image
from torch.autograd import grad
import torchvision.transforms as transforms
import torch
import time
import random
import os, logging
import numpy as np
from attack import inversion, dist_inversion
from generator import Generator
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import classify
import classify

def get_model(architecture,num_class):
    if architecture == 'linear':
        model = classify.VGG16(num_class)
    if architecture == 'softmax':
        model = classify.VGG16_VirtualSoftmax_1024(num_class)
    if architecture =='vgg16v':
        model = classify.VGG16_vanilla(1000)
    if architecture == 'vgg16':
        model = classify.VGG16(1000)
    elif architecture == 'vggvc':
        model = classify.VGG16(num_class)
    elif architecture == 'vgg16_softmax':
        model = classify.VGG16_Softmax(1000)
    elif architecture == 'vgg16_relu':
        model = classify.VGG16_ReLU(1000)
    elif architecture == 'vgg16_vs2000':
        model = classify.VGG16_VirtualSoftmax_2000(1000)
    elif architecture == 'vgg_sigmoid':
        model = classify.VGG16_Sigmoid(1000)
    elif architecture == 'vgg_virtualsoftmax_1024':
        model = classify.VGG16_VirtualSoftmax_1024(num_class)
    elif architecture == 'vgg_vs':
        model = classify.VGG16_VirtualSoftmax_1024(num_class)
    elif architecture == 'vgg16_vib':
        model = classify.VGG16_vib(1000)
    elif architecture=='vgg_vib_softmax':
        model = classify.VGG16_vib_softmax(1000)
    elif architecture=='resnet':
        model = classify.Resnet50(num_class)
    elif architecture == 'resnet_softmax':
        model = classify.Resnet50_softmax(num_class)
    elif architecture == 'resnet_vs':
        model = classify.IR152_vs1024(1024)
    elif architecture == 'resnet_vib':
        model = classify.IR152_vib(1000)
    elif architecture == 'facenet':
        model = classify.FaceNet64(1000)
    elif architecture == 'facenet_softmax':
        model = classify.FaceNet64_softmax(1000)
    elif architecture == 'facenet_vs':
        model = classify.FaceNet64_softmax(1024)
    elif architecture == 'facenet_vib':
        model = classify.FaceNet64_vib(1000)
    return model


# logger
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger



if __name__ == "__main__":
    global args, logger

    parser = ArgumentParser(description='Step2: targeted recovery')
    parser.add_argument('--model', default='VGG16', help='VGG16 | IR152 | FaceNet64')
    parser.add_argument('--device', type=str, default='4,5,6,7', help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--improved_flag', action='store_true', default=True, help='use improved k+1 GAN')
    parser.add_argument('--dist_flag', action='store_true', default=True, help='use distributional recovery')
    parser.add_argument('--path')
    parser.add_argument('--exp')
    parser.add_argument('--num_class',type=int,default=1000)
    args = parser.parse_args()
    logger = get_logger()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    z_dim = 100
    ###########################################
    ###########     load model       ##########
    ###########################################
    G = Generator(z_dim)
    G = torch.nn.DataParallel(G).cuda()
    if args.improved_flag == True:
        D = MinibatchDiscriminator()
        path_G = '../checkpoint/improved_celeba_G.tar'
        path_D = '../checkpoint/improved_celeba_D.tar'
    else:
        D = DGWGAN(3)
        path_G = '../checkpoint/celeba_G.tar'
        path_D = '../checkpoint/celeba_D.tar'
    
    D = torch.nn.DataParallel(D).cuda()
    ckp_G = torch.load(path_G)
    G.load_state_dict(ckp_G['state_dict'], strict=False)
    ckp_D = torch.load(path_D)
    D.load_state_dict(ckp_D['state_dict'], strict=False)

    T = get_model(args.model,args.num_class)
    path_T = args.path

    
    #T = torch.nn.DataParallel(T).cuda()
    
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T, strict=False)
    T=T.cuda()
    #E = FaceNet64(1000)
    #E = torch.nn.DataParallel(E).cuda()
    #path_E = '../checkpoint/FaceNet64_88.50.tar'
    #ckp_E = torch.load(path_E)
    #E.load_state_dict(ckp_E['state_dict'], strict=False)
    E = T
    ############         attack     ###########

    aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
    iter_times = 3000
    output_acc_list = np.zeros((iter_times))
    output_acc5_list = np.zeros((iter_times))
    for i in range(1):
        iden = torch.from_numpy(np.arange(50))

        # evaluate on the first 300 identities only
        for idx in range(6):
            #print("--------------------- Attack batch [%s]------------------------------" % idx)
            if args.dist_flag == True:
                acc, acc_5, acc_var, acc_var5, acc_list, acc5_list = dist_inversion(G, D, T, E, iden, itr=i, lr=2e-2, momentum=0.9, lamda=100, iter_times=iter_times, clip_range=1, improved=args.improved_flag, num_seeds=1, exp_name=args.exp)
            else:
                acc, acc_5, acc_var, acc_var5, acc_list, acc5_list = inversion(G, D, T, E, iden, itr=i, lr=2e-2, momentum=0.9, lamda=100, iter_times=iter_times, clip_range=1, improved=args.improved_flag, num_seeds=1, exp_name=args.exp)
            output_acc_list += np.array(acc_list)
            output_acc5_list += np.array(acc5_list)
            iden = iden + 50
            aver_acc += acc
            aver_acc5 += acc_5

            aver_var += acc_var 
            aver_var5 += acc_var5
            
    print('top1[\''+args.exp+'\'] = ',np.round(output_acc_list/6,4).tolist())
    print('top5[\''+args.exp+'\'] = ',np.round(output_acc5_list/6,4).tolist())
    print('Acc : ', aver_acc/6, 'ACC5 : ', aver_acc5/6, 'ACC_Var: ', aver_var/6, 'acc5_var:', aver_var5/6 )



    