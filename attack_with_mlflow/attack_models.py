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
import mlflow.pytorch
from mlflow.models import infer_signature
import mlflow

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
    parser.add_argument('--model', default='linear', help='VGG16 | IR152 | FaceNet64')
    parser.add_argument('--device', type=str, default='4,5,6,7', help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--attack_type', type=str, default='gmi')
    parser.add_argument('--path',default='./model.pth')
    parser.add_argument('--exp',default='testing')
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
    if args.attack_type=='kedmi':
        D = MinibatchDiscriminator()
        path_G = './checkpoint/improved_celeba_G.tar'
        path_D = './checkpoint/improved_celeba_D.tar'
    else:
        D = DGWGAN(3)
        path_G = '/workspace/data/data/celeba_G.tar'
        path_D = '/workspace/data/data/celeba_D.tar'
    
    D = torch.nn.DataParallel(D).cuda()
    ckp_G = torch.load(path_G)
    G.load_state_dict(ckp_G['state_dict'], strict=False)
    ckp_D = torch.load(path_D)
    D.load_state_dict(ckp_D['state_dict'], strict=False)

    T = get_model(args.model,args.num_class)
    path_T = args.path

    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T, strict=False)
    T=T.cuda()

    E = T

    ############         attack     ###########

    aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
    iter_times = 300
    output_acc_list = np.zeros((iter_times))
    output_acc5_list = np.zeros((iter_times))
    for i in range(1):
        iden = torch.from_numpy(np.arange(1000))

        # evaluate on the first 300 identities only
        for idx in range(1):
            #print("--------------------- Attack batch [%s]------------------------------" % idx)
            if args.attack_type == 'kedmi':
                acc, acc_5, acc_var, acc_var5, acc_list, acc5_list = dist_inversion(G, D, T, E, iden, itr=i, lr=2e-2, momentum=0.9, lamda=100, iter_times=iter_times, clip_range=1, improved=args.improved_flag, num_seeds=1, exp_name=args.exp)
            else:
                acc, acc_5, acc_var, acc_var5, acc_list, acc5_list = inversion(G, D, T, E, iden, itr=i, lr=2e-2, momentum=0.9, lamda=100, iter_times=iter_times, clip_range=1, improved=False, num_seeds=1, exp_name=args.exp)
            output_acc_list += np.array(acc_list)
            output_acc5_list += np.array(acc5_list)
            iden = iden + 50
            aver_acc += acc
            aver_acc5 += acc_5

            aver_var += acc_var 
            aver_var5 += acc_var5
            
    print('top1[\''+args.exp+'\'] = ',np.round(output_acc_list,4).tolist())
    print('top5[\''+args.exp+'\'] = ',np.round(output_acc5_list,4).tolist())
    print('Acc : ', aver_acc, 'ACC5 : ', aver_acc5, 'ACC_Var: ', aver_var, 'acc5_var:', aver_var5 )
    mlflow.log_metric("Top1", aver_acc)
    mlflow.log_metric("Top5", aver_acc5)



    
