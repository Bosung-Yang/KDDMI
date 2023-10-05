import torch, os, time, random, generator, discri
import numpy as np
import torch.nn as nn
import statistics
from argparse import ArgumentParser
from fid_score import calculate_fid_given_paths
from fid_score_raw import calculate_fid_given_paths0
import mlflow.pytorch
from mlflow.models import infer_signature
import mlflow
from torch.autograd import Variable
import torch.optim as optim
from generator import *
from discri import *
from utils import *
device = "cuda"

import sys

sys.path.append('../BiDO')
import model, utils
from utils import save_tensor_images



def reparameterize(mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return eps * std + mu


def inversion(G, D, T, E_list, iden, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=True, num_seeds=5, exp_name=' '):
    
    device = "cuda"
    num_classes = 1000
    save_img_dir = exp_name # all attack imgs
    os.makedirs(save_img_dir, exist_ok=True)
    success_dir = './ked_res_success'
    os.makedirs(success_dir, exist_ok=True)

    acc_list = []
    acc5_list = []
    iden = iden.view(-1).long().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    bs = iden.shape[0]

    G.eval()
    D.eval()
    T.eval()


    no = torch.zeros(bs) # index for saving all success attack images

    tf = time.time()

    #NOTE
    mu = Variable(torch.zeros(bs, 100), requires_grad=True)
    log_var = Variable(torch.ones(bs, 100), requires_grad=True)

    params = [mu, log_var]
    solver = optim.Adam(params, lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(solver, 1800, gamma=0.1)
    res = []
    res5 = []     
    for i in range(iter_times):
        z = reparameterize(mu, log_var)
        fake = G(z)
        if improved == True:
            _, label =  D(fake)
        else:
            label = D(fake)
        
        out = T(fake)[-1]
        
        for p in params:
            if p.grad is not None:
                p.grad.data.zero_()

        if improved:
            Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
        else:
            Prior_Loss = - label.mean()
        Iden_Loss = criterion(out, iden)
        Total_Loss = Prior_Loss + lamda * Iden_Loss

        Total_Loss.backward()
        solver.step()
        
        z = torch.clamp(z.detach(), -clip_range, clip_range).float()

        Prior_Loss_val = Prior_Loss.item()
        Iden_Loss_val = Iden_Loss.item()

        if (i+1) % 1 == 0:
            cnt = 0
            cnt5 = 0
            z = reparameterize(mu, log_var)
            fake_img = G(z.detach())
            eval_prob = T(fake_img)[-1]
            eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
            acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
            for j in range(bs):
                gt = iden[j].item()
                sample = fake_img[j]
                if eval_iden[j].item() == gt:
                    cnt+=1
                _, top5_idx = torch.topk(eval_prob[j],5)
                if gt in top5_idx:
                    cnt5+=1
            acc_list.append(acc)
            acc5_list.append(cnt5/bs)

    interval = time.time() - tf
    

    res = {'vgg':[], 'vib':[], 'hsic':[], 'kd':[], 'white':[]}
    res5 = {'vgg':[], 'vib':[], 'hsic':[], 'kd':[], 'white':[]}
    seed_acc = torch.zeros((bs, 5))
    for E, model_name in E_list: 
        E.eval()
        tf = time.time()
        z = reparameterize(mu, log_var)
        fake = G(z)
        score = T(fake)[-1]
        eval_prob = E(fake)[-1]
        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
        
        cnt, cnt5 = 0, 0
        for i in range(bs):
            gt = iden[i].item()
            sample = fake[i]
            save_tensor_images(sample.detach(), os.path.join(save_img_dir, "attack_iden_{}.png".format(gt)))

            if eval_iden[i].item() == gt:
                seed_acc[i, random_seed] = 1
                cnt += 1
                best_img = G(z)[i]
                no[i] += 1
            _, top5_idx = torch.topk(eval_prob[i], 5)
            if gt in top5_idx:
                cnt5 += 1
                
        interval = time.time() - tf
        res[model_name].append(cnt * 100.0 / bs)
        res5[model_name].append(cnt5 * 100.0 / bs)
        
        torch.cuda.empty_cache()
        interval = time.time() - tf
        print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 100.0 / bs))
        

    acc = statistics.mean(res['vgg'])
    acc_5 = statistics.mean(res5['vgg'])
    print()
    print("VGG : Acc:{:.2f}\tAcc_5:{:.2f}".format(acc, acc_5))
    print()

    acc = statistics.mean(res['vib'])
    acc_5 = statistics.mean(res5['vib'])
    print()
    print("vib : Acc:{:.2f}\tAcc_5:{:.2f}".format(acc, acc_5))
    print()

    acc = statistics.mean(res['hsic'])
    acc_5 = statistics.mean(res5['hsic'])
    print()
    print("hsic : Acc:{:.2f}\tAcc_5:{:.2f}".format(acc, acc_5))
    print()

    acc = statistics.mean(res['white'])
    acc_5 = statistics.mean(res5['white'])
    print()
    print("white : Acc:{:.2f}\tAcc_5:{:.2f}".format(acc, acc_5))
    print()
    return res, res5

if __name__ == '__main__':
    parser = ArgumentParser(description='Step2: targeted recovery')
    parser.add_argument('--dataset', default='celeba', help='celeba | cxr | mnist')
    parser.add_argument('--defense', default='reg', help='reg | vib | HSIC')
    parser.add_argument('--save_img_dir', default='./attack_res/')
    parser.add_argument('--success_dir', default='')
    parser.add_argument('--model_path', default='../BiDO/target_model')
    parser.add_argument('--verbose', action='store_true', help='')
    parser.add_argument('--iter', default=3000, type=int)
    parser.add_argument('--target')

    args = parser.parse_args()

    ############################# mkdirs ##############################
    args.save_img_dir = os.path.join(args.save_img_dir, args.dataset, args.defense)
    args.success_dir = args.save_img_dir + "/res_success"
    os.makedirs(args.success_dir, exist_ok=True)
    args.save_img_dir = os.path.join(args.save_img_dir, 'all')
    os.makedirs(args.save_img_dir, exist_ok=True)

    eval_path = "./eval_model"
    ############################# mkdirs ##############################

    if args.dataset == 'celeba':
        model_name = "VGG16"
        num_classes = 1000
        E_hsic = model.VGG16(num_classes,True)
        path_E = '../final_tars/BiDO_teacher_71.35_0.1_0.1.tar'
        E_hsic = nn.DataParallel(E_hsic).cuda()
        checkpoint = torch.load(path_E)
        ckp_E = torch.load(path_E)
        E_hsic.load_state_dict(ckp_E['state_dict'])

        E_vib = model.VGG16_vib(num_classes)
        path_E = '../final_tars/VIB_teacher_0.010_60.95.tar'
        E_vib = nn.DataParallel(E_vib).cuda()
        checkpoint = torch.load(path_E)
        ckp_E = torch.load(path_E)
        E_vib.load_state_dict(ckp_E['state_dict'])


        E_vgg = model.VGG16_V(num_classes)
        path_E = '../final_tars/eval/VGG16_79.23.tar'
        E_vgg = nn.DataParallel(E_vgg).cuda()
        checkpoint = torch.load(path_E)
        ckp_E = torch.load(path_E)
        E_vgg.load_state_dict(ckp_E['state_dict'])

        E_list = [(E_hsic, 'hsic'), (E_vib,'vib'), (E_vgg, 'vgg')]
        
        g_path = "./KED_G.tar"
        G = generator.Generator()
        G = nn.DataParallel(G).cuda()
        ckp_G = torch.load(g_path)
        G.load_state_dict(ckp_G['state_dict'], strict=False)

        d_path = "./KED_D.tar"
        D = discri.MinibatchDiscriminator()
        D = nn.DataParallel(D).cuda()
        ckp_D = torch.load(d_path)
        D.load_state_dict(ckp_D['state_dict'], strict=False)

        res_vgg = []
        res5_vgg = []
        res_vib = []
        res5_vib = []
        res_hsic = []
        res5_hsic = []
        res_kd = []
        res5_kd = []
        res_white = []
        res5_white = []
        
        if args.defense == 'HSIC' or args.defense == 'COCO':

            T = model.VGG16(num_classes, True)
            T = nn.DataParallel(T).cuda()
            path_T = '../final_tars/BiDO_teacher_71.92_0.1_0.1.tar'

            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'], strict=False)
            E_list.append((T,'white'))
                    
            res_all = []
            ids = 300
            times = 5
            ids_per_time = ids // times
            iden = torch.from_numpy(np.arange(ids_per_time))
            for idx in range(times):
                print("--------------------- Attack batch [%s]------------------------------" % idx)
                res,res5 = inversion(G, D, T, E_list, iden, iter_times=2000)
                res_all.append(res)
                iden = iden + ids_per_time
                                
                res_vgg.append(res['vgg'][0])
                res5_vgg.append(res5['vgg'][0])
                res_vib.append(res['vib'][0])
                res5_vib.append(res5['vib'][0])
                res_hsic.append(res['hsic'][0])
                res5_hsic.append(res5['hsic'][0])
                #res_kd.append(res['kd'])
                #res5_kd.append(res5['kd'])
                res_white.append(res['white'][0])
                res5_white.append(res5['white'][0])
                
                

        else:
            if args.defense == "VIB":
                path_T_list = [
                    '../final_tars/VIB_teacher_0.010_62.18.tar'
                ]
                for path_T in path_T_list:
                    T = model.VGG16_vib(num_classes)
                    T = nn.DataParallel(T).cuda()

                    checkpoint = torch.load(path_T)
                    ckp_T = torch.load(path_T)
                    T.load_state_dict(ckp_T['state_dict'])
                    E_list.append((T,'white'))

                    res_all = []
                    ids = 300
                    times = 5
                    ids_per_time = ids // times
                    iden = torch.from_numpy(np.arange(ids_per_time))
                    
                    for idx in range(times):
                        print("--------------------- Attack batch [%s]------------------------------" % idx)
                        res,res5 = inversion( G, D, T, E_list, iden, iter_times=2000)
                        iden = iden + ids_per_time
                        res_vgg.append(res['vgg'][0])
                        res5_vgg.append(res5['vgg'][0])
                        res_vib.append(res['vib'][0])
                        res5_vib.append(res5['vib'][0])
                        res_hsic.append(res['hsic'][0])
                        res5_hsic.append(res5['hsic'][0])
                        #res_kd.append(res['kd'])
                        #res5_kd.append(res5['kd'])
                        res_white.append(res['white'][0])
                        res5_white.append(res5['white'][0])


            elif args.defense == 'VGG16':

                path_T = '../final_tars/eval/VGG16_80.16.tar'
                # path_T = os.path.join(args.model_path, args.dataset, args.defense, "VGG16_reg_87.27.tar")
                T = model.VGG16_V(num_classes)

                T = nn.DataParallel(T).cuda()
                checkpoint = torch.load(path_T)
                ckp_T = torch.load(path_T)
                T.load_state_dict(ckp_T['state_dict'])
                E_list.append((T,'white'))

                res_all = []
                ids = 300
                times = 5
                ids_per_time = ids // times
                iden = torch.from_numpy(np.arange(ids_per_time))
                for idx in range(times):
                    print("--------------------- Attack batch [%s]------------------------------" % idx)
                    res, res5 = inversion( G, D, T, E_list, iden, lr=2e-2, iter_times=2000)
                    iden = iden + ids_per_time
                    res_vgg.append(res['vgg'][0])
                    res5_vgg.append(res5['vgg'][0])
                    res_vib.append(res['vib'][0])
                    res5_vib.append(res5['vib'][0])
                    res_hsic.append(res['hsic'][0])
                    res5_hsic.append(res5['hsic'][0])
                    #res_kd.append(res['kd'])
                    #res5_kd.append(res5['kd'])
                    res_white.append(res['white'][0])
                    res5_white.append(res5['white'][0])

                
        print(res_vgg)
        acc = statistics.mean(res_vgg)
        acc_var = statistics.stdev(res_vgg)
        acc_5 = statistics.mean(res5_vgg)
        acc_var5 = statistics.stdev(res5_vgg)                    
        print('-VGG16-')
        print("VGG : Acc:{:.4f} +/- {:.4f}\tAcc_5:{:.4f}+/- {:.4f}".format(acc,acc_var, acc_5,acc_var5))
        print()

        acc = statistics.mean(res_vib)
        acc_var = statistics.stdev(res_vib)
        acc_5 = statistics.mean(res5_vib)
        acc_var5 = statistics.stdev(res5_vib)                    
        print('-MID-')
        print("VGG : Acc:{:.4f} +/- {:.4f}\tAcc_5:{:.4f}+/- {:.4f}".format(acc,acc_var, acc_5,acc_var5))
        print()

        acc = statistics.mean(res_hsic)
        acc_var = statistics.stdev(res_hsic)
        acc_5 = statistics.mean(res5_hsic)
        acc_var5 = statistics.stdev(res5_hsic)                    
        print('-BiDO-')
        print("VGG : Acc:{:.4f} +/- {:.4f}\tAcc_5:{:.4f}+/- {:.4f}".format(acc,acc_var, acc_5,acc_var5))
        print()

        acc = statistics.mean(res_white)
        acc_var = statistics.stdev(res_white)
        acc_5 = statistics.mean(res5_white)
        acc_var5 = statistics.stdev(res5_white)                    
        print('-Fully-white box-')
        print("VGG : Acc:{:.4f} +/- {:.4f}\tAcc_5:{:.4f}+/- {:.4f}".format(acc,acc_var, acc_5,acc_var5))
        print()
