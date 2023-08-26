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


def inversion(G, D, T, E, iden, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=True, num_seeds=5, exp_name=' '):
    
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
    E.eval()

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
        
        out = T(fake)[1]
        
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
            eval_prob = T(fake_img)[1]
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
    

    res = []
    res5 = []
    seed_acc = torch.zeros((bs, 5))
    for random_seed in range(num_seeds):
        tf = time.time()
        z = reparameterize(mu, log_var)
        fake = G(z)
        score = T(fake)[1]
        eval_prob = E(fake)[1]
        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
        
        cnt, cnt5 = 0, 0
        for i in range(bs):
            gt = iden[i].item()
            sample = fake[i]
            save_tensor_images(sample.detach(), os.path.join(save_img_dir, "attack_iden_{}_{}.png".format(gt, random_seed)))

            if eval_iden[i].item() == gt:
                seed_acc[i, random_seed] = 1
                cnt += 1
                best_img = G(z)[i]
                save_tensor_images(best_img.detach(), os.path.join(success_dir, "{}_attack_iden_{}_{}.png".format(itr, gt+1, int(no[i]))))
                no[i] += 1
            _, top5_idx = torch.topk(eval_prob[i], 5)
            if gt in top5_idx:
                cnt5 += 1
                
        interval = time.time() - tf
        res.append(cnt * 1.0 / bs)
        res5.append(cnt5 * 1.0 / bs)

        torch.cuda.empty_cache()
        interval = time.time() - tf
        print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 100.0 / bs))
        

    acc = statistics.mean(res)
    acc_5 = statistics.mean(res5)
    acc_var = statistics.stdev(res)
    acc_var5 = statistics.stdev(res5)
    print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}\tacc_var5{:.4f}".format(acc, acc_5, acc_var, acc_var5))

    return acc, acc_5, acc_var, acc_var5

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
        if args.target=='HSIC':
            E = model.VGG16(num_classes,True)
            path_E = 'VGG16_0.050_0.200_68.20.tar'
        elif args.target == 'VIB':
            E = model.VGG16_vib(num_classes)
            path_E = './VIB_eval.tar'
        elif args.target =='VGG16':
            E = model.VGG16_V(num_classes)
            path_E = './VGG16_eval.tar'
        elif args.target =='KD':
            E = model.VGG16_V(num_classes)
            path_E = './KD.tar'
        E = nn.DataParallel(E).cuda()

        checkpoint = torch.load(path_E)
        ckp_E = torch.load(path_E)
        E.load_state_dict(ckp_E['state_dict'])
        
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

        if args.defense == 'HSIC' or args.defense == 'COCO':
            hp_ac_list = [
                # HSIC
                # 1
                (0.05, 0.5, 80.35),
                # (0.05, 1.0, 70.08),
                # (0.05, 2.5, 56.18),
                # 2
                # (0.05, 0.5, 78.89),
                # (0.05, 1.0, 69.68),
                # (0.05, 2.5, 56.62),
            ]
            for (a1, a2, ac) in hp_ac_list:
                print("a1:", a1, "a2:", a2, "test_acc:", ac)

                T = model.VGG16(num_classes, True)
                T = nn.DataParallel(T).cuda()

                model_tar = f"{model_name}_{a1:.3f}&{a2:.3f}_{ac:.2f}.tar"

                path_T = 'VGG16_0.050_0.200_68.20.tar'

                ckp_T = torch.load(path_T)
                T.load_state_dict(ckp_T['state_dict'], strict=False)
                        
                res_all = []
                ids = 300
                times = 5
                ids_per_time = ids // times
                iden = torch.from_numpy(np.arange(ids_per_time))
                for idx in range(times):
                    print("--------------------- Attack batch [%s]------------------------------" % idx)
                    res = inversion(args, G, D, T, E, iden, iter_times=2000, verbose=True)
                    res_all.append(res)
                    iden = iden + ids_per_time

                res = np.array(res_all).mean(0)
                
                print(f"Acc:{res[0]:.4f} (+/- {res[2]:.4f}), Acc5:{res[1]:.4f} (+/- {res[3]:.4f})")
                

        else:
            if args.defense == "VIB":
                path_T_list = [
                    'VIB.tar'
                ]
                for path_T in path_T_list:
                    T = model.VGG16_vib(num_classes)
                    T = nn.DataParallel(T).cuda()

                    checkpoint = torch.load(path_T)
                    ckp_T = torch.load(path_T)
                    T.load_state_dict(ckp_T['state_dict'])

                    res_all = []
                    ids = 300
                    times = 5
                    ids_per_time = ids // times
                    iden = torch.from_numpy(np.arange(ids_per_time))
                    for idx in range(times):
                        print("--------------------- Attack batch [%s]------------------------------" % idx)
                        res = inversion(args, G, D, T, E, iden, iter_times=2000, verbose=True)
                        res_all.append(res)
                        iden = iden + ids_per_time

                    res = np.array(res_all).mean(0)
                    print(f"Acc:{res[0]:.4f} (+/- {res[2]:.4f}), Acc5:{res[1]:.4f} (+/- {res[3]:.4f})")


            elif args.defense == 'VGG16':
                path_T = 'VGG16.tar'
                # path_T = os.path.join(args.model_path, args.dataset, args.defense, "VGG16_reg_87.27.tar")
                T = model.VGG16_V(num_classes)

                T = nn.DataParallel(T).cuda()

                checkpoint = torch.load(path_T)
                ckp_T = torch.load(path_T)
                T.load_state_dict(ckp_T['state_dict'])

                res_all = []
                ids = 300
                times = 5
                ids_per_time = ids // times
                iden = torch.from_numpy(np.arange(ids_per_time))
                for idx in range(times):
                    print("--------------------- Attack batch [%s]------------------------------" % idx)
                    res = inversion( G, D, T, E, iden, lr=2e-2, iter_times=2000)
                    res_all.append(res)
                    iden = iden + ids_per_time

                res = np.array(res_all).mean(0)
                print(f"Acc:{res[0]:.4f} (+/- {res[2]:.4f}), Acc5:{res[1]:.4f} (+/- {res[3]:.4f})")
        mlflow.log_metric("Top1", res[0])
        mlflow.log_metric('top1-std', res[2])
        mlflow.log_metric("Top5", res[1])
        mlflow.log_metric('top5_std', res[3])
