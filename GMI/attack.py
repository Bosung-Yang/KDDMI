import torch, os, time, random, generator, discri
import numpy as np
import torch.nn as nn
import statistics
from argparse import ArgumentParser
from fid_score import calculate_fid_given_paths
from fid_score_raw import calculate_fid_given_paths0

device = "cuda"

import sys

sys.path.append('../BiDO')
import model, utils
from utils import save_tensor_images


def inversion(args, G, D, T, E_list, iden, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500,
              clip_range=1, num_seeds=2, verbose=False):
    iden = iden.view(-1).long().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    bs = iden.shape[0]

    G.eval()
    D.eval()
    T.eval()

    flag = torch.zeros(bs)

    for random_seed in range(num_seeds):
        tf = time.time()

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        z = torch.randn(bs, 100).cuda().float()
        z.requires_grad = True
        v = torch.zeros(bs, 100).cuda().float()

        for i in range(iter_times):
            fake = G(z)
            label = D(fake)
            out = T(fake)[-1]

            if z.grad is not None:
                z.grad.data.zero_()

            Prior_Loss = - label.mean()
            Iden_Loss = criterion(out, iden)
            Total_Loss = Prior_Loss + lamda * Iden_Loss

            Total_Loss.backward()

            v_prev = v.clone()
            gradient = z.grad.data
            v = momentum * v - lr * gradient
            z = z + (- momentum * v_prev + (1 + momentum) * v)
            z = torch.clamp(z.detach(), -clip_range, clip_range).float()
            z.requires_grad = True

            Prior_Loss_val = Prior_Loss.item()
            Iden_Loss_val = Iden_Loss.item()

            if verbose:
                if (i + 1) % 500 == 0:
                    fake_img = G(z.detach())

                    eval_prob = E(fake_img)[-1]

                    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                    acc = iden.eq(eval_iden.long()).sum().item() * 100.0 / bs
                    print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i + 1,
                                                                                                        Prior_Loss_val,
                                                                                                        Iden_Loss_val,
                                                                                                        acc))

##################### evaluation ############################
    fake = G(z)
    res = {'vgg':[], 'vib':[], 'hsic':[], 'kd':[], 'white':[]}
    res5 = {'vgg':[], 'vib':[], 'hsic':[], 'kd':[], 'white':[]}
    for E, model_name in E_list: 
        E.eval()
        eval_prob = E(fake)[-1]
        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
        cnt, cnt5 = 0, 0
        for i in range(bs):
            gt = iden[i].item()

            sample = fake[i]
            save_tensor_images(sample.detach(),
                            os.path.join(args.save_img_dir,
                                            "attack_iden_{:03d}|{}.png".format(gt + 1, random_seed + 1)))

            if eval_iden[i].item() == gt:
                cnt += 1
                flag[i] = 1
                best_img = G(z)[i]
                save_tensor_images(best_img.detach(),
                                os.path.join(args.success_dir,
                                                "attack_iden_{:03d}|{}.png".format(gt + 1, random_seed + 1)))

            _, top5_idx = torch.topk(eval_prob[i], 5)
            if gt in top5_idx:
                cnt5 += 1

        res[model_name].append(cnt * 100.0 / bs)
        res5[model_name].append(cnt5 * 100.0 / bs)
        torch.cuda.empty_cache()
        interval = time.time() - tf

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
                
    acc = statistics.mean(res['kd'])
    acc_5 = statistics.mean(res5['kd'])
    print()
    print("KD : Acc:{:.2f}\tAcc_5:{:.2f}".format(acc, acc_5))
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

        E_kd = model.VGG16_V(num_classes)
        path_E = '../final_tars/student-BiDO_73.28.tar'
        E_kd = nn.DataParallel(E_kd).cuda()
        checkpoint = torch.load(path_E)
        ckp_E = torch.load(path_E)
        E_kd.load_state_dict(ckp_E['state_dict'])

        E_list = [(E_hsic, 'hsic'), (E_vib,'vib'), (E_vgg, 'vgg'), (E_kd,'kd')]

        
        g_path = "./G.tar"
        G = generator.Generator()
        G = nn.DataParallel(G).cuda()
        ckp_G = torch.load(g_path)
        G.load_state_dict(ckp_G['state_dict'], strict=False)

        d_path = "./D.tar"
        D = discri.DGWGAN()
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
                res, res5 = inversion(args, G, D, T, E_list, iden, lr=2e-2, iter_times=2000, verbose=False)
                iden = iden + ids_per_time
                res_vgg.append(res['vgg'][0])
                res5_vgg.append(res5['vgg'][0])
                res_vib.append(res['vib'][0])
                res5_vib.append(res5['vib'][0])
                res_hsic.append(res['hsic'][0])
                res5_hsic.append(res5['hsic'][0])
                res_kd.append(res['kd'][0])
                res5_kd.append(res5['kd'][0])
                res_white.append(res['white'][0])
                res5_white.append(res5['white'][0])


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
                    res, res5 = inversion(args, G, D, T, E_list, iden, lr=2e-2, iter_times=2000, verbose=False)
                    iden = iden + ids_per_time
                    res_vgg.append(res['vgg'][0])
                    res5_vgg.append(res5['vgg'][0])
                    res_vib.append(res['vib'][0])
                    res5_vib.append(res5['vib'][0])
                    res_hsic.append(res['hsic'][0])
                    res5_hsic.append(res5['hsic'][0])
                    res_kd.append(res['kd'][0])
                    res5_kd.append(res5['kd'][0])
                    res_white.append(res['white'][0])
                    res5_white.append(res5['white'][0])

        if args.defense == 'VGG16':

          path_T = '../final_tars/eval/VGG16_80.16.tar'
          # path_T = os.path.join(args.model_path, args.dataset, args.defense, "VGG16_reg_87.27.tar")
          T = model.VGG16_V(num_classes)

          T = nn.DataParallel(T).cuda()
          checkpoint = torch.load(path_T)
          ckp_T = torch.load(path_T)
          T.load_state_dict(ckp_T['state_dict'])
          E_list.append((T,'white'))


          ids = 300
          times = 5
          ids_per_time = ids // times
          iden = torch.from_numpy(np.arange(ids_per_time))
          for idx in range(times):
              print("--------------------- Attack batch [%s]------------------------------" % idx)
              res, res5 = inversion(args, G, D, T, E_list, iden, lr=2e-2, iter_times=2000, verbose=False)
              iden = iden + ids_per_time
              res_vgg.append(res['vgg'][0])
              res5_vgg.append(res5['vgg'][0])
              res_vib.append(res['vib'][0])
              res5_vib.append(res5['vib'][0])
              res_hsic.append(res['hsic'][0])
              res5_hsic.append(res5['hsic'][0])
              res_kd.append(res['kd'][0])
              res5_kd.append(res5['kd'][0])
              res_white.append(res['white'][0])
              res5_white.append(res5['white'][0])

        if args.defense == 'kd':

          path_T = '../final_tars/student-BiDO_73.28.tar'
          # path_T = os.path.join(args.model_path, args.dataset, args.defense, "VGG16_reg_87.27.tar")
          T = model.VGG16_V(num_classes)

          T = nn.DataParallel(T).cuda()
          checkpoint = torch.load(path_T)
          ckp_T = torch.load(path_T)
          T.load_state_dict(ckp_T['state_dict'])
          E_list.append((T,'white'))


          ids = 300
          times = 5
          ids_per_time = ids // times
          iden = torch.from_numpy(np.arange(ids_per_time))
          for idx in range(times):
              print("--------------------- Attack batch [%s]------------------------------" % idx)
              res, res5 = inversion(args, G, D, T, E_list, iden, lr=2e-2, iter_times=2000, verbose=False)
              iden = iden + ids_per_time
              res_vgg.append(res['vgg'][0])
              res5_vgg.append(res5['vgg'][0])
              res_vib.append(res['vib'][0])
              res5_vib.append(res5['vib'][0])
              res_hsic.append(res['hsic'][0])
              res5_hsic.append(res5['hsic'][0])
              res_kd.append(res['kd'][0])
              res5_kd.append(res5['kd'][0])
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
      
        acc = statistics.mean(res_kd)
        acc_var = statistics.stdev(res_kd)
        acc_5 = statistics.mean(res5_kd)
        acc_var5 = statistics.stdev(res5_kd)                    
        print('-KD-')
        print("VGG : Acc:{:.4f} +/- {:.4f}\tAcc_5:{:.4f}+/- {:.4f}".format(acc,acc_var, acc_5,acc_var5))
        print()
      
        acc = statistics.mean(res_white)
        acc_var = statistics.stdev(res_white)
        acc_5 = statistics.mean(res5_white)
        acc_var5 = statistics.stdev(res5_white)                    
        print('-Fully-white box-')
        print("VGG : Acc:{:.4f} +/- {:.4f}\tAcc_5:{:.4f}+/- {:.4f}".format(acc,acc_var, acc_5,acc_var5))
        print()
