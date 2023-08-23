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


def inversion(args, G, D, T, E, iden, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500,
              clip_range=1, num_seeds=5, verbose=False):
    iden = iden.view(-1).long().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    bs = iden.shape[0]

    G.eval()
    D.eval()
    T.eval()
    E.eval()

    flag = torch.zeros(bs)

    res = []
    res5 = []
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

                    if args.dataset == 'celeba':
                        eval_prob = E(utils.low2high(fake_img))[-1]
                    else:
                        eval_prob = E(fake_img)[-1]

                    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                    acc = iden.eq(eval_iden.long()).sum().item() * 100.0 / bs
                    print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i + 1,
                                                                                                        Prior_Loss_val,
                                                                                                        Iden_Loss_val,
                                                                                                        acc))

        fake = G(z)
        if args.dataset == 'celeba':
            eval_prob = E(utils.low2high(fake))[-1]
        else:
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

        res.append(cnt * 100.0 / bs)
        res5.append(cnt5 * 100.0 / bs)
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
    parser.add_argument('--target', default='VGG16')

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

        if args.target == 'VGG16':
            E = model.VGG16_V(num_classes)
        if args.target == 'VIB':
            E = model.VGG16_vib(num_classes)
        if args.target =='HSIC':
            E = model.VGG16(num_classes,hsic_training=True)
        e_path = args.target +'_eval.tar'
        E = nn.DataParallel(E).cuda()
        ckp_E = torch.load(e_path)
        E.load_state_dict(ckp_E['state_dict'], strict=False)

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

        if args.defense == 'HSIC' or args.defense == 'COCO':
            
            hp_ac_list = [

                (0.05, 0.5, 80.35),

            ]
            for (a1, a2, ac) in hp_ac_list:
                print("a1:", a1, "a2:", a2, "test_acc:", ac)

                T = model.VGG16(num_classes, True)
                T = nn.DataParallel(T).cuda()

                path_T = 'HSIC.tar'

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
            if args.defense == "vib":
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


            elif args.defense == 'reg':
                path_T = os.path.join(args.model_path, args.dataset, args.defense, "VGG16_reg_85.87.tar")
                # path_T = os.path.join(args.model_path, args.dataset, args.defense, "VGG16_reg_87.27.tar")
                T = model.VGG16(num_classes)

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
                    res = inversion(args, G, D, T, E, iden, lr=2e-2, iter_times=2000, verbose=True)
                    res_all.append(res)
                    iden = iden + ids_per_time

                res = np.array(res_all).mean(0)

                print(f"Acc:{res[0]:.4f} (+/- {res[2]:.4f}), Acc5:{res[1]:.4f} (+/- {res[3]:.4f})")

