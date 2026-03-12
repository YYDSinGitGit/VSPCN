# from __future__ import print_function
import argparse
import os
import random
import json
import shutil
import time
import warnings
import h5py
import numpy as np
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from dataloading import *

import models
from utils import *

from models.agpl import ZSL
from glo import *
from models.evaluation import *
from models.visualization import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

best_prec1 = 0
best_H = 0


def main():
    print(args.loss)
    global best_prec1, best_H


    ''' save path '''
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    ''' random seed '''
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
    else:
        args.seed = random.randint(1, 10000)

    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    print('==> random seed:', args.seed)

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    ''' model building '''
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        best_prec1 = 0
        H = 0
        model = models.__dict__[args.arch](args=args)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](args=args)
    print("=> is the backbone fixed: '{}'".format(args.is_fix))

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    cls_loss = nn.CrossEntropyLoss()
    print("hh")
    print(model.named_parameters())
    for n, p in model.named_parameters():
        if ('prompt' in n or 'head' in n or 'project' in n or 'channel' in n or 'cls_token' in n \
                or 'W' in n or 'V1' in n or 'V2' in n or 'w2v_p' in n or 'p_w2v' in n or 'V' in n\
                or 'prompt_w2v' in n or 'Vprompt_token' in n \
                or 'q_w0' in n or 'k_w0' in n or 'v_w0' in n or 'pos_embed' in n \
                or 'q_v0' in n or 'k_v0' in n or 'v_v0' in n or 'ca' in n or 'V1' in n \
                or 'V2' in n or 'conv4' in n or 'conv5' in n or 'R1' in n or 'R2' in n or 'R3' in n \
                or 'q_r0' in n or 'k_r0' in n or 'v_r0' in n \
                or 'reason' in n or 'ffn1' in n or 'class_' in n or 'scores' in n):
            p.requires_grad = True
        else:
            p.requires_grad = False

    params_list = [{'params': model.parameters()}]
    all_optimizer = torch.optim.Adam(params_list, lr=args.lr, weight_decay=args.weight_decay)

    ''' optionally resume from a checkpoint'''
    if args.resume:  # false
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cuda:{}'.format(args.gpu)))
            if (best_H == 0):
                best_H = checkpoint['best_H']
            print('=> pretrained best_H {:.4F}'.format(best_H))
            model.load_state_dict(checkpoint['state_dict'])
            if args.ftp == 'True':
                for p in model.parameters():
                    p.requires_grad = False
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(all_optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, all_optimizer, epoch, name_sematic)
        H, S, U, = test(val_loader1, val_loader2, model, args)

        is_best = H > best_H
        best_H = max(H, best_H)
        print('H:{}   S:{}  U:{}'.format(H, S, U))
        # # save model
        try:
            pre_path = save_path
        except:
            pre_path = ''
        if is_best:
            if (args.resume == ''):
                save_path = os.path.join(args.save_path,
                                         'H:{}_U:{}_S:{}_epoch:{}_lr:{}_att:{}_gap:{}_{}:{}:{}'.format(round(H, 3),
                                                                            round(U, 3),
                                                                            round(S, 3),
                                                                            epoch, args.lr,
                                                                            args.att_name,
                                                                            args.gap,
                                                                            'channel' if args.channel else '',
                                                                            args.ref,
                                                                            'attention' if args.attention else ''))
            try:
                os.remove(pre_path)
            except:
                pass
            stat = {
                'epoch': epoch + 1,
                'unseen': U,
                'seen': S,
                'best_H': best_H,
            }
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_H': best_H,
            }, filename=save_path)
            print('saving!!!!')
    print(
        f"Best ending,epoch={stat['epoch']:.3f},unseen={stat['unseen']:.3f},seen={stat['seen']:.3f},H={stat['best_H']:.3f}")


def map_label(label, classes):
    classes = torch.tensor(classes)
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label

def cosine_simmilar(att_attain,sematic_protype):
    similarity = F.cosine_similarity(att_attain.unsqueeze(1), sematic_protype.unsqueeze(0),dim=2)
    return similarity


def train(train_loader, model, optimizer, epoch, name_sematics):
    alpha = json.loads(args.alpha)
    alpha0 = json.loads(args.alpha0)
    alpha1 = json.loads(args.alpha1)
    alpha2 = json.loads(args.alpha2)
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(args.gpu, non_blocking=args.pin_memory)
        target = target.cuda(args.gpu, non_blocking=args.pin_memory)
        map_target = map_label(target, args.seen_c)
        map_target = map_target.to(torch.device(args.gpu))
        model = model.to(device=torch.device(args.gpu))
        cls, v_prompt, l_prompt, att, features, f_avg, l_class, v_class = model(input)  # [5,768],[200,768],[5,197,768]

        sim = torch.matmul(cls, att.transpose(0, 1))
        sim_v = torch.matmul(v_prompt, att.transpose(0, 1))
        sim_l = torch.matmul(l_prompt, att.transpose(0, 1))
    
        seen_c = torch.from_numpy(args.seen_c).to(torch.device(args.gpu))
        seen_c = seen_c.long()
        seen_c = seen_c.unsqueeze(0).expand(sim.size(0), -1)

        sim_c = torch.gather(sim, 1, seen_c)
        sim_vc = torch.gather(sim_v, 1, seen_c)
        sim_lc = torch.gather(sim_l, 1, seen_c)
        sim_lclass = torch.gather(l_class, 1, seen_c)
        sim_vclass = torch.gather(v_class, 1, seen_c)
        sim_att = att[seen_c[0]]
        vis_sematic = sim_att[map_target]

        loss = F.cross_entropy(sim_c, map_target)
        sim_c.softmax(dim=-1)
        loss_vclass = F.cross_entropy(sim_vclass, map_target)
        loss_ED = ((loss_vclass + loss)/F.kl_div(sim_vclass.softmax(dim=-1).log(), sim_c.softmax(dim=-1), reduction='batchmean')+1).log()
        loss_KD = F.kl_div(vis_sematic.softmax(dim=-1).log(), l_prompt.softmax(dim=-1), reduction='batchmean') / 2 + \
                  F.kl_div(l_prompt.softmax(dim=-1).log(), vis_sematic.softmax(dim=-1), reduction='batchmean') / 2 + \
                  (vis_sematic - l_prompt).norm(p=2, dim=-1).mean()
        loss_SKD = alpha2[0] * loss_KD
        loss_CED = alpha1[0] * loss_ED + alpha1[1] * loss_vclass
        
        loss_KD0 = torch.norm(vis_sematic - cls, dim=-1).sum() / cls.shape[0]
        loss_AR = alpha0[0] * loss_KD0

        loss = loss + alpha[0] * loss_AR  + alpha[1] * loss_CED + alpha[2] * loss_SKD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        args.print_freq = 50
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}] loss:'.format
                  (epoch, i, len(train_loader)), end='')
            print('loss {:.4f}'.format(loss))

def test(val_loader1, val_loader2, model, args):
    model.eval()
    s = test_seen(val_loader1, model, args.seen_c, args.num_seen)
    u = test_seen(val_loader2, model, args.unseen_c, args.num_unseen)
    h = (2 * s * u / (s + u))
    return h, s, u

def test_seen(val_loader, model, classes, nun_c):
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(args.gpu, non_blocking=args.pin_memory)
            target = target.cpu().numpy()
            cls, v_prompt, l_prompt, att, features, f_avg, l, v = model(input)
            logits = torch.matmul(cls, att.transpose(0, 1))  # [5,200]
            logits = logits.cpu().numpy()

            logits = np.exp(logits - np.max(logits, axis=1, keepdims=True)) / np.sum(
                np.exp(logits - np.max(logits, axis=1, keepdims=True)), axis=1, keepdims=True)
            logits[:, args.unseen_c] += args.gap

            if (i == 0):
                gt_s = target
                pre = np.argmax(logits, axis=1)

            else:
                gt_s = np.hstack([gt_s, target])
                pre = np.hstack([pre, np.argmax(logits, axis=1)])

        k = 0
        acc_per_class = [0.0] * nun_c
        for i in classes:
            idx = np.where(gt_s == i)[0]
            acc_per_class[k] = (sum(gt_s[idx] == pre[idx]) * 1.0 / len(idx))
            k += 1
    return np.mean(acc_per_class)


def test_fix_prompt(val_loader, classes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clipmodel, _ = clip.load('ViT-B/32', args.gpu)
    text_inputs = torch.cat([clip.tokenize(f"a phtoto of {c}") for c in args.classname]).to(args.gpu)
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(args.gpu, non_blocking=args.pin_memory)
            target = target.cpu().numpy()
            image_features = clipmodel.encode_image(input)
            text_features = clipmodel.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            similarity = similarity.cpu().numpy()
            if (i == 0):
                gt_s = target
                pre = np.argmax(similarity, axis=1)

            else:
                gt_s = np.hstack([gt_s, target])
                pre = np.hstack([pre, np.argmax(similarity, axis=1)])
    k = 0
    acc_per_class = [0.0] * classes.__len__()
    for i in classes:
        idx = np.where(gt_s == i)[0]
        acc_per_class[k] = (sum(gt_s[idx] == pre[idx]) * 1.0 / len(idx))
        k += 1

    return np.mean(acc_per_class)


if __name__ == '__main__':
    main()
