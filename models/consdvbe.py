import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from models.MPNCOV import MPNCOV
import torch.nn.functional as F
import models.resnet
import models.densenet
import models.senet
from models.operations import *
from glo import *
from einops import rearrange, repeat
from models.mytransformer import *
from models.attention import  *

import argparse
import json
with open('models/config.json', 'r') as f:
    config=json.load(f)
config = argparse.Namespace(**config)


import re
from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['dvbe']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Model(nn.Module):
    def __init__(self, args=None):
        super(Model, self).__init__()  # 为什么要到这里才开始继承呢
        self.inplanes = 64
        is_fix = args.is_fix
        sf_size = args.sf_size
        self.arch = args.backbone
        self.adj = args.adj
        self.sf = torch.nn.Parameter(torch.from_numpy(args.sf))#用全部的属性来训练
        self.sf.requires_grad = False
        self.seen_sf = torch.nn.Parameter(torch.from_numpy(args.seen_sf))
        self.seen_sf.requires_grad = False
        self.unseen_sf = torch.nn.Parameter(torch.from_numpy(args.unseen_sf))
        self.unseen_sf.requires_grad = False
        # super(Model, self).__init__()#为什么要到这里才开始继承呢
        ''' backbone net'''
        block = Bottleneck
        layers = [3, 4, 23, 3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        if(args.resume==''):
            for p in self.parameters():
                p.requires_grad = False

        if 'densenet' in self.arch:
            feat_dim = 1920
        else:
            feat_dim = 2048

        '''TransformerEncoder'''
        # self.embed_cv = nn.Sequential(nn.Linear(2048, 300))
        # encoder_layer = nn.TransformerEncoderLayer(d_model=300, nhead=args.h,batch_first=True)  #d_model输入Transformer编码层的特征维度，在编码层会将特征从300--2048--300进行一个前向传播
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.l)  # 默认用6层
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 300))
        # self.cls_token2 = nn.Parameter(torch.randn(1, 1, 300))
        # self.lin=nn.Linear(args.tf_dim, args.feat_dim)
        # self.embed_attr = nn.Sequential(nn.Linear(85, 300))#将属性从85维映射到300维
        # self.lin = nn.Sequential(nn.Linear(300, args.feat_dim))
        # self.pdist = nn.PairwiseDistance(p=2)

        "odr cross attention"
        self.embed_cv = nn.Sequential(nn.Linear(2048, 300))
        self.odr_token1 = nn.Parameter(torch.randn(1, 1, 300))
        self.odr_token2 = nn.Parameter(torch.randn(1, 1, 196))
        encoder_layer = nn.TransformerEncoderLayer(d_model=300, nhead=args.h, batch_first=True)
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=196, nhead=1,
                                                   batch_first=True)  # d_model输入Transformer编码层的特征维度，在编码层会将特征从300--2048--300进行一个前向传播
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.l)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer2, num_layers=args.l)
        self.lin2 = nn.Sequential(nn.Linear(496, args.feat_dim))



        '''crossattention'''
        if args.att:
            self.att_embed = nn.Linear(1, 300)
            self.cls = nn.Linear(300, 50)
            self.visual_attention = BertCrossattLayer(config)
            # decoder_layer = TransformerDecoderLayer(d_model=300,  # 解码层直接用了nn自带的解码层
            #                                     nhead=1,
            #                                     dim_feedforward=300,
            #                                     dropout=0.5,
            #                                     SAtt=True)
            decoder_layer = nn.TransformerDecoderLayer(d_model=300,  # 解码层直接用了nn自带的解码层
                                                       nhead=1,
                                                       dim_feedforward=300)
            self.transformer_decoder = nn.TransformerDecoder(  # 直接调用Transformer的解码层
                decoder_layer, num_layers=3)
        else:
            ''' Zero-Shot Recognition Module '''
            self.zsr_proj = nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            # self.zsr_sem = nn.Sequential(
            #     nn.Linear(sf_size, 1024),
            #     nn.LeakyReLU(),
            #     # GraphConv(sf_size,1024,self.adj),#构建了一个nc*nc的邻接矩阵
            #     # GraphConv(1024,feat_dim,self.adj),
            #     nn.Linear(1024, feat_dim),  # 将属性映射到2048维度
            #     nn.LeakyReLU(),
            # )
            #dvbe是将属性映射到2048，对比嵌入将语义信息映射到1024
            self.zsr_sem = nn.Sequential(
                nn.Linear(sf_size, 1024),
                nn.LeakyReLU(),
                # GraphConv(sf_size,1024,self.adj),#构建了一个nc*nc的邻接矩阵
                # GraphConv(1024,feat_dim,self.adj),
                # nn.Linear(1024, feat_dim),  # 将属性映射到2048维度
                # nn.LeakyReLU(),
            )
            self.zsr_aux = nn.Linear(feat_dim, args.num_seen)  # 这里做seen个类别就好了，只是用于辅助训练，测试的时候用不到

            ''' params ini '''
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        ''' Open-Domain Recognition Module '''
        # self.odr_proj1 = nn.Sequential(
        #     nn.Conv2d(feat_dim, 256, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )
        # self.odr_proj2 = nn.Sequential(
        #     nn.Conv2d(feat_dim, 256, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )
        # self.odr_spatial = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.Sigmoid(),
        # )
        # self.odr_channel = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(256, int(256 / 16), kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(int(256 / 16), 256, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.Sigmoid(),
        # )
        # self.odr_linear=nn.Sequential(
        #     nn.Linear(int(256 * (256 + 1) / 2), args.feat_dim),
        # )
        # if args.loss=='':
        #     self.odr_linear = nn.Sequential(
        #         nn.Linear(int(256 * (256 + 1) / 2), args.num_seen),
        #     )


    def cross_att(self, lang_input, visn_input):
        # Cross Attention
        # lang_att_output = self.visual_attention(lang_input, visn_input)#lang_input做q，visn_input做kv，结果是融合了视觉的语义信息
        visn_att_output = self.visual_attention(lang_input, visn_input)#visn_input做q，lang_input做kv，结果为融合了语义信息的视觉信息
        # return lang_att_output, visn_att_output
        return visn_att_output

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        last_conv = x




        '''Transformer'''
        # cls_tokens1 = repeat(self.cls_token, '1 n d -> b n d', b=x.shape[0])  # 4*100*300  #4*101*300
        # x=x.view(x.shape[0],x.shape[1],-1)#14*14*2048-->196*2048
        # x = self.embed_cv(x.permute(0, 2, 1))#196*2048-->196*300(这部分的参数是两个网络共享的)
        # xt = torch.cat([x, cls_tokens1], dim=1)
        # encod_feat=self.transformer_encoder(xt)#196*300-->196*300
        # # encod_feat = self.cross_att(xt,xt)  # 196*300-->196*300
        # odr_x=encod_feat[:,0,:]
        # # odr_x=encod_feat.mean(1)
        # odr_x = self.lin(odr_x)#不做线性了吧

        '''crossattention Transformer'''
        odr_tokens1 = repeat(self.odr_token1, '1 n d -> b n d', b=x.shape[0])  # 4*100*300  #4*101*300
        odr_tokens2 = repeat(self.odr_token2, '1 n d -> b n d', b=x.shape[0])
        x=x.view(x.shape[0],x.shape[1],-1)#14*14*2048-->196*2048
        x1 = self.embed_cv(x.permute(0, 2, 1))#196*2048-->196*300(这部分的参数是两个网络共享的)
        x2 = x1.permute(0,2,1)
        x1 = torch.cat([x1, odr_tokens1], dim=1)
        x2 = torch.cat([x2, odr_tokens2], dim=1)
        encod_feat1=self.transformer_encoder(x1)#196*300-->196*300
        encod_feat2 = self.transformer_encoder2(x2)  # 196*300-->196*300
        # encod_feat = self.cross_att(xt,xt)  # 196*300-->196*300
        odr_x1=encod_feat1[:,0,:]
        odr_x2 = encod_feat2[:, 0, :]
        odr_x = torch.cat([odr_x1, odr_x2], dim=-1)
        odr_x = self.lin2(odr_x)#不做线性了吧


        ''' ODR Module '''
        # x1 = self.odr_proj1(last_conv)
        # x2 = x1  # self.odr_proj1(last_conv)
        # # att gen（产生空间注意力图和通道注意力图）
        # att1 = self.odr_spatial(x1)
        # att2 = self.odr_channel(x2)
        # # att1
        # x1 = att2 * x1 + x1
        # x1 = x1.view(x1.size(0), x1.size(1), -1)
        # # att2
        # x2 = att1 * x2 + x2
        # x2 = x2.view(x2.size(0), x2.size(1), -1)
        # # covariance pooling（方差池化？）
        # x1 = x1 - torch.mean(x1, dim=2, keepdim=True)
        # x2 = x2 - torch.mean(x2, dim=2, keepdim=True)
        # A = 1. / x1.size(2) * x1.bmm(x2.transpose(1, 2))  # 矩阵乘法，这里为什么要取逆呢    1./
        # # norm
        # x = MPNCOV.SqrtmLayer(A, 5)
        # x = MPNCOV.TriuvecLayer(x)
        # odr_x = x.view(x.size(0), -1)
        # #linear
        # odr_x=self.odr_linear(odr_x)

        '''第二阶段'''

        if args.att:
            # att_input = att_input.unsqueeze(-1)
            # arr_emb=self.att_embed(att_input)
            # cls_tokens2 = repeat(self.cls_token, '1 n d -> b n d', b=arr_emb.shape[0])
            # arr_emb = torch.cat([arr_emb, cls_tokens2], dim=1)
            # lang_att_output, visn_att_output = self.cross_att(arr_emb, xt)#x~196*430*300
            # x = x.permute(1, 0, 2)
            # out = self.transformer_decoder(h_attr_batch, x)#40*batch*300,可以理解成batch个融合了视觉特征的语义信息，融合视觉特征的语义特征应该尽量保持与原始语义特征一致
            # zsr_logit = (self.lin(out)).reshape(out.shape[1],out.shape[0])
            # zsr_logit=out.mean(dim=2).permute(1,0)
            cls_tokens2 = repeat(self.cls_token2, '1 n d -> b n d', b=x.shape[0])
            '''全部语义信息扔进去'''
            attr=self.embed_attr(self.sf)

            h_attr_batch = attr.unsqueeze(0).repeat(x.shape[0],1 , 1)#40*430*300


            #视觉信息做q(将语义信息聚合到视觉信息)
            if args.a2v:
                xt2 = torch.cat([x, cls_tokens2], dim=1)
                output = self.cross_att(xt2, h_attr_batch)

            #语义信息做q（将视觉信息聚合到语义信息）
            else:
                h_attr_batch = torch.cat([h_attr_batch, cls_tokens2], dim=1)
                output = self.cross_att(h_attr_batch, xt)
            zsr_logit=output[:,0,:]
            zsr_logit = self.cls(zsr_logit)

            return odr_x, zsr_logit
        else:
            zsr_x = self.zsr_proj(last_conv).view(last_conv.size(0), -1)  # 提取视觉特征
            if args.allsemantic:
                zsr_classifier = self.zsr_sem(self.sf)
            else:
                if self.training:
                    zsr_classifier = self.zsr_sem(self.seen_sf)
                else:
                    zsr_classifier=self.zsr_sem(self.unseen_sf)
            w_norm = F.normalize(zsr_classifier, p=2, dim=1)  # 语义特征归一化
            x_norm = F.normalize(zsr_x, p=2, dim=1)  # 视觉特征归一化
            # zsr_logit = x_norm.mm(w_norm.permute(1, 0))  # 视觉特征语义特征相乘
            zsr_logit_aux = self.zsr_aux(zsr_x)  # 用于无语义视觉特征损失的正则化约束
            return (odr_x,zsr_logit_aux,0),(zsr_x,zsr_classifier)




class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.cls_loss = nn.CrossEntropyLoss()  # reduce=False)
        self.sigma = args.sigma

    def forward(self, label, logits):
        odr_logit = logits[0]
        zsr_logit = logits[1]
        zsr_logit_aux = logits[2]

        ''' ODR Loss '''
        prob = F.softmax(odr_logit, dim=1).detach()  # 固定了odr框架，训练什么呢？训练mw？
        y = prob[torch.arange(prob.size(0)).long(), label]  # 取出gt对应的概率
        mw = torch.exp(-(y - 1.0) ** 2 / self.sigma)  # 这个就是lamda，是一个自适应的边界控制 参数，y现在是一个gt对应的概率，也就是一个变量，mw是y的函数
        one_hot = torch.zerbatchos_like(odr_logit)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        odr_logit = odr_logit * (1 - one_hot * mw.view(mw.size(0), 1))
        L_odr = self.cls_loss(odr_logit, label)  # 当L_odr反向传播的时候只更新mw而没有更新odr框架的参数

        ''' ZSL Loss '''
        idx = torch.arange(zsr_logit.size(0)).long()
        L_zsr = (1 - zsr_logit[
            idx, label]).mean()  # gt对应的输出改立应该越大越好，所以-zsr_logit应该越小越好（这里的损失直接去概率的相反数，而这个概率是通过视觉信息和语义信息相乘得到的）

        L_aux = self.cls_loss(zsr_logit_aux, label)  # 辅助损失使用交叉熵

        total_loss = L_odr + L_zsr + L_aux

        return total_loss, L_odr, L_zsr, L_aux


class Dist(nn.Module):  # to compute the distance, such as: d_e(x,y) or d_d(x,y)
    def __init__(self, num_classes=10, num_centers=1, feat_dim=2, init='random'):
        super(Dist, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_centers = num_centers

        if init == 'random':
            self.centers = nn.Parameter(0.1 * torch.randn(num_classes * num_centers, self.feat_dim))
        else:
            self.centers = nn.Parameter(torch.Tensor(num_classes * num_centers, self.feat_dim))
            self.centers.data.fill_(0)

    def forward(self, features, center=None, metric=args.metric):
        if metric == 'l2':  # (使用向量的2范数来度量)
            f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)  # 对dim1求和的意义：求距离与所有的维度都相关,相当于对512个向量求2范数
            if center is None:
                c_2 = torch.sum(torch.pow(self.centers, 2), dim=1, keepdim=True)
                dist = f_2 - 2 * torch.matmul(features, torch.transpose(self.centers, 1, 0)) + torch.transpose(c_2, 1,
                                                                                                               0)
            else:  # default compute: d_e(features, P^k)
                c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)  # 对20个128维度的向量求一个2范数
                dist = f_2 - 2 * torch.matmul(features, torch.transpose(center, 1, 0)) + torch.transpose(c_2, 1,
                                                                                                         0)  # f2:512*1;中间项：512*20；最后项：1*20
            dist = dist / float(features.shape[1])
        else:  # （使用特征和原型的向量点击来度量，dist应该去相反数吧）
            if center is None:
                center = self.centers
            else:
                center = center
            dist = -features.matmul(center.t())  # 512*128x128*20-->512*20
        dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])  # [batch_size, number_class, num_centers]
        dist = torch.mean(dist, dim=2)  # [batch_size, number_class]
        return dist


def map_label(label, classes):
    # 列表转tensor
    classes = torch.tensor(classes)
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label


class SLCPLoss(nn.Module):
    def __init__(self, args):
        self.args = args
        self.weight_pl = float(args.lamda)
        super(SLCPLoss, self).__init__()
        self.Dist = Dist(num_classes=args.num_seen, feat_dim=args.feat_dim).to(torch.device(args.gpu)).to(torch.device(args.gpu))#每个原型的维度为feat_dim
        self.points = self.Dist.centers
        self.zero=nn.Parameter(torch.zeros(args.feat_dim))

    def forward(self, x, labels=None):
        dist_l2_p = self.Dist(x, center=self.points)  #x的维度是b*args.num_seen，原型的维度是feat_dim
        if args.is_softmax:
            logits = F.softmax(-dist_l2_p, dim=1)
        else:
            logits=-dist_l2_p
        if labels is None:
            return logits
        # labels = map_label(labels, self.args.seen_c)  # 如果zsl做200分类的话标签映射不能放外面了
        # labels = labels.to(self.args.gpu)
        loss_main = F.cross_entropy(-dist_l2_p, labels)  # 概率和距离成反相关

        center_batch = self.points[labels, :]  # 训练数据对应的原型：bs*128

        if args.sim=='':
            loss_r = F.mse_loss(x, center_batch) / 2  # 原型要尽量紧凑
        if args.sim=='cos':
            cos = torch.cosine_similarity(x, center_batch, dim=1)
            loss_r=-cos.mean()

        if args.center=='zero':
            o_center = self.zero
        elif args.center=='mean':
            o_center = self.points.mean(0)
        l_ = (self.points - o_center).pow(2).mean(1)
        # loss_outer = torch.exp(-l_.mean(0))
        loss_outer_std = torch.std(l_)  # 原型的位置约束使用方差约束

        loss = loss_main + self.weight_pl * loss_r + args.gama*loss_outer_std
        # self.weight_pl = min(self.weight_pl + 0.0001, 0.2)
        # print('*********loss={0}**********'.format(loss))
        return loss

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features,s=64.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.args=args
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features).cuda(args.gpu))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.criterion_cls=nn.CrossEntropyLoss()

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))#两个向量归一化之后，乘积就是他们的余弦值(weight可以理解成40个原型)
        if not self.training:
            return cosine
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))#求每个样本到原型之间的正弦值
        phi = cosine * self.cos_m - sine * self.sin_m#这个相当正则化？每个样本的m1*cos-m2*sin是什么
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)#如果余弦值足够大，则取phi
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=self.args.gpu)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)#这个是gt的onehot编码
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s#output反映的是是到每个原型的距离
        arc_loss=self.criterion_cls(output,label)
        return arc_loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Dis_Embed_Att(nn.Module):
    def __init__(self, opt):
        super(Dis_Embed_Att, self).__init__()
        self.fc1 = nn.Linear(2048+1024, 2048)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(2048, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, input):
        h = self.lrelu(self.fc1(input))#将3072的属性映射到2048
        h = self.fc2(h)#将2048维度映射到1，最后得到150*1，相当于做分类了
        return h

def caculate_const_loss(embed, semantics,input_label, relation_net):
    all_scores=torch.FloatTensor(embed.shape[0],args.num_seen).cuda()#表征2048个样本与150个类别之间的关系
    for i, i_embed in enumerate(embed):
        expand_embed = i_embed.repeat(args.num_seen, 1)#.reshape(embed.shape[0] * opt.nclass_seen, -1)
        all_scores[i]=(torch.div(relation_net(torch.cat((expand_embed, semantics), dim=1)),0.1).squeeze())#relation_net的作用是将150*3078(2048视觉+1024语义)的维度映射到150*1(一个样本在150个类别的得分),0.1是一个温度系数，可以改
    score_max, _ = torch.max(all_scores, dim=1, keepdim=True)
    # normalize the scores for stable training
    scores_norm = all_scores - score_max.detach()#这个最大的得分应该对应这分类吧，为什么这里要这么减呢？(这么相减之后，刚好gt对应位置的分数为0，其他位置为负数)
    mask = F.one_hot(input_label, num_classes=args.num_seen).float().cuda()#将标签处理成onehot编码
    exp_scores = torch.exp(scores_norm)#样本与语义之间的相关分数
    log_scores = scores_norm - torch.log(exp_scores.sum(1, keepdim=True))#这是在干啥呢？以上几行代码都在归一化吗
    cls_loss = -((mask * log_scores).sum(1) / mask.sum(1)).mean()#mask * log_scores用于取出2048个正样本对的得分;分母在干啥啊，为什么分母不用*log_scores呢(现在的操作岂不是固定了分母永远为1？)
    return cls_loss.cuda(args.gpu)

def dvbe(pretrained=False, loss_params=None, args=None):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Model(args).to(torch.device(args.gpu))
    if pretrained:
        model_dict = model.state_dict()
        # pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        pretrained_dict = torch.load("./data/resnet101-63fe2227.pth")
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


