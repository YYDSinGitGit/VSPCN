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


from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from collections import OrderedDict

import pickle

_tokenizer = _Tokenizer()


with open('models/config_s.json', 'r') as f:
    config_s=json.load(f)
config_s = argparse.Namespace(**config_s)
config_s.num_hidden_layers=args.l
config_s.num_attention_heads=args.h

with open('models/config_c.json', 'r') as f:
    config_c=json.load(f)
config_c = argparse.Namespace(**config_c)

config_c.num_hidden_layers=args.l2
config_c.num_attention_heads=args.h2


import re
from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['agpl']



class PromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        self.relu = nn.LeakyReLU()
        if args.att:
            self.sf = torch.tensor(args.sf, device=args.gpu)
            self.sf=(self.sf).to(torch.float16)

            self.sf = torch.tensor(args.sf, device=args.gpu)
            self.sf = (self.sf).to(torch.float16)

            # self.a2p1 = nn.Linear(args.natt, 512)
            # self.a2p2 = nn.Linear(512, 512 * args.nctx)
            self.a2p1 = nn.Linear(args.natt, args.natt)#mix att
            self.a2p2 = nn.Linear(args.natt, 512 * args.nctx)#312-->312-->1

            # 使用Xavier随机初始化方法初始化a2p1和a2p2的权重
            nn.init.xavier_uniform_(self.a2p1.weight)
            nn.init.xavier_uniform_(self.a2p2.weight)

            # 使用常数初始化方法初始化a2p1和a2p2的偏置
            nn.init.constant_(self.a2p1.bias, 0)
            nn.init.constant_(self.a2p2.bias, 0)

            # 将a2p1和a2p2的参数类型转换为float16
            self.a2p1.weight.data = self.a2p1.weight.data.to(torch.float16)
            self.a2p1.bias.data = self.a2p1.bias.data.to(torch.float16)
            self.a2p2.weight.data = self.a2p2.weight.data.to(torch.float16)
            self.a2p2.bias.data = self.a2p2.bias.data.to(torch.float16)

            if not args.csc:
                self.a2p3=nn.Linear(args.num_classes,1)#project 200 att to a general prompt
                nn.init.xavier_uniform_(self.a2p3.weight)
                nn.init.constant_(self.a2p3.bias, 0)
                self.a2p3.weight.data = self.a2p3.weight.data.to(torch.float16)
                self.a2p3.bias.data = self.a2p3.bias.data.to(torch.float16)

        if args.att:
            self.a2p = nn.Linear(args.natt, 512*args.nctx)
            self.relu = nn.LeakyReLU()
            nn.init.xavier_uniform_(self.a2p.weight)
            nn.init.constant_(self.a2p.bias, 0)
            self.a2p.weight.data = self.a2p.weight.data.to(torch.float16)
            self.a2p.bias.data = self.a2p.bias.data.to(torch.float16)
            if not args.csc:
                self.a2p3 = nn.Linear(args.num_classes, 1)
                nn.init.xavier_uniform_(self.a2p3.weight)
                nn.init.constant_(self.a2p3.bias, 0)
                self.a2p3.weight.data = self.a2p3.weight.data.to(torch.float16)
                self.a2p3.bias.data = self.a2p3.bias.data.to(torch.float16)

                # self.a2p4 = nn.Linear(args.num_classes, args.nctx)
                # nn.init.xavier_uniform_(self.a2p4.weight)
                # nn.init.constant_(self.a2p4.bias, 0)
                # self.a2p4.weight.data = self.a2p4.weight.data.to(torch.float16)
                # self.a2p4.bias.data = self.a2p4.bias.data.to(torch.float16)
                #
                # self.a2p5 = nn.Linear(256, 512)
                # nn.init.xavier_uniform_(self.a2p5.weight)
                # nn.init.constant_(self.a2p5.bias, 0)
                # self.a2p5.weight.data = self.a2p5.weight.data.to(torch.float16)
                # self.a2p5.bias.data = self.a2p5.bias.data.to(torch.float16)


        n_cls = len(classnames)
        n_ctx = args.nctx
        ctx_init = args.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = args.in_size
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # att
        if args.att_semantic:
            embedding = torch.cat([clip.tokenize(p) for p in args.att_name])
            att_name_lens = [len(_tokenizer.encode(name)) for name in args.att_name]
            max_att_len=max(att_name_lens)
            # max_att_len=3

            with torch.no_grad():
                embedding = clip_model.token_embedding(embedding).type(dtype)
            att_name_embedding=embedding[:,1:max_att_len+1,:]



            if args.fill_zeros:
                att_name_embedding = embedding[:, 1:max_att_len+1, :]
                num_w = torch.tensor(att_name_lens)
                ones = torch.ones((args.sf.shape[1], max_att_len), dtype=dtype)
                mask = torch.arange(max_att_len).expand_as(ones) >= num_w.unsqueeze(1)
                ones[mask] = 0
                ones = ones.unsqueeze(-1).expand(-1, -1, 512)



                self.gp=nn.Parameter((att_name_embedding*ones).mean(dim=0))


                if args.avg1:
                    # self.gp=nn.Parameter((att_name_embedding*ones).mean(dim=0))
                    with torch.no_grad():
                        embedding = clip.tokenize('a photo of')
                    embedding = clip_model.token_embedding(embedding).type(dtype)
                    embedding = embedding.view(embedding.shape[1], embedding.shape[2])
                    gp=embedding[1:4,:]
                    self.gp=nn.Parameter(gp)
                elif args.avg2:
                    self.gp=(att_name_embedding*ones).mean(dim=0)
                    self.gp=nn.Parameter(self.gp)
                    # self.gp = self.gp.mean(dim=0)
                    # self.gp = self.gp.view(1,512)
                    # self.gp=nn.Parameter(self.gp)

                else:
                    self.gp = nn.Parameter(att_name_embedding * ones)

                    self.an2p=nn.Linear(args.sf.shape[1],1)
                    self.an2p.weight.data = self.an2p.weight.data.to(torch.float16)
                    self.an2p.bias.data = self.an2p.bias.data.to(torch.float16)
                    nn.init.xavier_uniform_(self.an2p.weight)
                    nn.init.xavier_uniform_(self.an2p.weight)

        if args.instance:
            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(512, 512 // 16,dtype=torch.float16)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(512 // 16, 512,dtype=torch.float16))
            ]))

        if args.w2v:
            w2v = torch.from_numpy(args.w_to_v)
            w2v = w2v.to(torch.float16)
            self.w2v=w2v.to(args.gpu)
            self.w2v_prj1=nn.Linear(args.w_to_v.shape[1],512,dtype=torch.float16)
            self.w2v_prj2 = nn.Linear(args.w_to_v.shape[0], 1, dtype=torch.float16)



        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if args.csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)#[102  10  512]
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)#[10  512]
            nn.init.normal_(ctx_vectors, std=0.02)#different between ctx_vectors and prompt_prefix

            if args.att_semantic or args.w2v:

                if args.avg1:
                    n_semantic=max_att_len
                elif args.avg2:
                    n_semantic=max_att_len
                elif args.w2v:
                    n_semantic=1
                else:
                    n_semantic=max_att_len
            n_fill=n_ctx if not (args.att_semantic or args.w2v) else n_ctx+n_semantic
            prompt_prefix = " ".join(["X"] * n_fill)#how to use?




        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]#not the number of word
        prompts = [prompt_prefix + " " + name + "." for name in classnames]#xxxxxx + name

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)#clip_model.token_embedding=nn.embedding   [102, 77, 512]
            #why clip.tokenize + nn.embedding?
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS  [102, 1, 512]

        n_semantic=0#Temporary test
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx+n_semantic:, :])  # CLS, EOS(add a cls position)    [102, 66, 512](why the len of cls_name==1?)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor [102, 77] clip.tokenizer('xxxxxxxx+name')
        self.name_lens = name_lens
        self.class_token_position = args.position

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self,imgfeat=None):
        prefix = self.token_prefix  # [102 1 512] all the prefix[i] are same(sos)
        suffix = self.token_suffix

        if args.att_semantic:
            gp = self.gp
            if (not args.avg1 and not args.avg2):
                gp = self.an2p(gp.view(-1, args.sf.shape[1]))
                gp = self.relu(gp)
                gp = gp.view(-1, 512)

            #gp=self.gp.unsqueeze(0).expand(self.n_cls,-1,-1)
            gp = gp.unsqueeze(0).expand(self.n_cls, -1, -1)

        if args.w2v:
            gp=self.w2v_prj1(self.w2v)
            gp=self.relu(gp)
            gp = self.w2v_prj2(gp.transpose(0,1))
            gp=gp.transpose(0,1)

            gp = gp.unsqueeze(0).expand(self.n_cls, -1, -1)

        # [102 66 512]  suffix[:,-1,:] represent EOS
        if args.instance:
            if args.att:
                ctx = self.a2p1(self.sf)
                ctx = self.relu(ctx)
                ctx = self.a2p2(ctx)
                ctx = self.relu(ctx)
            if not args.csc:
                ctx=self.a2p3(ctx.view(ctx.shape[1],ctx.shape[0]))
                ctx=self.relu(ctx)
                ctx=ctx.view(args.nctx,-1)

            bias=self.meta_net(imgfeat)
            bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
            ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
            ctx_shifted = ctx + bias # imgfeat is considered as bias
            prompts = []
            if args.att_semantic:
                prefix=torch.cat([prefix,gp],dim=1)
            for ctx_shifted_i in ctx_shifted:
                ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
                pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
                prompts.append(pts_i)
            prompts = torch.stack(prompts)

            return prompts

        if args.att:
            ctx = self.a2p1(self.sf)     #Nc x Na->Nc x 512
            ctx = self.relu(ctx)

            # ctx = self.a2p5(ctx)  # Nc x Na->Nc x 512
            # ctx = self.relu(ctx)
            #
            # ctx = self.a2p4(ctx.view(ctx.shape[1], ctx.shape[0]))  # Nc x (512*nctx)  ->   nctx x 512
            # ctx = self.relu(ctx)
            # ctx = ctx.view(args.nctx, -1)

            if (args.nctx!=0):
                ctx = self.a2p2(ctx)#if nctx==1,this is redundant       Nc x 512-> Nc x (512*nctx)
                ctx = self.relu(ctx)
            # ctx = self.a2p(self.sf)

            if not args.csc:
                ctx=self.a2p3(ctx.view(ctx.shape[1],ctx.shape[0])) #Nc x (512*nctx)  ->   nctx x 512
                ctx=self.relu(ctx)
                ctx=ctx.view(args.nctx,-1)

            else:
                ctx = ctx.view(-1, args.nctx, 512)

        else:
            ctx = self.ctx





        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)


        if self.class_token_position == "end":
            if args.att_semantic or args.w2v:
                prefix=torch.cat([prefix,gp],dim=1)

            prompts = torch.cat(
                [

                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]#get sos
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim) question:suffix_i embed by 'xxxxx + name','xxx' is init randomly,but suffix_i.requires_grad=False
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class ZSL(nn.Module):
    def __init__(self,args):
        super().__init__()

        clip_model = load_clip_to_cpu(args)
        self.prompt_learner = PromptLearner(args, args.classname, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts#[102, 77] clip.tokenizer('xxxxxxxx+name')
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        logit_scale = self.logit_scale.exp()
        image_features = self.image_encoder(image.type(self.dtype))

        if args.instance:
            prompts = self.prompt_learner(image_features)

            logits = []
            for pts_i, imf_i in zip(prompts, image_features):
                text_features = self.text_encoder(pts_i, self.tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                l_i = logit_scale * imf_i @ text_features.t()
                logits.append(l_i)
            logits = torch.stack(logits)

            # if self.prompt_learner.training:
            #     return F.cross_entropy(logits, label)

            return logits

        else:
            prompts = self.prompt_learner()

        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)


        logits = logit_scale * image_features @ text_features.t()

        return logits

def load_clip_to_cpu(args):
    url = clip._MODELS[args.backbone]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

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

            if args.norm=='True':
                features_norm=F.normalize(features,dim=1)
                center_norm=F.normalize(center,dim=1)
                dist = -features_norm.matmul(center_norm.t())
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
        self.points = self.Dist.centers#随机初始化的原型
        # self.zero=nn.Parameter(torch.zeros(args.feat_dim))

    def forward(self, x, labels=None):
        dist_l2_p = self.Dist(x, center=self.points)  #x的维度是b*args.num_seen，原型的维度是feat_dim
        if args.is_softmax and args.norm=='':
            logits = F.softmax(-dist_l2_p, dim=1)
        else:
            logits=-dist_l2_p
        if labels is None:
            return logits
        # labels = map_label(labels, self.args.seen_c)  # 如果zsl做200分类的话标签映射不能放外面了
        # labels = labels.to(self.args.gpu)
        if args.norm=='True':
            idx = torch.arange(logits.size(0)).long()
            loss_main = (-logits[idx, labels]).mean()
        else:
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


class classifer(nn.Module):
    def __init__(self):
        super(classifer, self).__init__()
        self.lin=nn.Linear(512,args.num_seen)
        self.optim=torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.dtype = torch.float16
    def forward(self,input):
        output=self.lin(input)
        return output





def agpl(args):
    model = ZSL(args)
        # model_dict = model.state_dict()
        # # pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        # pretrained_dict = torch.load("./data/resnet101-63fe2227.pth")
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
    return model


