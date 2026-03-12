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
from models.attention import *
import torchvision.models.resnet as models

import argparse
import json

from collections import OrderedDict

import pickle

depth_ref = json.loads(args.depth_ref)
defusionV_inVit = True
defusionL_inVit = True

att_map=0

with open('models/config_s.json', 'r') as f:
    config_s = json.load(f)
config_s = argparse.Namespace(**config_s)
config_s.num_hidden_layers = args.l
config_s.num_attention_heads = args.h

with open('models/config_c.json', 'r') as f:
    config_c = json.load(f)
config_c = argparse.Namespace(**config_c)

config_c.num_hidden_layers = args.l2
config_c.num_attention_heads = args.h2

import re
from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['smp']


from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn


Norm = nn.LayerNorm


def truncated_normal_(tensor, mean=0, std=0.09):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def trunc_normal_(tensor, mean=0, std=.01):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class W2v_Attention(nn.Module):
    def __init__(self, dim, qkv_bias=False):
        super(W2v_Attention, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.toto0 = nn.Linear(dim, dim, bias=qkv_bias)
        self.act = nn.ReLU()
        self.scale = dim ** (-0.5)
        self.proj0 = nn.Linear(dim, dim)

    def forward(self, q, k, v):
        q, k, v = self.norm(q), self.norm(k), self.norm(v)
        q, k, v = self.toto0(q), self.toto0(k), self.toto0(v)
        attn = torch.einsum("qc,kc->qk", q, k)
        attn *= self.scale
        attn_softed = F.softmax(attn, dim=-1)
        out = torch.einsum("qk,kc->qc", attn_softed, v.float())
        out = self.proj0(out)

        return out


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 layer=0):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.layer = layer

    def forward(self, x):
        B, N, C = x.shape
 
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 layer=0,
                 w2v=None):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio,layer=layer)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        self.layer = layer

        if args.channel:  # true
            args.attention = False
            if args.attention:  # false
                self.channel_encoder_layer = nn.TransformerEncoderLayer(d_model=1, nhead=1,
                                                                        batch_first=True)
                self.transformer_channel = nn.TransformerEncoder(self.channel_encoder_layer, num_layers=1)
            else:
                self.channel_project1 = nn.Linear(in_features=768, out_features=384, bias=True)
                self.activate = nn.Sigmoid()
                self.channel_project2 = nn.Linear(in_features=384, out_features=768, bias=True)

        global depth_ref, defusionV_inVit, defusionL_inVit
        if self.layer in depth_ref:
            if defusionV_inVit:
                self.qV_norm, self.kV_norm, self.vV_norm = nn.LayerNorm(768), nn.LayerNorm(768), nn.LayerNorm(768)
                self.q_v00 = nn.Parameter(torch.zeros(768, 768), requires_grad=True)
                self.k_v00 = nn.Parameter(torch.zeros(768, 768), requires_grad=True)
                self.v_v00 = nn.Parameter(torch.zeros(768, 768), requires_grad=True)
                self.scores = nn.Parameter(torch.zeros(1, 196), requires_grad=True)
            if defusionL_inVit:
                self.w2vL = torch.from_numpy(w2v).to(torch.device(args.gpu)).to(torch.float32)
                self.V = nn.Parameter(trunc_normal_(torch.empty(300, 768)), requires_grad=True).to(
                    torch.device(args.gpu))
                self.w2v_ = 0
                self.qL_norm, self.kL_norm, self.vL_norm = norm_layer(768), norm_layer(768), norm_layer(768)
                self.q_w00 = nn.Linear(768, 768, bias=True)
                self.k_w00 = nn.Linear(768, 768, bias=True)
                self.v_w00 = nn.Linear(768, 768, bias=True)
                self.q_w0 = nn.Linear(768, 768, bias=True)
                self.k_w0 = nn.Linear(768, 768, bias=True)
                self.v_w0 = nn.Linear(768, 768, bias=True)
                if args.data == 'cub':
                    self.scoresL = nn.Parameter(torch.zeros(1, 312), requires_grad=True)
                elif args.data == 'sun':
                    self.scoresL = nn.Parameter(torch.zeros(1, 102), requires_grad=True)
                elif args.data == 'awa2':
                    self.scoresL = nn.Parameter(torch.zeros(1, 85), requires_grad=True)


    def forward(self, x):
        f_ori = x[3]
        f_res = x[2]
        wTov = x[1]
        x = x[0]
        global depth_ref, defusionV_inVit, defusionL_inVit
        if self.layer in depth_ref:
            B, N, C = x.shape
            if defusionV_inVit:
                q_00 = x[:, 1:2, :]  
                k_00 = x[:, 3:, :].detach()
                v_00 = x[:, 3:, :].detach()
                query, key, value = q_00, k_00, v_00
                query = torch.matmul(query, self.q_v00)
                key = torch.matmul(key, self.k_v00)
                value = torch.matmul(value, self.v_v00)
                scores = torch.einsum("bnd,bmd->bnm", query, key) / C ** .5
                if args.data=='cub':
                    softmax_qk = args.va*torch.nn.functional.softmax(scores, dim=-1)  +  \
                                 (1-args.va)*torch.nn.functional.softmax(self.scores.expand(B, -1, -1), dim=-1)
                    f_m0 = torch.einsum("bnm,bmd->bnd", softmax_qk, value)
                    f_m0 = 0.9*q_00+0.1*f_m0
                elif args.data=='sun':
                    softmax_qk = 0.0 * torch.nn.functional.softmax(scores, dim=-1) + \
                                 1.0 * torch.nn.functional.softmax(self.scores.expand(B, -1, -1), dim=-1)
                    f_m0 = torch.einsum("bnm,bmd->bnd", softmax_qk, value)
                    f_m0 = 0.9 * q_00 + 0.1 * f_m0
                elif args.data =='awa2':
                    softmax_qk = 0.95 * torch.nn.functional.softmax(scores, dim=-1) + \
                                 0.05 * torch.nn.functional.softmax(self.scores.expand(B, -1, -1), dim=-1)
                    f_m0 = torch.einsum("bnm,bmd->bnd", softmax_qk, value)
                    f_m0 = 0.1 * q_00 + 0.9 * f_m0
                x[:, 1:2, :] = f_m0[:, 0:1, :]

            if defusionL_inVit:
                q_00 = wTov.unsqueeze(dim=0).expand(B, -1, -1)
                k_00 = x[:, 3:, :].detach()
                v_00 = x[:, 3:, :].detach()
                query, key, value = q_00, k_00, v_00
                query, key, value = self.q_w00(query), self.k_w00(key), self.v_w00(value)
                scores = torch.einsum("bnd,bmd->bnm", query, key) / C ** .5
                softmax_qk = torch.nn.functional.softmax(scores, dim=-1)
                f_m0 = torch.einsum("bnm,bmd->bnd", softmax_qk, value)
                wTov = 0.5*q_00 + 0.5*f_m0

                q_0 = x[:, 2:3, :]  
                k_0 = wTov
                v_0 = wTov
                query, key, value = q_0, k_0, v_0
                query, key, value = self.q_w0(query), self.k_w0(key), self.v_w0(value)
                scores = torch.einsum("bnd,bmd->bnm", query, key) / C ** .5
                if args.data =="cub":
                    a = args.la*torch.nn.functional.softmax(scores, dim=-1)
                    b = (1-args.la)* torch.nn.functional.softmax(self.scoresL.expand(B, -1, -1), dim=-1)
                    softmax_qk = a + b
                    f_m0 = torch.einsum("bnm,bmd->bnd", softmax_qk, value)
                    f_m0 = q_0 + f_m0
                    x[:, 2:3, :] = x[:, 2:3, :]+ f_m0
                elif args.data=="sun":
                    a = 0.5 * torch.nn.functional.softmax(scores, dim=-1)
                    b = 0.5 * torch.nn.functional.softmax(self.scoresL.expand(B, -1, -1), dim=-1)
                    softmax_qk = a + b
                    f_m0 = torch.einsum("bnm,bmd->bnd", softmax_qk, value)
                    f_m0 = q_0 + f_m0
                    x[:, 2:3, :] = x[:, 2:3, :] + f_m0
                elif args.data=="awa2":
                    a = 0.8 * torch.nn.functional.softmax(scores, dim=-1)
                    b = 0.2 * torch.nn.functional.softmax(self.scoresL.expand(B, -1, -1), dim=-1)
                    softmax_qk = a + b
                    f_m0 = torch.einsum("bnm,bmd->bnd", softmax_qk, value)
                    f_m0 = q_0 + f_m0
                    x[:, 2:3, :] = x[:, 2:3, :] + f_m0
                wTov = wTov.mean(dim=0)
            
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return [x, wTov, f_res, f_ori]


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, att_name_emb=None,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.Vprompt_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True).to(torch.device(args.gpu))
        self.Lprompt_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True).to(torch.device(args.gpu))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_embed_prompt = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.prompt = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer, layer=i, w2v=args.w2v)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.att = torch.from_numpy(args.att).to(torch.device(args.gpu))
        self.project_att = nn.Linear(args.att.shape[1], embed_dim)

        if args.rand_init:  # false
            self.prompt = nn.Parameter(torch.rand(1, 1, embed_dim))
        else:
            self.att_name_emb = torch.from_numpy(att_name_emb).to(torch.device(args.gpu))
            self.att_name_emb = self.att_name_emb.float()
            self.project = nn.Linear(att_name_emb.shape[1], embed_dim)
            self.project2 = nn.Linear(att_name_emb.shape[0], args.np)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.Vprompt_token, std=0.02)
        nn.init.trunc_normal_(self.Lprompt_token, std=0.02)
        self.apply(_init_vit_weights)

        self.w2v = torch.from_numpy(args.w2v).to(torch.device(args.gpu)).to(torch.float32)
        self.V = nn.Parameter(trunc_normal_(torch.empty(300, 768)), requires_grad=True)
        self.prompt_w2v = nn.Embedding(1, 768).weight
        self.w2v_p = W2v_Attention(dim=768, qkv_bias=False)
        self.p_w2v = W2v_Attention(dim=768, qkv_bias=False)
        self.norm_x = nn.LayerNorm(768)
        
        self.R1 = nn.Linear(2048, 768, bias=True)
        self.qkv_norm = nn.LayerNorm(768)
        self.q_r0 = nn.Linear(768, 768, bias=True)
        self.k_r0 = nn.Linear(768, 768, bias=True)
        self.v_r0 = nn.Linear(768, 768, bias=True)


    def base_module(self, atten_attr, global_feat, seen_att, att_all, flag):
        N, C = global_feat.shape
        global_feat = global_feat
        gs_feat = torch.einsum('bc,cd->bd', global_feat, self.V1)
        gs_feat = F.softmax(atten_attr, dim=-1) * gs_feat + gs_feat
        return gs_feat

    def VSIMModule(self, x):
        N, C, W, H = x.shape
        x = x.reshape(N, C, W * H)
        query = torch.einsum('lw,wv->lv', self.w2v, self.W)
        atten_map = torch.einsum('lv,bvr->blr', query, x)
        intensity = self.ca(atten_map.view(N, -1, W, H))
        intensity = F.max_pool2d(intensity, kernel_size=(W, H)).view(N, -1)
        return intensity

    def forward_features(self, img):
        x = self.patch_embed(img)
        p_feat = self.Vprompt_token.expand(x.shape[0], -1, -1)
        
        f_res = 0
        f_ori = 0
        f_avg = 0
        q_0, k_0, v_0 = self.Vprompt_token.expand(x.shape[0],-1,-1), x.detach(), x.detach()  # [1,1,768]
        query, key, value = self.qkv_norm(q_0), self.qkv_norm(k_0), self.qkv_norm(v_0)
        query, key, value = self.q_r0(query), self.k_r0(key), self.v_r0(value)
        scores = torch.einsum("bnd,bmd->bnm", query, key) / (768 ** .5)
        softmax_qk = torch.nn.functional.softmax(scores, dim=-1)
        f_m0 = torch.einsum("bnm,bmd->bnd", softmax_qk, value)
        p_feat = f_m0


        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        v_prompt = p_feat
        l_prompt = self.Lprompt_token.expand(x.shape[0], -1, -1)

        self.w2v_ = torch.einsum('bc,cd->bd', self.w2v, self.V)
        self.prompt2 = self.p_w2v(self.prompt_w2v, self.w2v_, self.w2v_)

        if args.rand_init:  # false
            prompt = self.prompt.expand(x.shape[0], -1, -1)
        else:
            prompt = self.prompt2.expand(x.shape[0], -1, -1)

        if self.dist_token is None:
            x = torch.cat((prompt, x), dim=1)
            x = torch.cat((v_prompt, x), dim=1)
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        pos_p_embed = torch.cat((self.pos_embed[:, :1, :], self.pos_embed_prompt), dim=1)
        pos_p_embed = torch.cat((pos_p_embed, self.pos_embed[:, 1:, :]), dim=1)
        x = self.pos_drop(x + pos_p_embed)
        x = self.blocks([x, self.w2v_, f_res, f_ori])
        w2v = x[1]
        x = x[0]
        x = self.norm(x)

        return x[:, 0], x[:, 1], x[:, 2], x[:, 3:], f_avg, w2v

    def forward(self, x):
        x = x.to(torch.device(args.gpu))
        x_head, v_prompt, l_prompt, x_features, f_avg, w2v = self.forward_features(x)
        att = self.project_att(self.att)
        l_class = torch.einsum('bw,wc->bc', l_prompt, self.class_l)
        v_class = torch.einsum('bw,wc->bc', v_prompt, self.class_v)
        return x_head, v_prompt, l_prompt, att, x_features, f_avg, l_class, v_class


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(args):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=13,  # 12
                              num_heads=12,
                              representation_size=None,
                              num_classes=args.num_classes,
                              att_name_emb=args.att_name_emb)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model


def map_label(label, classes):
    classes = torch.tensor(classes)
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label


def smp(args):
    vit_model = vit_base_patch16_224(args)
    vit_model_path = "./pretrain/vit_base_patch16_224.pth"
    weights_dict = torch.load(vit_model_path)
    del_keys = ['head.weight', 'head.bias'] if vit_model.has_logits \
        else ['head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    vit_model.load_state_dict(weights_dict, strict=False)
    return vit_model


