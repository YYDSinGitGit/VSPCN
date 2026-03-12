import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Transformer(nn.Module):
    def __init__(self, ec_layer=1, dc_layer=1, dim_com=300,
                 dim_feedforward=2048, dropout=0.1, heads=1,
                 in_dim_cv=2048, in_dim_attr=300, SAtt=True,
                 aux_embed=True):
        super(Transformer, self).__init__()
        # input embedding（cv：2048-->300）
        self.embed_cv = nn.Sequential(nn.Linear(in_dim_cv, dim_com))
        if aux_embed:#辅助编码哪里用到了呢
            self.embed_cv_aux = nn.Sequential(nn.Linear(in_dim_cv, dim_com))
        self.embed_attr = nn.Sequential(nn.Linear(in_dim_attr, dim_com))#这个对于awa2其输入特征应该是85维度吧
        # transformer encoder
        self.transformer_encoder = MultiLevelEncoder_woPad(N=ec_layer,
                                                           d_model=dim_com,#模型的维度300，指的是Transformer的输入维度吗
                                                           h=1,
                                                           d_k=dim_com,
                                                           d_v=dim_com,
                                                           d_ff=dim_feedforward,
                                                           dropout=dropout)

        # my encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_com, nhead=1)#默认用8头
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)#默认用6层
        # src = torch.rand(10, 32, 512)
        # out = transformer_encoder(src)



        # transformer decoder
        decoder_layer = TransformerDecoderLayer(d_model=dim_com,#这里的解码层为啥要重写TransformerDecoderLayer呢？
                                                nhead=heads,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                SAtt=SAtt)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=dc_layer)


    def forward(self, f_cv, f_attr):
        # linearly map to common dim（300）
        h_cv = self.embed_cv(f_cv.permute(0, 2, 1))
        h_attr = self.embed_attr(f_attr)
        h_attr_batch = h_attr.unsqueeze(0).repeat(f_cv.shape[0], 1, 1)
        # visual encoder
        encod_feat=self.transformer_encoder(h_cv)
        return encod_feat


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048,
                 dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadGeometryAttention(d_model, d_k, d_v, h, dropout,#多头几何注意力是编码层的主要成分
                                                identity_map_reordering=identity_map_reordering,
                                                attention_module=attention_module,
                                                attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(#为什么有位置聪明前向传播网络？定义了两个全连接层
            d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights,
                attention_mask=None, attention_weights=None, pos=None):
        q, k = (queries + pos, keys +
                pos) if pos is not None else (queries, keys)#没有使用位置嵌入，kq的值直接用的是视觉信息
        att = self.mhatt(q, k, values, relative_geometry_weights,
                         attention_mask, attention_weights)#做了自注意+去几何相关
        att = self.lnorm(queries + self.dropout(att))#残差连接+layer归一化操作
        ff = self.pwff(att)#最后经过两个全连接层
        return ff#到这里编码完成了



class TransformerDecoderLayer(nn.TransformerDecoderLayer):#根据自己的任务需要重写解码器
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", SAtt=False):#不用做selfattention了吧
        super(TransformerDecoderLayer, self).__init__(d_model, nhead,
                                                      dim_feedforward=dim_feedforward,
                                                      dropout=dropout,
                                                      activation=activation)
        self.SAtt = SAtt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,#这文章相当于对语义信息先做了一个子注意力再输入交互
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if self.SAtt:
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,#做自注意力
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

# class TransformerEncoderLayer(nn.TransformerEncoderLayer):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
#                  activation="relu", SAtt=True):
#         super(TransformerDecoderLayer, self).__init__(d_model, nhead,
#                                                       dim_feedforward=dim_feedforward,
#                                                       dropout=dropout,
#                                                       activation=activation)
#         self.SAtt = SAtt#就这？
#
#     def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
#                 tgt_key_padding_mask=None, memory_key_padding_mask=None):
#         if self.SAtt:
#             tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
#                                   key_padding_mask=tgt_key_padding_mask)[0]
#             tgt = self.norm1(tgt + self.dropout1(tgt2))
#         tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
#                                    key_padding_mask=memory_key_padding_mask)[0]
#         tgt = tgt + self.dropout2(tgt2)
#         tgt = self.norm2(tgt)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout3(tgt2)
#         tgt = self.norm3(tgt)
#         return tgt



