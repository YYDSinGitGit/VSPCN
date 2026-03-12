import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransZero(nn.Module):
    def __init__(self, config, att, init_w2v_att, seenclass, unseenclass,
                 is_bias=True, bias=1, is_conservative=True):
        super(TransZero, self).__init__()
        self.config = config
        self.dim_f = config.dim_f#2048
        self.dim_v = config.dim_v#300
        self.nclass = config.num_class
        self.seenclass = seenclass
        self.unseenclass = unseenclass
        self.is_bias = is_bias#True
        self.is_conservative = is_conservative#True
        # class-level semantic vectors
        self.att = nn.Parameter(F.normalize(att), requires_grad=False)#属性值是不需要求梯度的，但是属性名的嵌入是需要求梯度的
        # GloVe features for attributes name
        self.V = nn.Parameter(F.normalize(init_w2v_att), requires_grad=True)#这个为什么要求梯度呢？
        # for self-calibration
        self.bias = nn.Parameter(torch.tensor(bias), requires_grad=False)
        mask_bias = np.ones((1, self.nclass))
        mask_bias[:, self.seenclass.cpu().numpy()] *= -1#为什么要将可见类别值为-1呢
        self.mask_bias = nn.Parameter(torch.tensor(
            mask_bias, dtype=torch.float), requires_grad=False)
        # mapping就（这个参数是用来融合语义信息和Transformer解码输出的）
        self.W_1 = nn.Parameter(nn.init.normal_(
            torch.empty(self.dim_v, config.tf_common_dim)), requires_grad=True)
        # transformer
        self.transformer = Transformer(
            ec_layer=config.tf_ec_layer,
            dc_layer=config.tf_dc_layer,
            dim_com=config.tf_common_dim,#300
            dim_feedforward=config.tf_dim_feedforward,#512（前向传播中的维度？）
            dropout=config.tf_dropout,
            SAtt=config.tf_SAtt,#True
            heads=config.tf_heads,#单头注意
            aux_embed=config.tf_aux_embed)#辅助编码
        # for loss computation
        self.log_softmax_func = nn.LogSoftmax(dim=1)#这个其实就交叉熵的一部分
        self.weight_ce = nn.Parameter(torch.eye(self.nclass), requires_grad=False)#构造了一个n*n的对角矩阵，计算损失的时候有用

    def forward(self, input, from_img=False):
        Fs = self.resnet101(input) if from_img else input
        # transformer-based visual-to-semantic embedding
        v2s_embed = self.forward_feature_transformer(Fs)
        # classification
        package = {'pred': self.forward_attribute(v2s_embed),#v2s_embed是视觉语义信息映射到属性空间的结果
                   'embed': v2s_embed}
        package['S_pp'] = package['pred']
        return package#返回logits

    def forward_feature_transformer(self, Fs):
        # visual
        if len(Fs.shape) == 4:
            shape = Fs.shape
            Fs = Fs.reshape(shape[0], shape[1], shape[2] * shape[3])#将图形的二维特征处理成一维的序列特征
        Fs = F.normalize(Fs, dim=1)
        # attributes
        V_n = F.normalize(self.V) if self.config.normalize_V else self.V#属性名称的编码
        # locality-augmented visual features(进行视觉和语义的交互：做了两件事：1去几何相关  2属性定位)
        Trans_out = self.transformer(Fs, V_n)#给定一个视觉信息和一个属性语义信息，使用Transformer：进行定位增强视觉信息？视觉和语义信息经过编码解码之后得到300*312维度的信息（可理解成一个语义信息）
        # embedding to semantic space
        embed = torch.einsum('iv,vf,bif->bi', V_n, self.W_1, Trans_out)#这个就是融合了语义编码和Transformer输出
        return embed#已完成将视觉信息映射到了属性编码空间

    def forward_attribute(self, embed):#最后预测属性的时候用（参数是Transformer的输出）
        embed = torch.einsum('ki,bi->bk', self.att, embed)#
        self.vec_bias = self.mask_bias*self.bias
        embed = embed + self.vec_bias
        return embed
    #这个损失可以用到dvbe的地方
    def compute_loss_Self_Calibrate(self, in_package):
        S_pp = in_package['pred']#这个是预测标签（200个类别）
        Prob_all = F.softmax(S_pp, dim=-1)#归一化输出成概率
        Prob_unseen = Prob_all[:, self.unseenclass]#输出为unseen类的概率
        assert Prob_unseen.size(1) == len(self.unseenclass)
        mass_unseen = torch.sum(Prob_unseen, dim=1)
        loss_pmp = -torch.log(torch.mean(mass_unseen))
        return loss_pmp

    def compute_aug_cross_entropy(self, in_package):
        Labels = in_package['batch_label']
        S_pp = in_package['pred']

        if self.is_bias:
            S_pp = S_pp - self.vec_bias#可见类样本的预测概率要减一分

        if not self.is_conservative:
            S_pp = S_pp[:, self.seenclass]
            Labels = Labels[:, self.seenclass]
            assert S_pp.size(1) == len(self.seenclass)

        Prob = self.log_softmax_func(S_pp)#把输出压缩到：-无穷~0（为什么不直接使用交叉熵损失呢）

        loss = -torch.einsum('bk,bk->b', Prob, Labels)#将Transformer的输出概率与属性算爱因斯坦积，要求这个乘积越大越好，添了一个-号，则应该越小越好（与原文说的交叉熵损失对不上啊）
        loss = torch.mean(loss)
        return loss

    def compute_reg_loss(self, in_package):
        tgt = torch.matmul(in_package['batch_label'], self.att)
        embed = in_package['embed']
        loss_reg = F.mse_loss(embed, tgt, reduction='mean')
        return loss_reg

    def compute_loss(self, in_package):
        if len(in_package['batch_label'].size()) == 1:
            in_package['batch_label'] = self.weight_ce[in_package['batch_label']]#self.weight_ce是一个200*200的对角矩阵,这里获得了训练样本gt的one-hot编码

        loss_CE = self.compute_aug_cross_entropy(in_package)#原文说的是交叉熵损失，但是里面只是一个简单的乘积
        loss_cal = self.compute_loss_Self_Calibrate(in_package)#自校验损失
        loss_reg = self.compute_reg_loss(in_package)#重构损失使用均方损失

        loss = loss_CE + self.config.lambda_ * \
            loss_cal + self.config.lambda_reg * loss_reg
        out_package = {'loss': loss, 'loss_CE': loss_CE,
                       'loss_cal': loss_cal, 'loss_reg': loss_reg}
        return out_package


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


class MultiLevelEncoder_woPad(nn.Module):
    def __init__(self, N, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048,
                 dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder_woPad, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,#编码层里面有一个attention和两个全连接层：300--》512  和512--》300
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])

        self.WGs = nn.ModuleList(#定义了一个线性层
            [nn.Linear(64, 1, bias=True) for _ in range(h)])
#总结：多水平编码==attention+线性
    def forward(self, input, attention_mask=None, attention_weights=None, pos=None):
        out = input
        for layer in self.layers:#这里不要做交叉注意，也用不到相对几何关系
            out = layer(out, out, out, relative_geometry_weights,
                        attention_mask, attention_weights, pos=pos)
        return out#编码完成


class TransformerDecoderLayer(nn.TransformerDecoderLayer):#根据自己的任务需要重写解码器
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", SAtt=True):
        super(TransformerDecoderLayer, self).__init__(d_model, nhead,
                                                      dim_feedforward=dim_feedforward,
                                                      dropout=dropout,
                                                      activation=activation)
        self.SAtt = SAtt#就这？

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,#tgt是属性信息，memory损失编码信息
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



