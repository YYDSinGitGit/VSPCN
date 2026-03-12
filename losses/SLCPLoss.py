import torch
import torch.nn as nn
import torch.nn.functional as f
from losses.Dist import Dist


class SLCPLoss(nn.CrossEntropyLoss):#可以看出这个损失是在交叉熵的基础上改进的
    def __init__(self, **options):
        super(SLCPLoss, self).__init__()#slcp损失继承了交叉熵损失
        self.use_gpu = options['gpu']
        self.weight_pl = float(options['apha'])#pl损失的权重，默认为0.1
        self.Dist = Dist(num_classes=options['seen_num_c'], feat_dim=options['feat_dim'])#参数传入一个识别类别数量和一个特征维度：  1：6  参数2：128（这个应该是求距离的吧）
        self.points = self.Dist.centers#随机初始化150个原型中心，每个点都是32896维度的
        self.seen_c= options['seen_c']
        self.device=options['gpu']

    '''标签映射2022.4.13,你这里只有将可见类的标签映射，但是不可见类进来怎么测试呢'''
    def labels_map(self,labels):
        seen_c=self.seen_c.tolist()
        map_labels = []  # 将标签处理成一个新的标签
        for i in range(len(labels)):  # 遍历所有的图片
            map_labels.append(seen_c.index(labels[i]))  # 其实就是将6类样的标签按照标签大小顺序处理成0-5
        return map_labels


    def forward(self, x, y, labels,is_train):
        if is_train:
            map_labels=self.labels_map(labels)#标签映射
            map_labels = torch.tensor(map_labels)
            map_labels=map_labels.cuda(self.device)
            dist_l2_p = self.Dist(x, center=self.points)#传入隐藏层特征：n*c，传入所有的原型（awa2共40个可见类）的中心：seenclass*c，求点到原型中心的距离，返回[batch, seenclass]的张量，表示batch个样本距离seenclass个原型中心的距离
            logits = f.softmax(-dist_l2_p, dim=1)#对第一个维度seenclass进行归一化操作（）每一列求和为1
            if labels is None:
                return logits, 0
        # loss_main = f.cross_entropy(-dist_l2_p, labels)#主损失用交叉熵
            loss_main = f.cross_entropy(-dist_l2_p, map_labels)#为什们用距离和标签可以计算交叉熵损失？
            center_batch = self.points[map_labels, :]#我需要根据标签取出batchsize个原型中心用于计算平均原型中心
            loss_r = f.mse_loss(x, center_batch) / 2#这个是pl损失，主要是尽量让每个样本的特征分布与其对应的原型紧凑
            o_center = self.points.mean(0)#这是seen个原型的平均中心，接近原点
            dist_x_o=f.pairwise_distance(x,o_center,p=2)
            l_ = (self.points - o_center).pow(2).mean(1)#seen个原型到原型平均中心的距离
        # loss_outer = torch.exp(-l_.mean(0))
            loss_outer_std = torch.std(l_)#特征分布与原型中心距离的标准差（本文提出的损失）

            loss = loss_main + self.weight_pl * loss_r + loss_outer_std#损失包含四部分
            return logits, loss
        else:
            o_center = self.points.mean(0)#这个self.point应该是已经学习过的point吧
            dist_x_o = f.pairwise_distance(x, o_center, p=2)
            return dist_x_o
