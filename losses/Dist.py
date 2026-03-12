import torch
import torch.nn as nn


class Dist(nn.Module):      # to compute the distance, such as: d_e(x,y) or d_d(x,y)
    def __init__(self, num_classes=10, num_centers=1, feat_dim=2, init='random'):
        super(Dist, self).__init__()
        self.feat_dim = feat_dim#特征数：128
        self.num_classes = num_classes#分类类别：6
        self.num_centers = num_centers#中心的个数？到底是一个还是40个

        if init == 'random':
            self.centers = nn.Parameter(0.1*torch.randn(num_classes * num_centers, self.feat_dim))#生成一个6*128的张量（可以理解成6个128维度的点，代表6个原型的中心）；Parameter（）的作用是使得参数能够训练
            pass######
        else:
            self.centers = nn.Parameter(torch.Tensor(num_classes * num_centers, self.feat_dim))
            self.centers.data.fill_(0)

    def forward(self, features, center=None, metric='l2'):
        if metric == 'l2':#使用l2距离来度量原型和原型中心的距离
            f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)#对特征进行求和

            if center is None:
                c_2 = torch.sum(torch.pow(self.centers, 2), dim=1, keepdim=True)
                dist = f_2 - 2*torch.matmul(features, torch.transpose(self.centers, 1, 0)) + torch.transpose(c_2, 1, 0)
            else:               # default compute: d_e(features, P^k)
                c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)#将32896维度的centers降维成1（为什么呢）
                dist = f_2 - 2*torch.matmul(features, torch.transpose(center, 1, 0)) + torch.transpose(c_2, 1, 0)
            dist = dist / float(features.shape[1])#bach*seenclass，表示每一样本到所有原型中心的距离
        else:
            if center is None:
                center = self.centers 
            else:
                center = center 
            dist = features.matmul(center.t())
        dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])   # [batch_size, seen_class, num_centers=1]
        dist = torch.mean(dist, dim=2)                                          # [batch_size, seen_class]，这里按照第三个维度进行了求平均
        return dist#衡量了batch个样本距离seen个样本的距离
