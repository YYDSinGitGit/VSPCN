from glo import *
import os
from utils import *
import datasets
import h5py
import json
import pickle
import clip
import PIL

if args.data.lower() == 'cub':
    img_path = '/xxx/data/cub/CUB_200_2011/images/'
elif args.data.lower() == 'awa2':
    img_path = '/xxx/data/awa2/Animals_with_Attributes2/JPEGImages/'
elif args.data.lower() == 'sun':
    img_path = 'xxx/data/sun/images/'
elif args.data.lower() == 'apy':
    img_path = 'xxxx'

traindir = os.path.join('/home/xxx/data/', args.data, 'train.list')
valdir1 = os.path.join('/home/xxx/data/', args.data, 'test_seen.list')
valdir2 = os.path.join('/home/xxx/data/', args.data, 'test_unseen.list')

_, transforms = clip.load('ViT-B/32', 'cuda:{}'.format(args.gpu))

train_dataset = datasets.ImageFolder(img_path, traindir, transforms)


args.distributed = args.world_size > 1

if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
else:
    train_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

val_loader1 = torch.utils.data.DataLoader(
    datasets.ImageFolder(img_path, valdir1, transforms),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

val_loader2 = torch.utils.data.DataLoader(
    datasets.ImageFolder(img_path, valdir2, transforms),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)


''' data load info '''
data_info = h5py.File(os.path.join('/home/xxx/data', args.data, 'data_info.h5'), 'r')
nc = data_info['all_att'][...].shape[0]
args.natt=data_info['all_att'][...].shape[1]
att_size = data_info['all_att'][...].shape[1]
seen_c = data_info['seen_class'][...]
unseen_c = data_info['unseen_class'][...]
semantic_data = {'seen_class': data_info['seen_class'][...],
                 'unseen_class': data_info['unseen_class'][...],
                 'all_class': np.arange(nc),
                 'all_att': data_info['all_att'][...]}
''' load semantic data'''
args.num_classes = nc
args.num_seen = data_info['seen_class'][...].shape[0]
args.num_unseen = data_info['unseen_class'][...].shape[0]
args.seen_c = seen_c
args.unseen_c = unseen_c


with open('/home/xxx/data/{}/name_int_sort_label.json'.format(args.data), 'r') as f:
    args.classname = list(json.load(f).keys())
if args.data=='cub':
    with open('w2v/CUB_attribute.pkl',"rb") as f:
        w2v=pickle.load(f)
elif args.data=='sun':
    with open('w2v/SUN_attribute.pkl',"rb") as f:
        w2v=pickle.load(f)
elif args.data=='awa2':
    with open('w2v/AWA2_attribute.pkl',"rb") as f:
        w2v=pickle.load(f)

args.w2v = w2v
args.att_size = att_size
args.att = semantic_data['all_att']
args.seen_att = args.att[seen_c, :]
args.unseen_att = args.att[unseen_c, :]
# adj
adj = adj_matrix(nc)
args.adj = adj


with open('/home/xxx/data/{}/att_name.txt'.format(args.data), 'r') as f:
    data = f.readlines()

with open('/home/xxx/data/{}/{}_attribute.pkl'.format(args.att_name,args.data), 'rb') as f:
    args.att_name_emb = pickle.load(f)