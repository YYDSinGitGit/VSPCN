import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', '-d', metavar='DATA', default='cub',
                    help='dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='smp', )
parser.add_argument('--backbone', default='ViT-B/16', help='backbone')
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=5, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr1', default=1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr2', default=1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr', default=1e-4, type=float,
                    metavar='LR', help='joint fine tuning')
parser.add_argument('--depth_ref', default='[7, 8, 9, 10, 11, 12]', type=str,
                    metavar='depth_ref', help='depth in ref')
parser.add_argument('--epoch_decay', default=30, type=int,#20
                    metavar='LR', help='initial learning rate')
parser.add_argument('--epoch_decay_rate', default=0.125, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str,
                    help='checkpoint')
parser.add_argument('--pretrained', dest='pretrained', default=True, action='store_true',
                    help='use pre-trained model')
parser.add_argument('--al_lsemantic', action='store_false',
                    help='training with all semantic')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=6666, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--is_fix', dest='is_fix', default=True, action='store_true',
                    help='is_fix.')
parser.add_argument('--is_softmax', dest='is_softmax', default=True, action='store_true',
                    help='use softmax in slcp.')
parser.add_argument('--pin_memory', action='store_false',default=False, help='pin_memory.')

''' data proc '''
parser.add_argument('--flippingtest', dest='flippingtest', action='store_true',
                    help='flipping test.')


parser.add_argument('--att1', type=str, default='', help="bert or transformer")
parser.add_argument('--v2a', type=str, default='False', help="")
parser.add_argument('--ftp', type=str, default='True', help="")
''' transformer1 '''
parser.add_argument('--l', default=2, type=int,
                    help='num of layer')
parser.add_argument('--l2', default=2, type=int,
                    help='num of layer')
parser.add_argument('--h', default=10, type=int,
                    help='num of h')

''' transformer2 '''
parser.add_argument('--dc_layer', default=1, type=int,help='the number of decoderlayer')
parser.add_argument('--a2v', '-a2v', default=False,help='use attention in the second stage' )
parser.add_argument('--h2', default=1, type=int,
                    help='num of h2')

''' loss '''

parser.add_argument('--sigma', dest='sigma', default=0.6, type=float,
                    help='sigma.')
parser.add_argument('--loss', type=str, default='Slcp', help="loss for the first stage")
parser.add_argument('--metric', type=str, default='', help="metric for prototyle learning")
parser.add_argument('--lamda', type=float, default=0.5, help="weight for center loss")
parser.add_argument('--w_aux', type=float, default=0, help="weight for aux loss")
parser.add_argument('--w_zsl', type=float, default=1, help="weight for aux loss")
parser.add_argument('--gama', type=float, default=0, help="weight for center loss")
parser.add_argument('--feat_dim', type=int, default=40,#sun为150
                    help="classifier32:128,resnet50:2048,RAN:1024,resnest:2048")
parser.add_argument('--center', type=str, default='mean', help="The mean center of the prototype")
parser.add_argument('--sim', type=str, default='', help="The mean center of the prototype")
parser.add_argument('--norm', type=str, default='', help="")


'''arc'''
parser.add_argument('--scale', type=float, default=64.0, help="weight for arc_margin loss")
parser.add_argument('--margin', type=float, default=0.50, help="weight for arc_margin loss")


parser.add_argument('--tf_dim', type=int, default=300, help="dim for transformer")


'''optimizer'''
parser.add_argument('--one_opt', default=True, help='use one optimizer.')

'''Ablation experiment'''
parser.add_argument('--wo', type=str, default='', help="")


'''test'''

parser.add_argument('--strategy', default=1, type=float,
                    help='1:openset')

parser.add_argument('--base', default=0.3, type=float,
                    help='base for threshold')
parser.add_argument('--steps', default=71, type=int,
                    help='steps for threshold')
parser.add_argument('--stride', default=0.01, type=float,
                    help='base for threshold')

parser.add_argument('--test', action='store_false', default=False,  help='pin_memory.')

'''visualization'''
parser.add_argument('--feature', default='',  type=str,help='pin_memory.')
parser.add_argument('--visualize', default='True',  type=str,help='pin_memory.')


'''numbers of prompt'''
parser.add_argument('--np', type=int, default=1, help="dim for transformer")
parser.add_argument('--ctx_init', type=str, default='', help="")

parser.add_argument('--csc', action='store_true',  help='pin_memory.')
parser.add_argument('--position', default='end',  type=str,help='pin_memory.')
parser.add_argument('--prompt', type=str, default='', help="")
parser.add_argument('--att_name', type=str, default='clip', help="")
parser.add_argument('--att', action='store_true',  help='pin_memory.')
parser.add_argument('--att_semantic', action='store_true',  help='pin_memory.')
parser.add_argument('--fill_zeros', action='store_true',  help='pin_memory.')
parser.add_argument('--instance', action='store_true',  help='pin_memory.')
parser.add_argument('--avg1', action='store_true',  help='pin_memory.')
parser.add_argument('--avg2', action='store_true',  help='pin_memory.')
parser.add_argument('--rand_init', action='store_true',  help='pin_memory.')
parser.add_argument('--gap', type=float, default=0.5, help='gap')
parser.add_argument('--ref', type=int, default=8, help='depth for channel interaction')
parser.add_argument('--channel', action='store_false', help='use channel')
parser.add_argument('--spacial', action='store_true', help='use spacial')

parser.add_argument('--attention', action='store_true', help='use attention')

# loss
parser.add_argument('--alpha', default='[0.5, 0.8, 0.9]', type=str,#0.001
                    metavar='alpha', help='alpha')
parser.add_argument('--alpha0', default='[0.02]', type=str,#0.001
                    metavar='alpha0', help='alpha0')
parser.add_argument('--alpha1', default='[0.05, 0.5]', type=str,#0.001
                    metavar='alpha1', help='alpha1')
parser.add_argument('--alpha2', default='[0.2]', type=str,#0.001
                    metavar='alpha2', help='alpha2')
parser.add_argument('--va', default=0.05, type=float,
                    metavar='va',     help='va')
parser.add_argument('--la', default=0.8, type=float,
                    metavar='la',     help='la')


args = parser.parse_args()
args.save_path = './' + 'output/' + args.data
args.resume=''
if args.checkpoint!='':
    args.resume='/xxx/output/'+args.data+'/'+args.checkpoint
print(args)