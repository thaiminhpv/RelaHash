import argparse
import logging
import os

import torch

import configs
import train

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(asctime)s: %(message)s',
                    datefmt='%d-%m-%y %H:%M:%S')

torch.backends.cudnn.benchmark = True
configs.default_workers = os.cpu_count()

parser = argparse.ArgumentParser(description='RelaHash')
parser.add_argument('--nbit', default=64, type=int, help='number of bits')
parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='training epochs')
parser.add_argument('--ds', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet100', 'nuswide'],
                    help='dataset')
# loss related
parser.add_argument('--beta', default=8, type=float, help='beta param')
parser.add_argument('--margin', default=0.5, type=float, help='softmax loss margin')
parser.add_argument('--tag', default='test')

# codebook generation
parser.add_argument('--init-centroids-method', default='M', choices=['N', 'U', 'B', 'M', 'H'], help='N = sign of gaussian; '
                                                                                    'B = bernoulli; '
                                                                                    'M = MaxHD'
                                                                                    'H = Hadamard matrix')
parser.add_argument('--wandb', action='store_true', default=False, help='enable wandb logging')

parser.add_argument('--seed', default=420, help='seed number; default: 420')

parser.add_argument('--device', default='cuda:0')

args = parser.parse_args()

config = {
    'arch': 'RelaHash',
    'arch_kwargs': {
        'nbit': args.nbit,
        'nclass': 0,  # will be updated below
        'batchsize': args.bs,
        'init_method': args.init_centroids_method,
        'pretrained': True,
        'freeze_weight': False,
        'device': args.device,
    },
    'batch_size': args.bs,
    'dataset': args.ds,
    'multiclass': args.ds == 'nuswide',
    'dataset_kwargs': {
        'resize': 256 if args.ds in ['nuswide'] else 224,
        'crop': 224,
        'norm': 2,
        'evaluation_protocol': 1,  # only affect cifar10
        'reset': False,
        'separate_multiclass': False,
    },
    'optim': 'adam',
    'optim_kwargs': {
        'lr': args.lr,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'nesterov': False,
        'betas': (0.9, 0.999)
    },
    'epochs': args.epochs,
    'scheduler': 'step',
    'scheduler_kwargs': {
        'step_size': int(args.epochs * 0.8),
        'gamma': 0.1,
        'milestones': '0.5,0.75'
    },
    'save_interval': 0,
    'eval_interval': 10,
    'tag': args.tag,
    'seed': args.seed,

    # loss_param
    'beta': args.beta,
    'm': args.margin,
    'wandb_enable': args.wandb,
    'device': args.device
}

config['arch_kwargs']['nclass'] = configs.nclass(config)
config['R'] = configs.R(config)

logdir = (f'logs/{config["arch"]}{config["arch_kwargs"]["nbit"]}_'
          f'{config["dataset"]}_{config["dataset_kwargs"]["evaluation_protocol"]}_'
          f'{config["epochs"]}_'
          f'{config["optim_kwargs"]["lr"]}_'
          f'{config["optim"]}_')

if config['tag'] != '':
    logdir += f'/{config["tag"]}_{config["seed"]}_'
else:
    logdir += f'/{config["seed"]}_'

# make sure no overwrite problem
count = 0
orig_logdir = logdir
logdir = orig_logdir + f'{count:03d}'
tag = args.tag

while os.path.isdir(logdir):
    count += 1
    logdir = orig_logdir + f'{count:03d}'

config['logdir'] = logdir

count = 0
orig_logdir = logdir
logdir = orig_logdir + f'{count:03d}'

train.main(config)
