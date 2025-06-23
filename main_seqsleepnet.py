
import os
import time
import json
import random
import datetime
import warnings
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import sys; sys.path.append(os.path.dirname(__file__))
import distributed as dist
import torchutils as utils
from eegreader import ToTensor, SeqEEGDataset
from engine import train_epoch, evaluate
from datasets import sleepedfreader
from datasets import massreader
from models.sleepnet import TinySleepNet
from models.seqsleepnet import SeqSleepNet


sleep_datasets = {
    'sleepedf' : {
        'data_dir' : 'e:/eegdata/sleep/sleepedf153/sleep-cassette/',
        'output_dir' : 'e:/eegdata/sleep/sleepedf153/sleep-cassette/output/',
    },
    'mass' : {
        'data_dir' : '/home/yuty2009/data/eegdata/sleep/mass/',
        'output_dir' : '/home/yuty2009/data/eegdata/sleep/mass/output/',
    },
}

parser = argparse.ArgumentParser(description='Training from Scratch')
parser.add_argument('-D', '--dataset', default='sleepedf', metavar='PATH',
                    help='dataset used')
parser.add_argument('-a', '--arch', metavar='ARCH', default='tinysleepnet',
                    help='model architecture (default: tinysleepnet)')
parser.add_argument('-v', '--view', metavar='VIEW', default='st',
                    help='which views used (default: st)')
parser.add_argument('--pretrained', 
                    default='',
                    metavar='PATH', help='path to pretrained model (default: none)')
parser.add_argument('--use_sma', action='store_true')
parser.add_argument('--early_stop', action='store_true')
parser.add_argument('--freeze_encoder', action='store_true')
parser.add_argument('-p', '--patch-size', default=20, type=int, metavar='N',
                    help='patch size (default: 20) when dividing the long signal into windows')
parser.add_argument('--embed_dim', default=192, type=int, metavar='N',
                    help='embedded feature dimension (default: 192)')
parser.add_argument('--num_layers', default=3, type=int, metavar='N',
                    help='number of transformer layers (default: 6)')
parser.add_argument('--num_heads', default=6, type=int, metavar='N',
                    help='number of heads for multi-head attention (default: 6)')
parser.add_argument('--global_pool', action='store_true', default=True)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--folds', default=10, type=int, metavar='N',
                    help='number of folds cross-valiation (default: 20)')
parser.add_argument('--start-fold', default=0, type=int, metavar='N',
                    help='manual fold number (useful on restarts)')
parser.add_argument('--splits', default='', type=str, metavar='PATH',
                    help='path to cross-validation splits file (default: none)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optimizer', default='adamw', type=str,
                    choices=['adam', 'adamw', 'sgd', 'lars'],
                    help='optimizer used to learn the model')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', type=float, default=1e-8, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0')
parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                    help='epochs to warmup LR')
parser.add_argument('--schedule', default='cos', type=str,
                    choices=['cos', 'step'],
                    help='learning rate schedule (how to change lr)')
parser.add_argument('--lr_drop', default=[0.6, 0.8], nargs='*', type=float,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-s', '--save-freq', default=50, type=int,
                    metavar='N', help='save frequency (default: 100)')
parser.add_argument('-e', '--evaluate', action='store_true',
                    help='evaluate on the test dataset')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='local rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--mp', '--mp-dist', action='store_true', default=False,
                    help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training',
                    dest='mp_dist')
parser.add_argument('--use_amp', action='store_true', default=False,
                    help='Use mixed precision training')


def main(gpu, args):
    args.gpu = gpu
    args = dist.init_distributed_process(args)

    # enable flash attention
    torch.backends.cuda.sdp_kernel(
        enable_flash=True, 
        enable_math=False, 
        enable_mem_efficient=False
    )

    if args.seed is not None:
        if args.gpu is not None:
            args.seed += args.gpu
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Data loading code
    print("=> loading dataset {} from '{}'".format(args.dataset, args.data_dir))
    if args.dataset == 'sleepedf':
        datalist, labellist, subjects = sleepedfreader.load_dataset_preprocessed(args.data_dir+'processed/')
    elif args.dataset == 'mass':
        datalist, labellist, subjects = massreader.load_dataset_preprocessed(args.data_dir+'processed/')
    else:
        raise NotImplementedError
    
    print('Data for %d subjects has been loaded' % len(datalist))
    num_subjects = len(datalist)
    args.n_wavlen = 3000
    args.num_classes = 5
    args.n_seqlen = 15

    tf_epoch = ToTensor()

    args.writer = None
    if not args.distributed or args.rank == 0:
        with open(args.output_dir + "/args.json", 'w') as fid:
            default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
            json.dump(args.__dict__, fid, indent=2, default=default)
        args.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, f"log"))

    if len(args.splits) == 0 or not os.path.exists(args.splits):
        kfold = KFold(n_splits=args.folds, shuffle=True)
        splits_train, splits_test = [], []
        for (a, b) in kfold.split(np.arange(num_subjects)):
            splits_train.append(a)
            splits_test.append(b)
        np.savez(args.output_dir + '/splits.npz', splits_train=splits_train, splits_test=splits_test)
    else:
        splits = np.load(args.splits, allow_pickle=True)
        print(f"Loaded splits from {args.splits}")
        splits_train, splits_test = splits['splits_train'], splits['splits_test']
        np.savez(args.output_dir + '/splits.npz', splits_train=splits_train, splits_test=splits_test)

    # k-fold cross-validation
    train_accus, train_losses = np.zeros(args.folds), np.zeros(args.folds)
    valid_accus, valid_losses = np.zeros(args.folds), np.zeros(args.folds)
    test_accus,  test_losses  = np.zeros(args.folds), np.zeros(args.folds)
    for fold in range(args.start_fold, args.folds):

        idx_train, idx_test = splits_train[fold], splits_test[fold]

        # num_train = int(0.9*len(idx_train))
        # idx_train, idx_valid = idx_train[:num_train], idx_train[num_train:]
        idx_train, idx_valid = train_test_split(idx_train, test_size=0.1, random_state=1243)

        trainsets = [SeqEEGDataset(datalist[i], labellist[i], args.n_seqlen, tf_epoch) for i in idx_train]
        validsets = [SeqEEGDataset(datalist[i], labellist[i], args.n_seqlen, tf_epoch) for i in idx_valid]
        testsets  = [SeqEEGDataset(datalist[i], labellist[i], args.n_seqlen, tf_epoch) for i in idx_test]

        train_dataset = torch.utils.data.ConcatDataset(trainsets)
        valid_dataset = torch.utils.data.ConcatDataset(validsets)
        test_dataset = torch.utils.data.ConcatDataset(testsets)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False,
        )

        # create model
        print("=> creating sleep model")
        if args.arch in ['tinysleepnet', 'TinySleepNet']:
            base_encoder = TinySleepNet(0, args.n_wavlen)
        else:
            raise NotImplementedError
        
        args.model_sma = None
        model = SeqSleepNet(base_encoder, args.num_classes, n_seqlen = args.n_seqlen)
        print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "M parameters")

        model = dist.convert_model(args, model)

        criterion = nn.CrossEntropyLoss().to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

        best_accu, best_loss = 0., 0.
        best_modelpath = os.path.join(args.output_dir, f"checkpoint/fold_{fold}/best.pth.tar")
        for epoch in range(args.epochs):
            start = time.time()
            utils.adjust_learning_rate(optimizer, epoch, args)
            lr = optimizer.param_groups[0]["lr"]

            train_losses[fold], train_accus[fold] = utils.train_epoch(
                train_loader, model, criterion, optimizer, epoch, args)
            
            valid_losses[fold], valid_accus[fold] = utils.evaluate(
                valid_loader, model, criterion, epoch, args)

            if hasattr(args, 'writer') and args.writer:
                args.writer.add_scalar(f"Fold_{fold}/Accu/train", train_accus[fold], epoch)
                args.writer.add_scalar(f"Fold_{fold}/Accu/valid", valid_accus[fold], epoch)
                args.writer.add_scalar(f"Fold_{fold}/Loss/train", train_losses[fold], epoch)
                args.writer.add_scalar(f"Fold_{fold}/Loss/valid", valid_losses[fold], epoch)
                args.writer.add_scalar(f"Fold_{fold}/Misc/learning_rate", lr, epoch)

            print(
                f"Fold: {fold}, Epoch: {epoch}, "
                f"Train accu: {train_accus[fold]:.3f}, loss: {train_losses[fold]:.3f}, "
                f"Valid accu: {valid_accus[fold]:.3f}, loss: {valid_losses[fold]:.3f}, "
                f"Epoch time = {time.time() - start:.3f}s"
            )
            
            # if args.output_dir and epoch > 0 and (epoch+1) % args.save_freq == 0:
            if epoch > 0 and valid_accus[fold] > best_accu:
                best_accu = valid_accus[fold]
                utils.save_model(model, best_modelpath)

        utils.load_model(model, best_modelpath, strict=True)
        test_losses[fold], test_accus[fold] = utils.evaluate(
            test_loader, model, criterion, epoch, args)
        print(f"Fold: {fold}, Epoch: {epoch}, "
            f"Test accu: {test_accus[fold]:.3f}, loss: {test_losses[fold]:.3f}")

        # Save intermediate results
        folds = [f"fold_{i}" for i in range(args.folds)]
        df_results = pd.DataFrame({
            'folds': folds,
            'train_accus': train_accus,
            'train_losses': train_losses,
            'valid_accus': valid_accus,
            'valid_losses': valid_losses,
            'test_accus' : test_accus,
            'test_losses' : test_losses,
        })
        df_results.to_csv(os.path.join(args.output_dir, 'results_' + model._get_name() + '.csv'))

    # Average over folds
    folds = [f"fold_{i}" for i in range(args.folds)] + ['average']
    train_accus = np.append(train_accus, np.mean(train_accus))
    train_losses = np.append(train_losses, np.mean(train_losses))
    valid_accus = np.append(valid_accus, np.mean(valid_accus))
    valid_losses = np.append(valid_losses, np.mean(valid_losses))
    test_accus  = np.append(test_accus, np.mean(test_accus))
    test_losses  = np.append(test_losses, np.mean(test_losses))
    df_results = pd.DataFrame({
        'folds': folds,
        'train_accus': train_accus,
        'train_losses': train_losses,
        'valid_accus': valid_accus,
        'valid_losses': valid_losses,
        'test_accus' : test_accus,
        'test_losses' : test_losses,
    })
    df_results.to_csv(os.path.join(args.output_dir, 'results_' + model._get_name() + '.csv'))


if __name__ == '__main__':

    args = parser.parse_args()

    args.data_dir = sleep_datasets[args.dataset]['data_dir']
    args.output_dir = sleep_datasets[args.dataset]['output_dir']

    output_prefix = f"seq_{args.arch}"
    output_prefix += "/session_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if not hasattr(args, 'output_dir'):
        args.output_dir = args.data_dir
    args.output_dir = os.path.join(args.output_dir, output_prefix)
    os.makedirs(args.output_dir)
    print("=> results will be saved to {}".format(args.output_dir))

    args = dist.init_distributed_mode(args)
    if args.mp_dist:
        if args.world_size > args.ngpus:
            print(f"Training with {args.world_size // args.ngpus} nodes, "
                  f"waiting until all nodes join before starting training")
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main, args=(args,), nprocs=args.ngpus, join=True)
    else:
        main(args.gpu, args)
