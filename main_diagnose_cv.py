
import os
import csv
import time
import json
import datetime
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import os, sys; sys.path.append(os.path.dirname(__file__)+"./")
import torchutils as utils
from focalloss import FocalLoss
from engine import train_epoch, evaluate, sensitivity_specificity
from engine import SeqClassificationDataset, SleepSequenceCollator
from models.gpt_longseq import GPTLongSeqTransformer


sleep_datasets = {
    'cap_sleepedf' : {
        'data_dir' : 'e:/eegdata/sleep/cap_sleepedf/',
        'output_dir' : 'e:/eegdata/sleep/cap_sleepedf/output/',
    },
    'isruc' : {
        'data_dir' : '/home/yuty2009/data/eegdata/sleep/isruc/',
        'output_dir' : '/home/yuty2009/data/eegdata/sleep/isruc/output/',
    },
}

parser = argparse.ArgumentParser(description='Train and evaluate the Sleep Sequence Classification Model')
parser.add_argument('-D', '--dataset', default='cap_sleepedf', metavar='PATH',
                help='dataset used')
parser.add_argument('-a', '--arch', metavar='ARCH', default='gpt',
                help='model architecture (default: gpt)')
parser.add_argument('--seg_seqlen', default=60, type=int, metavar='N',
                    help='maximum acceptable sequence length (default: 30)')
parser.add_argument('--embed_dim', default=48, type=int, metavar='N',
                    help='embedded feature dimension (default: 192)')
parser.add_argument('--num_layers', default=3, type=int, metavar='N',
                    help='number of transformer layers (default: 6)')
parser.add_argument('--num_heads', default=6, type=int, metavar='N',
                    help='number of heads for multi-head attention (default: 6)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--folds', default=10, type=int, metavar='N',
                    help='number of folds cross-valiation (default: 20)')
parser.add_argument('--start-fold', default=0, type=int, metavar='N',
                    help='manual fold number (useful on restarts)')
parser.add_argument('--splits', default='', type=str, metavar='PATH',
                    help='path to cross-validation splits file (default: none)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', type=float, default=1e-8, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0')
parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                    help='epochs to warmup LR')
parser.add_argument('--schedule', default='cos', type=str,
                    choices=['cos', 'step'],
                    help='learning rate schedule (how to change lr)')
parser.add_argument('--lr_drop', default=[0.6, 0.8], nargs='*', type=float,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-s', '--save-freq', default=10, type=int,
                metavar='N', help='save frequency (default: 100)')
parser.add_argument('-e', '--evaluate', default=False, action='store_true',
                    help='evaluate on the test dataset')
    

def main():

    args = parser.parse_args()

    args.data_dir = sleep_datasets[args.dataset]['data_dir']
    args.output_dir = sleep_datasets[args.dataset]['output_dir']
    args.pretrained = 'output/gpt_shhs_pretrained/60_48_3_6.pth.tar'

    output_prefix = f"{args.arch}"
    output_prefix += "/session_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if not hasattr(args, 'output_dir'):
        args.output_dir = args.data_dir
    args.output_dir = os.path.join(args.output_dir, output_prefix)
    os.makedirs(args.output_dir)
    print("=> results will be saved to {}".format(args.output_dir))

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.manual_seed(1243)
    # np.random.seed(1243)

    # Data loading code
    print("=> loading dataset {} from '{}'".format(args.dataset, args.data_dir))
    ann_f = open(os.path.join(args.data_dir, 'annotations.txt'), newline='')
    reader = csv.reader(ann_f)
    seqdata = [row[0:] for row in reader]
    # seqdata = crop_sleep_period(seqdata)
    sub_f = open(os.path.join(args.data_dir, 'subject_labels.txt'), newline='')
    reader = csv.reader(sub_f)
    labeldata = np.array(list(reader)).flatten()
    labeldata = [int(lb) for lb in labeldata]
    labeldata = np.asarray(labeldata)
    if args.dataset == 'cap_sleepedf':
        labeldata[labeldata > 0] = 1 # 0: normal, 1: abnormal  # cap
    elif args.dataset == 'mnc':
        labeldata[labeldata != 1] = 0 # 0: normal + hypersomnia, 1: narcolepsy

    ## select data
    # target_labels = ['0', '1']
    # seqdata_selected = []
    # labeldata_selected = []
    # for i in range(len(labeldata)):
    #     if labeldata[i] in target_labels:
    #         seqdata_selected.append(seqdata[i])
    #         labeldata_selected.append(labeldata[i])
    # seqdata = seqdata_selected
    # labeldata = labeldata_selected
    
    print('Data for %d subjects has been loaded' % len(labeldata))
    num_subjects = len(labeldata)

    args.vocab_size = 6     # 5 sleep stages + padding token (5)
    
    args.writer = None
    with open(args.output_dir + "/args.json", 'w') as fid:
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        json.dump(args.__dict__, fid, indent=2, default=default)
    args.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, f"log"))

    if args.start_fold <= 0:
        kfold = KFold(n_splits=args.folds, shuffle=True, random_state=1243)
        splits_train, splits_test = [], []
        for (a, b) in kfold.split(np.arange(num_subjects)):
            splits_train.append(a)
            splits_test.append(b)
        np.savez(args.output_dir + '/splits.npz', splits_train=splits_train, splits_test=splits_test)
    else:
        splits = np.load(args.splits, allow_pickle=True)
        splits_train, splits_test = splits['splits_train'], splits['splits_test']

    # k-fold cross-validation
    train_accus, train_losses = np.zeros(args.folds), np.zeros(args.folds)
    test_accus,  test_losses  = np.zeros(args.folds), np.zeros(args.folds)
    test_cms, sensis, specis = np.zeros((args.folds, 2, 2)), np.zeros(args.folds), np.zeros(args.folds)
    test_ytrues, test_yprobs = [], []
    for fold in range(args.start_fold, args.folds):

        idx_train, idx_test = splits_train[fold], splits_test[fold]

        train_seqdata = [seqdata[i] for i in idx_train]
        train_labeldata = [labeldata[i] for i in idx_train]
        test_seqdata = [seqdata[i] for i in idx_test]
        test_labeldata = [labeldata[i] for i in idx_test]

        train_dataset = SeqClassificationDataset(train_seqdata, train_labeldata)
        test_dataset = SeqClassificationDataset(test_seqdata, test_labeldata)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=SleepSequenceCollator(args.seg_seqlen),
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=SleepSequenceCollator(args.seg_seqlen),
        )

        model = GPTLongSeqTransformer(
            num_classes = 2,
            vocab_size = args.vocab_size,
            seg_seqlen = args.seg_seqlen,
            embed_dim = args.embed_dim,
            num_layers = args.num_layers,
            num_heads = args.num_heads,
            pretrained = args.pretrained,
        )
        print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "M parameters")
        criterion = FocalLoss(alpha=2).to(args.device) # nn.CrossEntropyLoss().to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

        model = model.to(args.device)

        for epoch in range(args.epochs):
            start = time.time()
            utils.adjust_learning_rate(optimizer, epoch, args)
            lr = optimizer.param_groups[0]["lr"]

            train_losses[fold], train_accus[fold], _, _ = train_epoch(
                train_loader, model, criterion, optimizer, epoch, args)
            
            test_losses[fold], test_accus[fold], y_trues, y_probs = evaluate(
                test_loader, model, criterion, epoch, args)
            
            y_preds = np.argmax(y_probs, axis=-1)
            test_cms[fold] = confusion_matrix(y_true=y_trues, y_pred=y_preds)
            sensis[fold], specis[fold] = sensitivity_specificity(test_cms[fold])

            if args.output_dir and epoch > 0 and (epoch+1) % args.save_freq == 0:
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    }, epoch + 1,
                    is_best=False,
                    save_dir=os.path.join(args.output_dir, f"checkpoint/fold_{fold}"))

            if hasattr(args, 'writer') and args.writer:
                args.writer.add_scalar(f"Fold_{fold}/Accu/train", train_accus[fold], epoch)
                args.writer.add_scalar(f"Fold_{fold}/Accu/test",  test_accus[fold], epoch)
                args.writer.add_scalar(f"Fold_{fold}/Loss/train", train_losses[fold], epoch)
                args.writer.add_scalar(f"Fold_{fold}/Loss/test",  test_losses[fold], epoch)
                args.writer.add_scalar(f"Fold_{fold}/Misc/learning_rate", lr, epoch)

            print(f"Fold: {fold}, Epoch: {epoch}, "
                f"Train accu: {train_accus[fold]:.3f}, loss: {train_losses[fold]:.3f}, "
                f"Test accu: {test_accus[fold]:.3f}, loss: {test_losses[fold]:.3f}, "
                f"Sensitivity: {sensis[fold]:.3f}, Specificity: {specis[fold]:.3f}, "
                f"Epoch time = {time.time() - start:.3f}s")
            
        test_ytrues.append(y_trues)
        test_yprobs.append(y_probs)

    test_ytrues = np.concatenate(test_ytrues)
    test_yprobs = np.concatenate(test_yprobs)
    df_yy = pd.DataFrame({
        'y_true': test_ytrues,
        'y_prob': test_yprobs[:, 1],
    })
    df_yy.to_csv(os.path.join(args.output_dir, 'yy_' + model._get_name() + '.csv'))

    # Average over folds
    folds = [f"fold_{i}" for i in range(args.folds)] + ['average']
    train_accus = np.append(train_accus, np.mean(train_accus))
    train_losses = np.append(train_losses, np.mean(train_losses))
    test_accus  = np.append(test_accus, np.mean(test_accus))
    test_losses  = np.append(test_losses, np.mean(test_losses))
    sensis = np.append(sensis, np.mean(sensis))
    specis = np.append(specis, np.mean(specis))
    cm = np.sum(test_cms, axis=0)
    df_results = pd.DataFrame({
        'folds': folds,
        'train_accus': train_accus,
        'train_losses': train_losses,
        'test_accus' : test_accus,
        'test_losses' : test_losses,
        'test_sensis' : sensis,
        'test_specis' : specis,
    })
    df_results.to_csv(os.path.join(args.output_dir, 'results_' + model._get_name() + '.csv'))
    with open(os.path.join(args.output_dir, 'confusion_matrix.txt'), 'w') as f:
        csv.writer(f, delimiter=' ').writerows(cm)
        f.close()
    print(df_results)
    print(cm)


if __name__ == "__main__":

    main()
