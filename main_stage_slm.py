
import tqdm
import torch
import datetime
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

import os, sys; sys.path.append(os.path.dirname(__file__)+"/../")
import torchutils as utils
from eegreader import ToTensor, EEGDataset, SeqEEGDataset
import datasets.sleepedfreader as sedfreader
import datasets.massreader as massreader
from models.sleepnet import TinySleepNet
from models.seqsleepnet import SeqSleepNet
from models.gpt_transformers import GPTLM


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

parser = argparse.ArgumentParser(description='Evaluate the Sleep Model')
parser.add_argument('-D', '--dataset', default='sleepedf', metavar='PATH',
                    help='dataset used')
parser.add_argument('-a', '--arch', metavar='ARCH', default='tinysleepnet',
                help='model architecture (default: tinysleepnet)')
parser.add_argument('--seq', default=True, action='store_true',
                    help='use sequential model (default: True)')
args = parser.parse_args()

args.data_dir = sleep_datasets[args.dataset]['data_dir']
args.output_dir = sleep_datasets[args.dataset]['output_dir']

output_prefix = f"slm_seq_{args.arch}"
output_prefix += "/session_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
if not hasattr(args, 'output_dir'):
    args.output_dir = args.data_dir
args.output_dir = os.path.join(args.output_dir, output_prefix)
os.makedirs(args.output_dir)
print("=> results will be saved to {}".format(args.output_dir))

args.sm_pretrained = 'output/gpt_shhs_pretrained/90_48_3_6.pth.tar'
if not args.seq:
    args.pretrained = 'output/tinysleepnet_pretrained/best.pth.tar'
else:
    args.pretrained = 'output/seq_tinysleepnet_pretrained/best.pth.tar'

args.seg_seqlen = 90
args.embed_dim = 48
args.num_layers = 3
args.num_heads = 6

# Data loading code
print("=> loading dataset {} from '{}'".format(args.dataset, args.data_dir))
if args.dataset == 'sleepedf':
    data, labels, subjects = sedfreader.load_dataset_preprocessed(args.data_dir+'processed/')
elif args.dataset == 'mass':
    data, labels, subjects = massreader.load_dataset_preprocessed(args.data_dir+'processed/')
else:
    raise NotImplementedError

print('Data for %d subjects has been loaded' % len(data))
num_subjects = len(data)
args.num_classes = 5
args.n_wavlen = 3000
args.n_seqlen = 15
args.topk = (1, 5)
args.epochs = 0

tf_epoch = ToTensor()

idx_full = np.arange(num_subjects)
if not args.seq:
    base_classes = 5
    all_datasets = [EEGDataset(data[i], labels[i], tf_epoch) for i in idx_full]
else:
    base_classes = 0
    all_datasets = [SeqEEGDataset(data[i], labels[i], args.n_seqlen, tf_epoch) for i in idx_full]

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create model
args.view = 'st'
args.patch_size = 20
base_encoder = TinySleepNet(base_classes, args.n_wavlen)

args.model_sma = None
if not args.seq:
    model = base_encoder
else:
    model = SeqSleepNet(base_encoder, args.num_classes, n_seqlen = args.n_seqlen)
print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "M parameters")
utils.load_model(model, args.pretrained, True)
model = model.to(args.device)
model.eval()

args.vocab_size = 6 # 5 sleep stages + padding token (5)
sleep_model = GPTLM(
    vocab_size = args.vocab_size,
    max_seqlen = args.seg_seqlen,
    embed_dim = args.embed_dim,
    num_layers = args.num_layers, 
    num_heads = args.num_heads,
)
utils.load_checkpoint(args.sm_pretrained, sleep_model, strict=True)
sleep_model = sleep_model.to(args.device)
sleep_model.eval()

alpha = 0.1
ngram = 30
ytrues, ypreds, ycorrecteds = [], [], []
for sub in tqdm.tqdm(idx_full):
    test_loader = torch.utils.data.DataLoader(all_datasets[sub], batch_size=256)
    
    yprob_sub, ytrue_sub = [], []
    for data, target in test_loader:
        data = data.to(args.device)
        target = target.to(args.device)
        # compute output
        output = model(data)
        yprob_sub.append(output.detach().clone())
        ytrue_sub.append(target.detach().clone())
    yprob_sub = torch.concatenate(yprob_sub)
    ytrue_sub = torch.concatenate(ytrue_sub)
    ypred_sub = torch.argmax(yprob_sub, dim=-1)

    ytrues += ytrue_sub.cpu().numpy().tolist()
    ypreds += ypred_sub.cpu().numpy().tolist()

    labels_corrected_sub = sleep_model.correct(yprob_sub, ngram=ngram, lm_weight=alpha).flatten()
    ycorrecteds += labels_corrected_sub.cpu().numpy().tolist()

accu_raw = accuracy_score(ytrues, ypreds)
mf1_raw = f1_score(ytrues, ypreds, average='macro')
kappa_raw = cohen_kappa_score(ytrues, ypreds)
accu_sm = accuracy_score(ytrues, ycorrecteds)
mf1_sm = f1_score(ytrues, ycorrecteds, average='macro')
kappa_sm = cohen_kappa_score(ytrues, ycorrecteds)

savepath = os.path.join(args.output_dir, 'result.txt')
out_f = open(savepath, 'w')

out_str = f"accu_raw: {accu_raw}, accu_sm: {accu_sm}, " \
        f"mf1_raw: {mf1_raw}, mf1_sm: {mf1_sm}, " \
        f"kappa_raw: {kappa_raw}, kappa_sm: {kappa_sm}"
out_f.write(out_str + "\n")
print(out_str)
