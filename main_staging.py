
import mne
import torch
import argparse
import numpy as np

import os, sys; sys.path.append(os.path.dirname(__file__))
import torchutils as utils
from eegreader import ToTensor, SeqEEGDataset
from models.sleepnet import TinySleepNet
from models.seqsleepnet import SeqSleepNet
from models.gpt_transformers import GPTLM
from yasa.others import sliding_window
from yasa.staging_epoched import SleepStaging


parser = argparse.ArgumentParser(description='Evaluate the Sleep Model')
parser.add_argument('-a', '--arch', metavar='ARCH', default='tinysleepnet',
                help='model architecture: tinysleepnet or yasa (default: tinysleepnet)')
args = parser.parse_args()

# Model parameters and paths, do not change them
args.n_seqlen = 15
args.seg_seqlen = 90
args.embed_dim = 48
args.num_layers = 3
args.num_heads = 6
args.pretrained = 'output/seq_tinysleepnet_pretrained/best.pth.tar'
args.sm_pretrained = 'output/gpt_shhs_pretrained/90_48_3_6.pth.tar'
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# edf file path
edf_path = 'e:/eegdata/sleep/sleepedf153/sleep-cassette/SC4001E0-PSG.edf'

# load the edf file
signal = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
print('The channels are:', signal.ch_names)
print('The sampling frequency is:', signal.info['sfreq'])

sample_f = 100
channels = signal.info['ch_names']
# signal = signal.set_eeg_reference(ref_channels=["A1"])
# signal_notched = signal.notch_filter(freqs=50, notch_widths=2)
# signal_processed = signal_notched.filter(l_freq=0.3, h_freq=35)
signal_processed = signal.resample(sfreq=sample_f)

# eog_name, emg_name = None, None
# for ch_name in channels:
#     if ch_name.startswith('EOG'):
#         eog_name = ch_name
#         break
# for ch_name in channels:
#     if ch_name.startswith('EMG'):
#         emg_name = ch_name
#         break

eeg_name = 'EEG Fpz-Cz' # 修改成数据中的EEG通道名，通常是C4-M1或者C3-M2
signal_eeg = signal_processed.get_data(picks=eeg_name, units=dict(eeg="uV", emg="uV", eog="uV", ecg="uV"))

freq_broad = (0.4, 30)
dt_filt = mne.filter.filter_data(
    signal_eeg, sample_f, l_freq=freq_broad[0], h_freq=freq_broad[1], verbose=False)
times, data_epochs = sliding_window(dt_filt, sf=sample_f, window=30)

if args.arch == 'yasa':
    sls = SleepStaging(data_epochs, eeg_name=eeg_name, eog_name=None, emg_name=None)
    yprob = sls.predict_proba()
    cols = ['W', 'N1', 'N2', 'N3', 'R'] # ['N1', 'N2', 'N3', 'R', 'W'] => ['W', 'N1', 'N2', 'N3', 'R']
    yprob = yprob[cols].to_numpy()
    ypred = np.argmax(yprob, axis=-1)
    yprob = torch.FloatTensor(yprob).to(args.device)
elif args.arch == 'tinysleepnet':
    # create model
    base_encoder = TinySleepNet(0, 3000)
    model = SeqSleepNet(base_encoder, 5, n_seqlen=args.n_seqlen)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "M parameters")
    utils.load_model(model, args.pretrained, True)
    model = model.to(args.device)
    model.eval()

    # epoch_seqs = []
    # for idx in range(len(data_epochs)):
    #     epoch_seq = np.zeros((args.n_seqlen,)+data_epochs.shape[1:])
    #     idx1 = idx + 1
    #     if idx1 < args.n_seqlen:
    #         epoch_seq[-idx1:] = data_epochs[:idx1]
    #     else:
    #         epoch_seq = data_epochs[idx1-args.n_seqlen:idx1]
    #     epoch_seqs.append(epoch_seq)

    # epoch_seqs = np.array(epoch_seqs)
    # epoch_seqs = torch.FloatTensor(epoch_seqs).to(args.device)
    # epoch_seqs = epoch_seqs.permute(0, 1, 3, 2) # (n_epochs, n_seqlen, n_wavlen, n_channels)
    # epoch_seqs = epoch_seqs.unsqueeze(2) # (n_epochs, n_seqlen, 1, n_wavlen, n_channels)
    data_epochs = data_epochs.transpose(0, 2, 1) # (n_epochs, n_wavlen, n_channels)
    labels_epochs = np.zeros(data_epochs.shape[0])
    dataset_sub = SeqEEGDataset(data_epochs, labels_epochs, args.n_seqlen, ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset_sub, batch_size=64)
    yprob = []
    for data, target in test_loader:
        data = data.to(args.device)
        # compute output
        output = model(data)
        yprob.append(output.detach().clone())
    yprob = torch.concatenate(yprob)
    ypred = torch.argmax(yprob, dim=-1)
    ypred = ypred.cpu().numpy()

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
ypred_corrected = sleep_model.correct(yprob, ngram=ngram, lm_weight=alpha).flatten()
ypred_corrected = ypred_corrected.cpu().numpy()

f_out = edf_path.replace('.edf', '_sleep_stage.csv')
f_out_corrected = edf_path.replace('.edf', '_sleep_stage_corrected.csv')
np.savetxt(f_out, ypred, fmt='%d', delimiter='\n')
np.savetxt(f_out_corrected, ypred_corrected, fmt='%d', delimiter='\n')
