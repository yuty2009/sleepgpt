
import os
import math
import copy
import tqdm
import torch
import numpy as np
from copy import deepcopy
import common.distributed as dist
   

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, model_path, patience=7, stop_epoch=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.model_path = model_path
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def __call__(self, epoch, val_loss, best_acc, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, best_acc, model)
        elif score < self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, best_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, best_acc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased ({} --> {}).  Saving model ...{}'.format(self.val_loss_min, val_loss, self.model_path))
        obj = {
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
        }
        torch.save(obj, self.model_path)
        self.val_loss_min = val_loss


class RememberBest(object):
    """
    Class to remember and restore 
    the parameters of the model and the parameters of the
    optimizer at the epoch with the best performance.

    Parameters
    ----------
    column_name: str
        The best value in this column should indicate the epoch with the
        best performance (e.g. misclass might make sense).
    order: {1, -1} 
        1 means descend order, that is lower best_value is better, such as misclass.
        -1 means ascend order, that is larger best_value is better, such as accuracy.
        
    Attributes
    ----------
    best_epoch: int
        Index of best epoch
    """

    def __init__(self, column_name, order=1):
        self.column_name = column_name
        self.best_epoch = 0
        if order not in (1, -1):
            assert 'order should be either 1 or -1'
        self.order = order
        self.best_value = order * float("inf")
        self.model_state_dict = None
        self.optimizer_state_dict = None

    def remember_epoch(self, epochs_df, model, optimizer):
        """
        Remember this epoch: Remember parameter values in case this epoch
        has the best performance so far.
        
        Parameters
        ----------
        epochs_df: `pandas.Dataframe`
            Dataframe containing the column `column_name` with which performance
            is evaluated.
        model: `torch.nn.Module`
        optimizer: `torch.optim.Optimizer`

        """
        i_epoch = len(epochs_df) - 1
        current_val = float(epochs_df[self.column_name].iloc[-1])
        if self.order > 0 and current_val <= self.best_value:
            self.best_epoch = i_epoch
            self.best_value = current_val
            self.model_state_dict = deepcopy(model.state_dict())
            self.optimizer_state_dict = deepcopy(optimizer.state_dict())
        elif self.order < 0 and current_val >= self.best_value:
            self.best_epoch = i_epoch
            self.best_value = current_val
            self.model_state_dict = deepcopy(model.state_dict())
            self.optimizer_state_dict = deepcopy(optimizer.state_dict())

    def reset_to_best_model(self, epochs_df, model, optimizer):
        """
        Reset parameters to parameters at best epoch and remove rows 
        after best epoch from epochs dataframe.
        
        Modifies parameters of model and optimizer, changes epochs_df in-place.
        
        Parameters
        ----------
        epochs_df: `pandas.Dataframe`
        model: `torch.nn.Module`
        optimizer: `torch.optim.Optimizer`

        """
        # Remove epochs past the best one from epochs dataframe
        epochs_df.drop(range(self.best_epoch + 1, len(epochs_df)), inplace=True)
        model.load_state_dict(self.model_state_dict)
        optimizer.load_state_dict(self.optimizer_state_dict)


class MovingAvg(torch.nn.Module):
    # https://github.com/salesforce/ensemble-of-averages/blob/main/domainbed/algorithms.py
    def __init__(self, network, start_iter=100):
        super().__init__()
        self.network = network
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()
        self.sma_start_iter = start_iter
        self.global_iter = 0
        self.sma_count = 0
    def update_sma(self):
        self.global_iter += 1
        new_dict = {}
        if self.global_iter>=self.sma_start_iter:
            self.sma_count += 1
            for (name,param_q), (_,param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                    new_dict[name] = (
                       (param_k.data.detach().clone() * self.sma_count + param_q.data.detach().clone()) / (1.+self.sma_count)
                    )
        else:
            for (name,param_q), (_,param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                    new_dict[name] = param_q.detach().data.clone()
        self.network_sma.load_state_dict(new_dict)
    def forward(self, x, **kwargs):
        return self.network_sma(x, **kwargs)


def train_epoch(data_loader, model, criterion, optimizer, epoch, args):
    model.train()
    total_loss, total_num = 0.0, 0
    if not hasattr(args, 'topk'): args.topk = (1,)
    total_corrects = torch.zeros(len(args.topk), dtype=torch.float)

    show_bar = False
    if not hasattr(args, 'distributed') or not args.distributed or \
       not hasattr(args, 'rank') or args.rank == 0:
        show_bar = True
    data_bar = tqdm.tqdm(data_loader) if show_bar else data_loader

    if hasattr(args, 'use_amp') and args.use_amp:
        if not hasattr(args, 'scaler'):
            args.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    else:
        args.use_amp = False
        args.scaler = None

    if hasattr(args, 'use_sma') and args.use_sma:
        if not hasattr(args, 'model_sma') or args.model_sma is None:
            args.model_sma = MovingAvg(model, 10 * len(data_loader))

    for data, target in data_bar:
        data = data.to(args.device)
        target = target.to(args.device)
        # compute output
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            output = model(data)
            if isinstance(output, (list, tuple)):
                output = output[0]
            loss = criterion(output, target)
        # compute gradient and do SGD step
        if args.use_amp and args.scaler is not None:
            args.scaler.scale(loss).backward()
            args.scaler.step(optimizer)
            args.scaler.update()
            optimizer.zero_grad()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if hasattr(args, 'use_sma') and args.use_sma and \
            hasattr(args, 'model_sma') and args.model_sma is not None:
            args.model_sma.update_sma()

        loss = dist.all_reduce(loss)
        total_loss += loss.item()
        total_num += data.size(0)
        logits = output[0] if isinstance(output, tuple) else output
        preds = torch.argsort(logits, dim=-1, descending=True)
        for i, k in enumerate(args.topk):
            total_corrects[i] += torch.sum((preds[:, 0:k] \
                == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        accuks = 100 * total_corrects / total_num

        if show_bar:
            info = "Train Epoch: [{}/{}] lr: {:.6f} Loss: {:.4f} ".format(
                epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss/len(data_loader))
            info += ' '.join(["Acc@{}: {:.2f}".format(k, accuk) 
                            for k, accuk in zip(args.topk, accuks)])
            data_bar.set_description(info)
    
    return [total_loss/len(data_loader)] + [accuk for accuk in accuks]


def evaluate(data_loader, model, criterion, epoch, args):
    model.eval()
    total_loss, total_num = 0.0, 0
    if not hasattr(args, 'topk'): args.topk = (1,)
    total_corrects = torch.zeros(len(args.topk), dtype=torch.float)

    show_bar = False
    if not hasattr(args, 'distributed') or not args.distributed or \
       not hasattr(args, 'rank') or args.rank == 0:
        show_bar = True
    data_bar = tqdm.tqdm(data_loader) if show_bar else data_loader

    for data, target in data_bar:
        data = data.to(args.device)
        target = target.to(args.device)
        # compute output
        if hasattr(args, 'use_sma') and args.use_sma and \
            hasattr(args, 'model_sma') and args.model_sma is not None:
            output = args.model_sma(data)
        else:
            output = model(data)
        if isinstance(output, (list, tuple)):
            output = output[0]
        loss = criterion(output, target)

        loss = dist.all_reduce(loss)
        total_loss += loss.item()
        total_num += data.size(0)
        logits = output[0] if isinstance(output, tuple) else output
        preds = torch.argsort(logits, dim=-1, descending=True)
        for i, k in enumerate(args.topk):
            total_corrects[i] += torch.sum((preds[:, 0:k] \
                == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        accuks = 100 * total_corrects / total_num

        if show_bar:
            info = "Test  Epoch: [{}/{}] Loss: {:.4f} ".format(
                epoch, args.epochs, total_loss/len(data_loader))
            info += ' '.join(["Acc@{}: {:.2f}".format(k, accuk) 
                            for k, accuk in zip(args.topk, accuks)])
            data_bar.set_description(info)
    
    return [total_loss/len(data_loader)] + [accuk for accuk in accuks]


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule after warmup"""
    if not hasattr(args, 'warmup_epochs'):
        args.warmup_epochs = 0
    if not hasattr(args, 'min_lr'):
        args.min_lr = 0.
    if epoch < args.warmup_epochs:
        lr = max(args.min_lr, args.lr * epoch / args.warmup_epochs)
    else:
        lr = args.lr
        if args.schedule in ['cos', 'cosine']:  # cosine lr schedule
            # lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) # without warmup
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        elif args.schedule in ['step', 'stepwise']:  # stepwise lr schedule
            for milestone in args.lr_drop:
                lr *= 0.1 if epoch >= int(milestone * args.epochs) else 1.
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def change_learning_rate(optimizer, lr):
    """ Set the learning rate to a fixed value """
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def save_model(model, savepath):
    if not os.path.exists(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))
    torch.save(model.state_dict(), savepath)


def load_model(model, loadpath, strict=False):
    if os.path.isfile(loadpath):
        checkpoint = torch.load(loadpath, map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=strict)
        if not strict: print(msg.missing_keys)
        print("=> loaded checkpoint '{}'".format(loadpath))
    else:
        print("=> no checkpoint found at '{}'".format(loadpath))


def save_checkpoint(state, epoch, is_best, save_dir='./'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_path = os.path.join(
        save_dir, 'chkpt_{:04d}.pth.tar'.format(epoch)
        )
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'best.pth.tar')
        torch.save(state, best_path)
        

def load_checkpoint(ckptpath, model, optimizer=None, args=None, strict=False):
    if os.path.isfile(ckptpath):
        checkpoint = torch.load(ckptpath, map_location='cpu')
        state_dict = convert_state_dict(checkpoint['state_dict'])
        msg = model.load_state_dict(state_dict, strict=strict)
        print(msg.missing_keys)
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(args.device)
        if args is not None:
            args.start_epoch = 0
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(ckptpath, args.start_epoch))
        else:
            print("=> loaded checkpoint '{}'".format(ckptpath))
    else:
        print("=> no checkpoint found at '{}'".format(ckptpath))


def convert_state_dict(state_dict):
    firstkey = next(iter(state_dict))
    if firstkey.startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not k.endswith('total_ops') and not k.endswith('total_params'):
                name = k[7:] # 7 = len('module.')
                new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

        
if __name__ == '__main__':

    x = torch.randn([10, 3, 4, 5])
    