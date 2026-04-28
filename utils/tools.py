import os
import numpy as np
import torch
import torch.nn as nn
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Iterable
from sklearn.decomposition import PCA

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate if epoch < 2 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'type5':
        lr_adjust = {epoch: args.learning_rate if epoch % 10 == 0 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'TST':
        if scheduler is not None:
            lr_adjust = {epoch: scheduler.get_last_lr()[0]}
        else:
            lr_adjust = {}
    else:
        lr_adjust = {}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'\tEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'\tValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

   
class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean
    

def save_to_csv(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    data = pd.DataFrame({'true': true, 'preds': preds})
    data.to_csv(name, index=False, sep=',')


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def visual_weights(weights, name='./pic/test.pdf'):
    """
    Weights visualization
    """
    fig, ax = plt.subplots()
    # im = ax.imshow(weights, cmap='plasma_r')
    im = ax.imshow(weights, cmap='YlGnBu')
    fig.colorbar(im, pad=0.03, location='top')
    plt.savefig(name, dpi=500, pad_inches=0.02)
    plt.close()


def plot_token_distribution(train_tokens: torch.Tensor, test_tokens: torch.Tensor, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    train_tokens = train_tokens.flatten()
    test_tokens = test_tokens.flatten()
    train_ids, train_counts = np.unique(train_tokens, return_counts=True)
    test_ids, test_counts = np.unique(test_tokens, return_counts=True)

    plt.clf()
    plt.bar(train_ids, train_counts, label='Train')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_token_distribution.png'))

    plt.clf()
    plt.bar(test_ids, test_counts, label='Test')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'test_token_distribution.png'))

    plt.clf()
    plt.bar(train_ids, train_counts, label='Train')
    plt.bar(test_ids, test_counts, label='Test')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_test_token_distribution.png'))
    plt.clf()


def plot_PCA(train_ids, x, save_path, max_token_num):
    train_tokens = train_ids.flatten()
    token_ids, token_counts = np.unique(train_tokens, return_counts=True)
    counts = np.zeros((max_token_num,))
    counts[token_ids] = token_counts

    used = np.where(counts > 0)
    x = x[used]
    counts = counts[used]

    pca = PCA(n_components=2)
    x_reduced = pca.fit_transform(x)
    scatter = plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=counts, cmap='hot')
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA with weights')
    plt.colorbar(scatter)
    plt.savefig(save_path)
    plt.clf()


def statistic_freqs(train_ids):
    train_tokens = train_ids.flatten()
    _, token_counts = np.unique(train_tokens, return_counts=True)
    total = len(train_tokens)
    for threshold in [10, 5, 2, 1.5, 1.2, 1, 0.8, 0.7, 0.6, 0.5, 0.2, 0.1]:
        count_floor = total * (threshold / 100.0)
        print(f'Freqs large than {threshold}%: {np.sum(token_counts >= count_floor)}')


def plot_token_distribution_with_stratify(
    train_tokens: torch.Tensor,
    test_tokens: torch.Tensor,
    save_dir: str,
    max_token_num=255,
    freq=True,
):
    os.makedirs(save_dir, exist_ok=True)

    train_tokens = train_tokens.flatten()
    test_tokens = test_tokens.flatten()
    train_ids, train_counts = np.unique(train_tokens, return_counts=True)
    test_ids, test_counts = np.unique(test_tokens, return_counts=True)

    if freq:
        train_counts = train_counts / len(train_tokens)
        test_counts = test_counts / len(test_tokens)

    plt.clf()
    plt.bar(train_ids, train_counts, label='Train')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_token_distribution.png'))

    plt.clf()
    plt.bar(test_ids, test_counts, label='Test')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'test_token_distribution.png'))

    train_hist = np.zeros((max_token_num,))
    test_hist = np.zeros((max_token_num,))
    train_hist[train_ids] = train_counts
    test_hist[test_ids] = test_counts

    low = np.minimum(train_hist, test_hist)
    high = np.maximum(train_hist, test_hist)
    low_colors = ['blue' if train_value < test_value else 'orange' for train_value, test_value in zip(train_hist, test_hist)]
    high_colors = ['orange' if train_value < test_value else 'blue' for train_value, test_value in zip(train_hist, test_hist)]

    plt.clf()
    x = np.arange(len(train_hist))
    plt.bar(x, low, color=low_colors, label='Test')
    plt.bar(x, high, bottom=low, color=high_colors, label='Train')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_test_token_distribution.png'))
    plt.clf()


def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    return clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def plot_and_save_reconstruction(model, test_loader, save_path, dims_to_plot=None):
    model.eval()
    test_data_iter = iter(test_loader)
    batch_x, batch_y, _, _ = next(test_data_iter)

    sample_index = 0
    sample_data_x = batch_x[sample_index].unsqueeze(0).float().to(next(model.parameters()).device)
    sample_data_y = batch_y[sample_index].unsqueeze(0).float().to(next(model.parameters()).device)

    with torch.no_grad():
        reconstructed, _, _ = model(sample_data_x, sample_data_y)

    original_data = sample_data_y.squeeze(0).cpu().numpy()
    reconstructed_data = reconstructed.squeeze(0)[-sample_data_y.size(1):].cpu().numpy()

    os.makedirs(save_path, exist_ok=True)
    if original_data.ndim == 1 or (original_data.ndim == 2 and original_data.shape[1] == 1):
        original_data = original_data.squeeze()
        reconstructed_data = reconstructed_data.squeeze()

        plt.figure(figsize=(12, 4))
        plt.plot(original_data, label='Original')
        plt.plot(reconstructed_data, label='Reconstructed')
        plt.title("Single-variable Reconstruction Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'single_variable_reconstruction.pdf'))
        plt.close()
        return

    num_dims = original_data.shape[1]
    if dims_to_plot is None:
        dims_to_plot = list(range(min(4, num_dims)))

    fig, axes = plt.subplots(len(dims_to_plot), 1, figsize=(12, len(dims_to_plot) * 2))
    if len(dims_to_plot) == 1:
        axes = [axes]
    for idx, dim in enumerate(dims_to_plot):
        axes[idx].plot(original_data[:, dim], label=f'Original Dim {dim}')
        axes[idx].plot(reconstructed_data[:, dim], label=f'Recon Dim {dim}')
        axes[idx].set_title(f"Data Comparison - Dim {dim}")
        axes[idx].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'multi_variable_dim_comparison.pdf'))
    plt.close()

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    for dim in dims_to_plot:
        ax[0].plot(original_data[:, dim], label=f'Dim {dim}')
        ax[1].plot(reconstructed_data[:, dim], label=f'Dim {dim}')
    ax[0].set_title("Original Data - Selected Dims")
    ax[1].set_title("Reconstructed Data - Selected Dims")
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'multi_variable_total_reconstruction.pdf'))
    plt.close()


class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims  # The new order of dimensions

    def forward(self, x):
        return x.permute(*self.dims)
    
class Reshape(nn.Module):
    def __init__(self, *dims):
        super(Reshape, self).__init__()
        self.dims = dims  # The new order of dimensions

    def forward(self, x):
        return x.reshape(*self.dims)

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# def set_random_seed(random_seed):
#     torch.manual_seed(random_seed)
#     # torch.cuda.manual_seed(random_seed)
#     # torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
#     # torch.backends.cudnn.deterministic = True
#     # torch.backends.cudnn.benchmark = False
#     np.random.seed(random_seed)
#     random.seed(random_seed)
