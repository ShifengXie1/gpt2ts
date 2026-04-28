import json
import torch
import random
import os
import sys
import numpy as np
import warnings

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

warnings.filterwarnings('ignore')
from data_provider.data_factory import data_provider
from args import args
from process import Trainer
from models.GPT2ClusterVQ import VQVAE as GPT2ClusterVQ
# from dataset import Dataset
import torch.utils.data as Data

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def get_data(flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader

def main():
    seed_everything(seed=2024)

    # train_dataset = Dataset(device=args.device, mode='train', args=args)
    # train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    # test_dataset = Dataset(device=args.device, mode='test', args=args)
    # test_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    train_data, train_loader = get_data(flag='train')
    vali_data, vali_loader = get_data(flag='val')
    test_data, test_loader = get_data(flag='test')

    print('dataset initial ends')
    # model = VQVAE(data_shape=(args.token_len, args.enc_in), hidden_dim=args.d_model, n_embed=args.n_embed,
    #                 wave_length=args.wave_length, block_num=args.block_num)
    
    # model = W_VQVAE(args)
    
    if args.vq_model != 'GPT2ClusterVQ':
        raise ValueError('Invalid VQ model name. Use GPT2ClusterVQ.')
    model = GPT2ClusterVQ(args)
    
    
    print('model initial ends')

    trainer = Trainer(args, model, train_loader, vali_loader, test_loader, verbose=True)
    print('trainer initial ends')

    if args.is_training:
        trainer.train()

    trainer.test()


if __name__ == '__main__':
    main()

