import argparse
import os

import torch

from exp.exp_main import Exp_Main
from utils.tools import set_random_seed


def build_args():
    parser = argparse.ArgumentParser()
    
    # Model params
    parser.add_argument('--seq_len', type = int, default = 512, help = 'length of the look back window')
    parser.add_argument('--pred_len', type = int, default = 96, help = 'prediction length')
    parser.add_argument('--label_len', type=int, default=0, help='label length')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size')
    parser.add_argument('--token_batch_size', type=int, default=None, help='batch size for token-level GPT/LoRA training')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'learning rate')
    parser.add_argument('--weight_decay', type = float, default = 0.00, help = 'pytorch weight decay factor')
    parser.add_argument('--patch_len', type = int, default = 16, help = 'Patch size')
    parser.add_argument('--stride', type = int, default = 16, help = 'Stride')
    parser.add_argument('--patience', type = int, default = 10, help = 'patience')
    parser.add_argument('--train_epochs', type = int, default = 10, help = 'train epochs')
    parser.add_argument('--lradj', type=str, default='none', help='learning-rate adjustment policy')
    parser.add_argument('--n_layers', type=int, default=0, help='number of GPT-2 layers to keep; 0 keeps all layers')
    parser.add_argument('--cluster_num', type=int, default=512, help='number of historical patch motifs')
    parser.add_argument('--cluster_normalize', type=bool, default=False, help='z-normalize each patch before motif clustering')
    parser.add_argument('--cluster_seed', type=int, default=None, help='seed for motif clustering')
    parser.add_argument('--kmeans_iters', type=int, default=30, help='k-means iterations for historical patch motif clustering')
    parser.add_argument('--candidate_token_count', type=int, default=0, help='number of GPT vocab tokens available for motif assignment; 0 uses all non-special tokens')
    parser.add_argument('--token_train_stride', type=int, default=1, help='stride over train patch-token windows for GPT/LoRA training')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank for GPT-2 attention projections')
    parser.add_argument('--lora_alpha', type=float, default=16.0, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lora_target', type=str, default='c_attn,c_proj', help='comma-separated GPT-2 attention projection names')
    parser.add_argument('--gpt_local_path', type=str, default='./gpt', help='local GPT-2 folder')
    parser.add_argument('--gpt_local_files_only', type=bool, default=True, help='load GPT-2 from local files only')
    parser.add_argument('--use_pretrained_gpt2', type=bool, default=True, help='load pretrained GPT-2 weights')
    
    # Environment
    parser.add_argument('--use_gpu', type = bool, default = True, help = 'use gpu')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--use_amp', action = 'store_true', help = 'use automatic mixed precision training', default = False)
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    
    # Input data
    parser.add_argument('--data', type = str, default = 'ETTh1', choices = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'], help = 'dataset')
    parser.add_argument("--features", type=str, default="S", choices=["S", "M"])
    parser.add_argument("--augmentation_ratio", type=float, default=0.0, help="data augmentation ratio")
    

    args = parser.parse_args()
    
    # GPU setup
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.model = 'gpt2ts'
    args.task_name = 'long_term_forecast'
    args.embed = 'timeF'
    args.seasonal_patterns = None

    
    # Dataset presets
    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'root_path': './data/ETT/', 'T': 'OT', 'freq': 'h', 'S': [1, 1], 'M': [7, 7]},
        'ETTh2': {'data': 'ETTh2.csv', 'root_path': './data/ETT/', 'T': 'OT', 'freq': 'h', 'S': [1, 1], 'M': [7, 7]},
        'ETTm1': {'data': 'ETTm1.csv', 'root_path': './data/ETT/', 'T': 'OT', 'freq': 't', 'S': [1, 1], 'M': [7, 7]},
        'ETTm2': {'data': 'ETTm2.csv', 'root_path': './data/ETT/', 'T': 'OT', 'freq': 't', 'S': [1, 1], 'M': [7, 7]}
    }
    
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.root_path = data_info['root_path']
        args.target = data_info['T']
        args.freq = data_info['freq']
        args.c_in = data_info[args.features][0]
        args.c_out = data_info[args.features][1]

    return args

def build_results_dir(args):
    results_dir = args.results_dir if hasattr(args, "results_dir") else "./results"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def main():
    args = build_args()
    build_results_dir(args)
    set_random_seed(args.seed)
    if args.cluster_seed is None:
        args.cluster_seed = args.seed

    setting = 'gpt2ts_{}_dec-sl{}_pl{}_bt{}_ptl{}_stl{}_sd{}'.format(args.data, args.seq_len, args.pred_len, args.batch_size, args.patch_len, args.stride, args.seed)
    
    exp = Exp_Main(args)
    
    print('Start Training- {}'.format(setting))
    exp.train(setting)
        
    print('Start Testing- {}'.format(setting))
    exp.test(setting)


if __name__ == "__main__":
    main()
