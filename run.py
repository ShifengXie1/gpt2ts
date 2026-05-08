import argparse
import os

import torch

from exp.exp_main import Exp_Main
from utils.tools import set_random_seed


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def build_args():
    parser = argparse.ArgumentParser()
    
    # Model
    parser.add_argument('--model', type = str, required = False, choices = ['gpt2ts'], default = 'gpt2ts', help = 'model of experiment')
    parser.add_argument('--task_name', type = str, required = False, choices = ['long_term_forecast'], default = 'long_term_forecast')
    
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
    parser.add_argument('--dropout', type = float, default = 0.05, help = 'dropout for output layer')
    parser.add_argument('--embedding_dropout', type = float, default = 0.05, help = 'dropout for embedding layer')
    parser.add_argument('--patience', type = int, default = 10, help = 'patience')
    parser.add_argument('--train_epochs', type = int, default = 10, help = 'train epochs')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'smoothL1'])
    parser.add_argument('--lradj', type=str, default='none', help='learning-rate adjustment policy')
    parser.add_argument('--n_layers', type=int, default=0, help='number of GPT-2 layers to keep; 0 keeps all layers')
    parser.add_argument('--cluster_num', type=int, default=128, help='number of patch/vocab clusters')
    parser.add_argument('--cluster_normalize', type=bool, default=True, help='normalize embeddings when assigning clusters')
    parser.add_argument('--cluster_seed', type=int, default=None, help='seed for cluster init and random center mapping')
    parser.add_argument('--patch_match_tol', type=float, default=1e-6, help='tolerance for exact patch lookup before nearest-neighbor fallback')
    parser.add_argument('--patch_level_normalize', type=str2bool, default=True, help='normalize each patch before clustering and nearest-neighbor matching')
    parser.add_argument('--patch_norm_eps', type=float, default=1e-5, help='epsilon for patch-level normalization')
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
    parser.add_argument('--use_multi_gpu', action = 'store_true', help = 'use multiple gpus', default = False)
    parser.add_argument('--devices', type = str, default = '0,1', help = 'device ids of multile gpus')
    parser.add_argument('--use_amp', action = 'store_true', help = 'use automatic mixed precision training', default = False)
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    
    # Input data
    parser.add_argument('--data', type = str, default = 'ETTh1', choices = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'Weather', 'Traffic'], help = 'dataset')
    parser.add_argument("--features", type=str, default="S", choices=["M", "S", "MS"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--c_in", type=int, default=None)
    parser.add_argument("--c_out", type=int, default=None)
    parser.add_argument("--embed", type=str, default="timeF", help="time feature encoding type")
    parser.add_argument("--freq", type=str, default="h", help="time feature frequency")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly", help="seasonality for M4")
    parser.add_argument("--augmentation_ratio", type=float, default=0.0, help="data augmentation ratio")
    parser.add_argument("--model_id", type=str, default="ETTh1", help="dataset id used by UEA loader")
    

    args = parser.parse_args()
    
    # GPU setup
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        args.gpu = int(args.devices.split(',')[0])

    
    # Dataset presets
    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'root_path': './data/ETT/', 'T': 'OT', 'freq': 'h', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'root_path': './data/ETT/', 'T': 'OT', 'freq': 'h', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'root_path': './data/ETT/', 'T': 'OT', 'freq': 't', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'root_path': './data/ETT/', 'T': 'OT', 'freq': 't', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'Weather': {'data': 'weather.csv', 'root_path': './data/weather/', 'T': 'OT', 'freq': 'h', 'M': [21, 21], 'S': [1, 1], 'MS': [21, 1]},
        'Traffic': {'data': 'traffic.csv', 'root_path': './data/traffic/', 'T': 'OT', 'freq': 'h', 'M': [862, 862], 'S': [1, 1], 'MS': [862, 1]},
        'Electricity': {'data': 'electricity.csv', 'root_path': './data/electricity/', 'T': 'OT', 'freq': 'h', 'M': [321, 321], 'S': [1, 1], 'MS': [321, 1]}
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

    setting = '{}_{}_dec-sl{}_pl{}_bt{}_ptl{}_stl{}_sd{}'.format(args.model, args.data, args.seq_len, args.pred_len, args.batch_size, args.patch_len, args.stride, args.seed)
    
    exp = Exp_Main(args)
    
    print('Start Training- {}'.format(setting))
    exp.train(setting)
        
    print('Start Testing- {}'.format(setting))
    exp.test(setting)


if __name__ == "__main__":
    main()
