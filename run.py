import argparse
import os

import torch

from exp.exp_main import Exp_Main
from utils.tools import set_random_seed


def build_args():
    parser = argparse.ArgumentParser()
    
    '''模型'''
    parser.add_argument('--model', type = str, required = False, choices = ['gpt2ts'], default = 'gpt2ts', help = 'model of experiment')
    parser.add_argument('--task_name', type = str, required = False, choices = ['long_term_forecast'], default = 'long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1, help='kept for script compatibility')
    
    '''模型参数'''
    parser.add_argument('--seq_len', type = int, default = 512, help = 'length of the look back window')
    parser.add_argument('--pred_len', type = int, default = 96, help = 'prediction length')
    parser.add_argument('--label_len', type=int, default=0, help='label length')
    parser.add_argument('--d_model', type = int, default = 256, help = 'embedding dimension')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'learning rate')
    parser.add_argument('--weight_decay', type = float, default = 0.00, help = 'pytorch weight decay factor')
    parser.add_argument('--patch_len', '--patch_size', dest='patch_len', type = int, default = 16, help = 'Patch size')
    parser.add_argument('--stride', type = int, default = 8, help = 'Stride')
    parser.add_argument('--dropout', type = float, default = 0.05, help = 'dropout for mixer')
    parser.add_argument('--embedding_dropout', type = float, default = 0.05, help = 'dropout for embedding layer')
    parser.add_argument('--patience', type = int, default = 10, help = 'patience')
    parser.add_argument('--train_epochs', type = int, default = 10, help = 'train epochs')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'smoothL1'])
    parser.add_argument('--lradj', type=str, default='none', help='learning-rate adjustment policy')
    parser.add_argument('--n_layers', type=int, default=0, help='number of GPT-2 layers to keep; 0 keeps all layers')
    parser.add_argument('--num_clusters', type=int, default=64, help='number of clusters for GPT vocab embeddings')
    parser.add_argument('--cluster_sample_size', type=int, default=8192, help='deprecated; kept for compatibility and ignored')
    parser.add_argument('--vocab_cluster_sample_size', type=int, default=20000, help='max vocab embeddings sampled for k-means')
    parser.add_argument('--kmeans_iters', type=int, default=8, help='k-means iterations')
    parser.add_argument('--cluster_residual_scale', type=float, default=1.0, help='scale for distance-preserving cluster residuals')
    parser.add_argument('--cluster_normalize', type=bool, default=True, help='normalize embeddings when assigning clusters')
    parser.add_argument('--cluster_seed', type=int, default=None, help='seed for cluster init and random center mapping')
    parser.add_argument('--history_lookup_temperature', type=float, default=0.2, help='temperature for mapping predicted embeddings to history patches')
    parser.add_argument('--hard_patch_lookup', action='store_true', default=False, help='use nearest historical patch at evaluation time')
    parser.add_argument('--forecast_temperature', type=float, default=1.0, help='temperature for token embedding prediction')
    parser.add_argument('--forecast_top_k', type=int, default=64, help='use top-k token embeddings for differentiable predicted embedding')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank for GPT-2 attention projections')
    parser.add_argument('--lora_alpha', type=float, default=16.0, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lora_target', type=str, default='c_attn,c_proj', help='comma-separated GPT-2 attention projection names')
    parser.add_argument('--gpt_local_path', type=str, default=None, help='local GPT-2 folder')
    parser.add_argument('--gpt_model_name', type=str, default='openai-community/gpt2', help='fallback HuggingFace GPT-2 model id')
    parser.add_argument('--gpt_local_files_only', type=bool, default=True, help='load GPT-2 from local files only')
    parser.add_argument('--use_pretrained_gpt2', type=bool, default=True, help='load pretrained GPT-2 weights')
    
    '''环境'''
    parser.add_argument('--use_gpu', type = bool, default = True, help = 'use gpu')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--use_multi_gpu', action = 'store_true', help = 'use multiple gpus', default = False)
    parser.add_argument('--devices', type = str, default = '0,1', help = 'device ids of multile gpus')
    parser.add_argument('--use_amp', action = 'store_true', help = 'use automatic mixed precision training', default = False)
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    
    '''输入数据'''
    parser.add_argument('--data', type = str, default = 'ETTh1', choices = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'Weather', 'Traffic'], help = 'dataset')
    parser.add_argument("--features", type=str, default="M", choices=["M", "S", "MS"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--c_in", type=int, default=None)
    parser.add_argument("--c_out", type=int, default=None)
    parser.add_argument("--embed", type=str, default="timeF", help="time feature encoding type")
    parser.add_argument("--freq", type=str, default="h", help="time feature frequency")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly", help="seasonality for M4")
    parser.add_argument("--augmentation_ratio", type=float, default=0.0, help="data augmentation ratio")
    parser.add_argument("--model_id", type=str, default="ETTh1", help="dataset id used by UEA loader")
    

    args = parser.parse_args()
    
    '''GPU设置'''
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.cluster_seed is None:
        args.cluster_seed = args.seed
    
    '''数据集'''
    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'root_path': './data/ETT/', 'T': 'OT', 'freq': 'h', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'root_path': './data/ETT/', 'T': 'OT', 'freq': 'h', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'root_path': './data/ETT/', 'T': 'OT', 'freq': 't', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'root_path': './data/ETT/', 'T': 'OT', 'freq': 't', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'Weather': {'data': 'weather.csv', 'root_path': './data/weather/', 'T': 'OT', 'freq': 'h', 'M': [21, 21], 'S': [1, 1], 'MS': [21, 1]},
        'Traffic': {'data': 'traffic.csv', 'root_path': './data/traffic/', 'T': 'OT', 'freq': 'h', 'M': [862, 862], 'S': [1, 1], 'MS': [862, 1]},
        'Electricity': {'data': 'electricity.csv', 'root_path': './data/electricity/', 'T': 'OT', 'freq': 'h', 'M': [321, 321], 'S': [1, 1], 'MS': [321, 1]},
        'ILI':  {'data': 'national_illness.csv', 'root_path': './data/illness/', 'T': 'OT', 'freq': 'h', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'Solar':  {'data': 'solar_AL.txt', 'root_path': './data/solar/', 'T': None, 'freq': 'h', 'M': [137, 137], 'S': [None, None], 'MS': [None, None]},
    }
    
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.root_path = data_info['root_path']
        args.target = data_info['T']
        args.freq = data_info['freq']
        args.c_in = data_info[args.features][0]
        args.c_out = data_info[args.features][1]
        if args.target_col is None:
            args.target_col = args.target
    
    return args

def build_results_dir(args):
    results_dir = args.results_dir if hasattr(args, "results_dir") else "./results"
    os.makedirs(results_dir, exist_ok=True)
    return 


def main():
    args = build_args()
    build_results_dir(args)
    set_random_seed(args.seed)

    setting = '{}_{}_dec-sl{}_pl{}_dm{}_bt{}_ptl{}_stl{}_sd{}'.format(args.model, args.data, args.seq_len, args.pred_len, args.d_model, args.batch_size, args.patch_len, args.stride, args.seed)
    
    exp = Exp_Main(args)
    
    print('Start Training- {}'.format(setting))
    exp.train(setting)
        
    print('Start Testing- {}'.format(setting))
    loss_mse, loss_mae = exp.test(setting)


if __name__ == "__main__":
    main()
