import argparse
import os

import torch

from exp.exp_main import TokenLLM_Main
from utils.tools import set_random_seed

def build_args():
    parser = argparse.ArgumentParser()
    
    '''模型'''
    parser.add_argument('--model', type = str, required = False, choices = ['gpt2ts'], default = 'gpt2ts', help = 'model of experiment')
    parser.add_argument('--task_name', type = str, required = False, choices = ['long_term_forecast'], default = 'long_term_forecast')
    
    '''模型参数'''
    parser.add_argument('--seq_len', type = int, default = 512, help = 'length of the look back window')
    parser.add_argument('--pred_len', type = int, default = 96, help = 'prediction length')
    parser.add_argument('--d_model', type = int, default = 256, help = 'embedding dimension')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'learning rate')
    parser.add_argument('--weight_decay', type = float, default = 0.00, help = 'pytorch weight decay factor')
    parser.add_argument('--patch_len', type = int, default = 16, help = 'Patch size')
    parser.add_argument('--stride', type = int, default = 8, help = 'Stride')
    parser.add_argument('--dropout', type = float, default = 0.05, help = 'dropout for mixer')
    parser.add_argument('--embedding_dropout', type = float, default = 0.05, help = 'dropout for embedding layer')
    parser.add_argument('--patience', type = int, default = 10, help = 'patience')
    parser.add_argument('--train_epochs', type = int, default = 10, help = 'train epochs')
    
    '''环境'''
    parser.add_argument('--use_gpu', type = bool, default = True, help = 'use gpu')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--use_multi_gpu', action = 'store_true', help = 'use multiple gpus', default = False)
    parser.add_argument('--devices', type = str, default = '0,1', help = 'device ids of multile gpus')
    parser.add_argument('--use_amp', action = 'store_true', help = 'use automatic mixed precision training', default = False)
    
    '''输入数据'''
    parser.add_argument('--data', type = str, default = 'ETTh1', choices = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'Weather', 'Traffic'], help = 'dataset')
    parser.add_argument("--features", type=str, default="M", choices=["M", "S", "MS"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--c_in", type=int, default=None)
    parser.add_argument("--c_out", type=int, default=None)
    

    args = parser.parse_args()
    
    '''GPU设置'''
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    '''数据集'''
    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'Weather': {'data': 'weather.csv', 'root_path': './data/weather/', 'T': 'OT', 'M': [21, 21], 'S': [1, 1], 'MS': [21, 1]},
        'Traffic': {'data': 'traffic.csv', 'root_path': './data/traffic/', 'T': 'OT', 'M': [862, 862], 'S': [1, 1], 'MS': [862, 1]},
        'Electricity': {'data': 'electricity.csv', 'root_path': './data/electricity/', 'T': 'OT', 'M': [321, 321], 'S': [1, 1], 'MS': [321, 1]},
        'ILI':  {'data': 'national_illness.csv', 'root_path': './data/illness/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'Solar':  {'data': 'solar_AL.txt', 'root_path': './data/solar/', 'T': None, 'M': [137, 137], 'S': [None, None], 'MS': [None, None]},
    }
    
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.root_path = data_info['root_path']
        args.target = data_info['T']
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
    
    exp = TokenLLM_Main(args)
    
    print('Start Training- {}'.format(setting))
    exp.train(setting)
        
    print('Start Testing- {}'.format(setting))
    loss_mse, loss_mae = exp.test(setting)


if __name__ == "__main__":
    main()
