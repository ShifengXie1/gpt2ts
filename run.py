import argparse
import os

import torch

from exp.exp_main import TokenLLM_Main
from utils.tools import set_random_seed

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--features", type=str, default="M", choices=["M", "S", "MS"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--c_in", type=int, default=None)
    parser.add_argument("--c_out", type=int, default=None)
    

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

    args = parser.parse_args()
    
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

    exp = TokenLLM_Main(args)
    

    


if __name__ == "__main__":
    main()
