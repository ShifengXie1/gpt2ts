import argparse
import os

import torch

from exp.exp_token_llm import TokenLLM_Main, build_setting
from utils.tools import set_random_seed


def str2bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def infer_num_channels(args):
    if not args.use_multivariate:
        return 1

    csv_path = os.path.join(args.root_path, args.data_path)
    with open(csv_path, "r", encoding="utf-8") as file:
        header = file.readline().strip().split(",")
    return max(1, len(header) - 1)


def resolve_gpt_local_path(path):
    if path is None:
        return None
    return os.path.abspath(path)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_training", type=int, default=0)
    parser.add_argument("--zero_shot", type=str2bool, default=False)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt2ts")
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default="./data/ETT")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--patch_size", "--patch_len", dest="patch_size", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--scaling_eps", type=float, default=1e-8)
    parser.add_argument("--gpt_model_name", type=str, default="openai-community/gpt2")
    parser.add_argument("--gpt_local_path", type=str, default="./gpt")
    parser.add_argument("--use_pretrained_gpt2", type=str2bool, default=True)
    parser.add_argument("--prefer_local_gpt2", type=str2bool, default=True)
    parser.add_argument("--gpt_local_files_only", type=str2bool, default=True)
    parser.add_argument("--tokenizer_path", "--vqvae_model_path", dest="tokenizer_path", type=str, default=None)
    parser.add_argument("--vq_model", type=str, default="GPT2ClusterVQ")
    parser.add_argument("--tokenizer_d_model", type=int, default=64)
    parser.add_argument("--tokenizer_block_num", "--block_num", dest="tokenizer_block_num", type=int, default=2)
    parser.add_argument("--wave_length", type=int, default=None)
    parser.add_argument("--n_embed", type=int, default=None)
    parser.add_argument("--revin", type=int, default=1)
    parser.add_argument("--affine", type=int, default=0)
    parser.add_argument("--subtract_last", type=int, default=0)
    parser.add_argument("--chan_indep", type=int, default=0)
    parser.add_argument("--entropy_penalty", type=float, default=0.0)
    parser.add_argument("--entropy_temp", type=float, default=0.5)
    parser.add_argument("--gpt2_hidden_size", type=int, default=768)
    parser.add_argument("--init_gpt2_codebook", type=str2bool, default=False)
    parser.add_argument("--num_ts_clusters", type=int, default=128)
    parser.add_argument("--num_lm_clusters", type=int, default=128)
    parser.add_argument("--ts_centers_path", type=str, default=None)
    parser.add_argument("--cluster_tau", type=float, default=0.5)
    parser.add_argument("--cluster_hard", type=str2bool, default=False)
    parser.add_argument("--allowed_token_count", type=int, default=4096)
    parser.add_argument("--encoder_dropout", type=float, default=None)
    parser.add_argument("--tcn_encoder_layers", type=int, default=3)
    parser.add_argument("--tcn_encoder_kernel_size", type=int, default=3)
    parser.add_argument("--decoder_dropout", type=float, default=0.2)
    parser.add_argument("--tcn_decoder_layers", type=int, default=3)
    parser.add_argument("--tcn_decoder_kernel_size", type=int, default=3)
    parser.add_argument("--debug_shapes", type=str2bool, default=False)
    parser.add_argument("--ts_embedding_mode", type=str, default="soft_vocab")
    parser.add_argument("--ts_mapping_tau", type=float, default=0.2)
    parser.add_argument("--ts_mapping_top_k", type=int, default=0)
    parser.add_argument("--ts_mapping_normalize", type=str2bool, default=True)
    parser.add_argument("--lambda_ts_mapping_entropy", type=float, default=0.0)
    parser.add_argument("--use_revin", type=str2bool, default=True)
    parser.add_argument("--use_alignment", type=str2bool, default=True)
    parser.add_argument("--use_trend_loss", type=str2bool, default=True)
    parser.add_argument("--use_token_distribution_loss", type=str2bool, default=True)
    parser.add_argument("--lambda_token", type=float, default=0.2)
    parser.add_argument("--lambda_lm_token", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lambda_pred", type=float, default=1.0)
    parser.add_argument("--lambda_point", type=float, default=0.5)
    parser.add_argument("--lambda_diff", type=float, default=0.2)
    parser.add_argument("--lambda_trend", type=float, default=0.2)
    parser.add_argument("--lambda_tokenizer_latent", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_diagnostics", type=str2bool, default=True)
    parser.add_argument("--grad_check_interval", type=int, default=100)
    parser.add_argument("--grad_warn_threshold", type=float, default=1e-12)
    parser.add_argument("--lambda_center_anchor", type=float, default=0.001)
    parser.add_argument("--lradj", type=str, default="type3")
    parser.add_argument("--scheduler_type", type=str, default="warmup_cosine")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--use_multivariate", type=str2bool, default=False)
    parser.add_argument("--use_multi_gpu", type=str2bool, default=False)
    parser.add_argument("--use_amp", type=str2bool, default=False)
    parser.add_argument("--target_col", type=str, default="OT")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")
    parser.add_argument("--augmentation_ratio", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    args.use_multivariate = bool(args.use_multivariate)
    args.use_gpu = bool(args.use_gpu) and torch.cuda.is_available()
    args.use_multi_gpu = bool(args.use_multi_gpu)
    args.use_amp = bool(args.use_amp) and args.use_gpu
    args.gpt_local_path = resolve_gpt_local_path(args.gpt_local_path)
    args.tokenizer_path = None if args.tokenizer_path in {None, ""} else os.path.abspath(args.tokenizer_path)
    args.num_ts_clusters = max(1, int(args.num_ts_clusters))
    if args.n_embed is None:
        args.n_embed = args.num_ts_clusters
    args.n_embed = max(1, int(args.n_embed))
    args.wave_length = args.patch_size if args.wave_length is None else max(1, int(args.wave_length))
    args.num_lm_clusters = max(1, int(args.num_lm_clusters))
    args.ts_centers_path = None if args.ts_centers_path in {None, ""} else os.path.abspath(args.ts_centers_path)
    args.cluster_tau = max(float(args.cluster_tau), 1e-6)
    args.cluster_hard = bool(args.cluster_hard)
    args.ts_mapping_tau = max(float(args.ts_mapping_tau), 1e-6)
    args.ts_mapping_top_k = max(0, int(args.ts_mapping_top_k))
    args.ts_mapping_normalize = bool(args.ts_mapping_normalize)
    args.use_revin = bool(args.use_revin)
    args.encoder_dropout = args.dropout if args.encoder_dropout is None else float(args.encoder_dropout)
    args.tcn_encoder_layers = max(1, int(args.tcn_encoder_layers))
    args.tcn_encoder_kernel_size = int(args.tcn_encoder_kernel_size)
    if args.tcn_encoder_kernel_size < 1 or args.tcn_encoder_kernel_size % 2 == 0:
        raise ValueError("--tcn_encoder_kernel_size must be a positive odd integer.")
    args.tcn_decoder_layers = max(1, int(args.tcn_decoder_layers))
    args.tcn_decoder_kernel_size = int(args.tcn_decoder_kernel_size)
    if args.tcn_decoder_kernel_size < 1 or args.tcn_decoder_kernel_size % 2 == 0:
        raise ValueError("--tcn_decoder_kernel_size must be a positive odd integer.")
    args.use_token_distribution_loss = bool(args.use_token_distribution_loss)
    args.lambda_center_anchor = float(args.lambda_center_anchor)
    args.grad_diagnostics = bool(args.grad_diagnostics)
    args.grad_check_interval = max(1, int(args.grad_check_interval))
    args.grad_warn_threshold = max(0.0, float(args.grad_warn_threshold))
    args.patch_size = max(2, int(args.patch_size))
    args.patch_len = args.patch_size
    args.allowed_token_count = max(args.num_lm_clusters, int(args.allowed_token_count))
    args.devices = str(args.devices if args.devices is not None else args.gpu)
    args.device_ids = [
        int(device)
        for device in args.devices.replace(" ", "").split(",")
        if device != ""
    ] or [args.gpu]
    if args.use_multi_gpu:
        args.gpu = args.device_ids[0]

    args.features = "M" if args.use_multivariate else "S"
    args.target = args.target_col
    args.c_in = infer_num_channels(args)
    args.c_out = args.c_in if args.use_multivariate else 1
    return args


def main():
    args = build_args()
    if args.seed is not None:
        set_random_seed(args.seed)

    variable_mode = "multivariate_fusion" if args.use_multivariate else "univariate_target"
    print(
        f"Variable mode: {variable_mode} | "
        f"features={args.features} | c_in={args.c_in} | c_out={args.c_out} | "
        f"target_col={args.target_col}"
    )

    setting = build_setting(args)
    exp = TokenLLM_Main(args)

    should_train = bool(args.is_training) and not args.zero_shot
    load_checkpoint = should_train or bool(args.checkpoint)

    if should_train:
        print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        exp.train(setting)

    if args.zero_shot:
        mode = "zero-shot evaluating"
    else:
        mode = "testing" if args.is_training else "evaluating"

    print(f">>>>>>>{mode} : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    exp.test(
        setting,
        checkpoint_path=args.checkpoint,
        load_checkpoint=load_checkpoint,
    )


if __name__ == "__main__":
    main()
