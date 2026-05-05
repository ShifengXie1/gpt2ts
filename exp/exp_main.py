import os
import time
from datetime import datetime

import numpy as np
import torch
from torch import optim

from data_provider.data_factory import data_provider
from models import gpt2ts
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate


class TokenLLM_Main:
    # 初始化实验入口，创建设备、模型和结果目录。
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.results_dir = args.results_dir if args.results_dir else "./results"
        os.makedirs(self.results_dir, exist_ok=True)

        self.min_test_loss = float("inf")
        self.min_test_mae = float("inf")
        self.epoch_for_min_test_loss = -1

    # 根据运行参数选择当前实验使用的计算设备。
    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
            return torch.device(f"cuda:{self.args.gpu}")
        return torch.device("cpu")

    # 构建 GPT2TS 模型实例。
    def _build_model(self):
        return gpt2ts.Model(self.args).float()

    # 根据数据划分标记加载数据集和 DataLoader。
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    # 选择优化器并只优化可训练参数。
    def _select_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return optim.Adam(trainable_params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

    # 根据参数选择训练损失函数。
    def _select_criterion(self):
        criterions = {"mse": torch.nn.MSELoss(), "smoothL1": torch.nn.SmoothL1Loss()}
        loss_name = getattr(self.args, "loss", "mse")
        try:
            return criterions[loss_name]
        except KeyError as exc:
            raise ValueError(f"Invalid loss: {loss_name}") from exc

    # 构建并返回当前实验的模型检查点目录。
    def _checkpoint_dir(self, setting):
        path = os.path.join(self.results_dir, setting, "checkpoints")
        os.makedirs(path, exist_ok=True)
        return path

    # 构建并返回当前实验的结果保存目录。
    def _result_dir(self, setting):
        path = os.path.join(self.results_dir, setting)
        os.makedirs(path, exist_ok=True)
        return path

    # 将测试指标写入文本结果文件。
    def _save_test_results(self, setting, metrics):
        result_path = os.path.join(self.results_dir, "result.txt")
        with open(result_path, "w", encoding="utf-8") as file:
            file.write(f"saved_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"setting: {setting}\n")
            file.write(f"features: {self.args.features}\n")
            file.write(f"target_col: {self.args.target_col}\n")
            file.write(
                "test_loss={test_loss:.6f} | mse={mse:.6f}, mae={mae:.6f}, "
                "rmse={rmse:.6f}, mape={mape:.6f}, mspe={mspe:.6f}\n".format(**metrics)
            )

    # 从 DataLoader 批数据中取出输入序列和目标序列。
    def _unpack_batch(self, batch):
        if len(batch) < 2:
            raise ValueError("Expected a batch containing at least input and target tensors.")
        return batch[0], batch[1]

    # 从目标序列中截取预测长度对应的监督窗口。
    def _target_window(self, target):
        return target[:, -self.args.pred_len :, : self.args.c_out]

    # 执行模型前向传播并兼容 AMP 混合精度。
    def _forward_model(self, batch_x, target=None):
        amp_enabled = bool(self.args.use_amp and self.device.type == "cuda")
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output = self.model(batch_x, target)
        if hasattr(output, "pred"):
            return output.pred
        return output

    # 处理一个 batch，完成设备迁移、前向传播和真实值切片。
    def _process_one_batch(self, batch):
        batch_x, target = self._unpack_batch(batch)
        batch_x = batch_x.to(dtype=torch.float, device=self.device)
        target = target.to(dtype=torch.float, device=self.device)
        true = self._target_window(target)
        pred = self._forward_model(batch_x, target)
        return pred, true

    # 在训练前基于训练集拟合时序聚类和词表聚类。
    def _fit_clusters(self, train_loader):
        if hasattr(self.model, "fit_token_clusters"):
            print("\tFitting time/vocab embedding clusters ...")
            self.model.fit_token_clusters(train_loader, device=self.device)

    # 在验证集或测试集上评估模型并返回 MSE 和 MAE。
    def vali(self, vali_data, vali_loader, criterion):
        was_training = self.model.training
        self.model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for batch in vali_loader:
                pred, true = self._process_one_batch(batch)
                preds.append(pred.detach())
                trues.append(true.detach())

        if was_training:
            self.model.train()

        preds = torch.cat(preds, dim=0).cpu()
        trues = torch.cat(trues, dim=0).cpu()
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds.numpy(), trues.numpy())
        return mse, mae

    # 执行完整训练流程，包括聚类拟合、训练循环、验证和早停。
    def train(self, setting, optunaTrialReport=None):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        checkpoint_dir = self._checkpoint_dir(setting)
        self._fit_clusters(train_loader)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        amp_enabled = bool(self.args.use_amp and self.device.type == "cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for batch in train_loader:
                model_optim.zero_grad(set_to_none=True)
                pred, true = self._process_one_batch(batch)
                loss = criterion(pred, true)

                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                train_loss.append(loss.detach().item())

            train_loss = float(np.mean(train_loss)) if train_loss else float("nan")
            vali_loss, vali_mae = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mae = self.vali(test_data, test_loader, criterion)

            if test_loss < self.min_test_loss:
                self.min_test_loss = test_loss
                self.min_test_mae = test_mae
                self.epoch_for_min_test_loss = epoch

            print("Epoch {}: cost time: {:.2f} sec".format(epoch + 1, time.time() - epoch_time))
            print(
                "\tEpoch {0}: Steps- {1} | Train Loss: {2:.5f} "
                "Vali.MSE: {3:.5f} Vali.MAE: {4:.5f} Test.MSE: {5:.5f} Test.MAE: {6:.5f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, vali_mae, test_loss, test_mae
                )
            )

            early_stopping(vali_loss, self.model, checkpoint_dir)
            if early_stopping.early_stop:
                print("\tEarly stopping")
                break
            if np.isnan(train_loss):
                print("\tStopping: train-loss-nan")
                break

            adjust_learning_rate(model_optim, None, epoch + 1, self.args)

            if optunaTrialReport is not None:
                optunaTrialReport.report(vali_loss, epoch)
                if optunaTrialReport.should_prune():
                    break

        best_model_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    # 在测试集上推理、计算指标并保存预测结果。
    def test(self, setting):
        test_data, test_loader = self._get_data(flag="test")
        criterion = self._select_criterion()
        self.model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for batch in test_loader:
                pred, true = self._process_one_batch(batch)
                preds.append(pred.detach())
                trues.append(true.detach())

        preds = torch.cat(preds, dim=0).cpu()
        trues = torch.cat(trues, dim=0).cpu()
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds.numpy(), trues.numpy())
        test_loss = criterion(preds, trues).item()
        print("mse: {}, mae: {}".format(mse, mae))

        result_dir = self._result_dir(setting)
        np.save(os.path.join(result_dir, "metrics.npy"), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(result_dir, "pred.npy"), preds.numpy())
        np.save(os.path.join(result_dir, "true.npy"), trues.numpy())

        metrics = {
            "test_loss": test_loss,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "mspe": mspe,
        }
        self._save_test_results(setting, metrics)
        return mse, mae
