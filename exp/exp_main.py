import os
import time
from datetime import datetime

import numpy as np
import torch
from torch import optim

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from models import gpt2ts
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate


class Exp_Main(Exp_Basic):
    # Initialize experiment state.
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

        self.results_dir = args.results_dir if args.results_dir else "./results"
        os.makedirs(self.results_dir, exist_ok=True)

        self.min_test_loss = float("inf")
        self.min_test_mae = float("inf")
        self.epoch_for_min_test_loss = -1
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.amp_enabled = bool(self.args.use_amp and self.device.type == "cuda")

    # Build the GPT2TS model.
    def _build_model(self):
        return gpt2ts.GPT2TS(self.args).float()

    # Load a split and DataLoader.
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    # Optimize trainable params only.
    def _select_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return optim.Adam(trainable_params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

    # Select the training loss.
    def _select_criterion(self):
        criterions = {"mse": torch.nn.MSELoss(), "smoothL1": torch.nn.SmoothL1Loss()}
        try:
            return criterions[self.args.loss]
        except KeyError as exc:
            raise ValueError(f"Invalid loss: {self.args.loss}") from exc

    # Align prediction horizon and channels.
    def _align_prediction_and_target(self, pred, target):
        pred = pred[:, -self.args.pred_len :, :]
        target = target[:, -self.args.pred_len :, :]
        if self.args.features == "MS":
            c_out = int(getattr(self.args, "c_out", target.shape[-1]) or target.shape[-1])
            if c_out > 0:
                pred = pred[:, :, -c_out:]
                target = target[:, :, -c_out:]
        return pred, target

    # Process one batch.
    def _process_one_batch(self, batch):
        if len(batch) < 2:
            raise ValueError("Expected a batch containing at least input and target tensors.")
        batch_x, target = batch[0], batch[1]
        
        batch_x = batch_x.to(dtype=torch.float, device=self.device)
        target = target.to(dtype=torch.float, device=self.device)
        
        if self.amp_enabled:
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                output = self.model(batch_x)
        else:
            output = self.model(batch_x) 
        pred = output.pred if hasattr(output, "pred") else output
        pred, target = self._align_prediction_and_target(pred, target)
        return pred, target

    # Evaluate a split.
    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for batch in vali_loader:
                pred, true = self._process_one_batch(batch)
                preds.append(pred)
                trues.append(true)     

            preds = torch.cat(preds, dim=0).cpu()
            trues = torch.cat(trues, dim=0).cpu()
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

            mae, mse, rmse, mape, mspe = metric(preds.numpy(), trues.numpy())
            self.model.train()
            return mse, mae

    # Run training and early stopping.
    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        checkpoint_dir = os.path.join(self.results_dir, setting, self.timestamp, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        
        scaler = torch.amp.GradScaler(enabled=self.amp_enabled)

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for batch in train_loader:
                model_optim.zero_grad(set_to_none=True)
                pred, true = self._process_one_batch(batch)
                loss = criterion(pred, true)

                if self.amp_enabled:
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
                "\tEpoch {0}: Steps- {1} | Train Loss: {2:.5f} Vali.MSE: {3:.5f} Vali.MAE: {4:.5f} Test.MSE: {5:.5f} Test.MAE: {6:.5f}".format(
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

        best_model_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    # Test and save results.
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

        save_dir = os.path.join(self.results_dir, setting, self.timestamp)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "metrics.npy"), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(save_dir, "pred.npy"), preds.numpy())
        np.save(os.path.join(save_dir, "true.npy"), trues.numpy())

        metrics = {
            "test_loss": test_loss,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "mspe": mspe,
        }
        
        result_path = os.path.join(self.results_dir, "result.txt")
        with open(result_path, "a", encoding="utf-8") as file:
            file.write(f"saved_at: {self.timestamp}\n")
            file.write(f"setting: {setting}\n")
            file.write(f"features: {self.args.features}\n")
            file.write(f"target_col: {self.args.target_col}\n")
            file.write(
                "test_loss={test_loss:.6f} | mse={mse:.6f}, mae={mae:.6f}, "
                "rmse={rmse:.6f}, mape={mape:.6f}, mspe={mspe:.6f}\n".format(**metrics)
            )
            file.write(f"\n")
            
        return mse, mae
