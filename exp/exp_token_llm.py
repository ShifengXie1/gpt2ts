import os
import time
from datetime import datetime

import numpy as np
import torch
from torch import optim

from data_provider.data_factory import data_provider
from models import gpt2ts
from utils.metrics import metric


def build_setting(args):
    return (
        f"{args.task_name}_{args.data}_{args.model}_"
        f"sl{args.seq_len}_pl{args.pred_len}_ps{args.patch_len}"
    )


class TokenLLM_Main:
    def __init__(self, args):
        self.args = args
        self.result_dir = None
        self.best_checkpoint_path = None
        self.device = self._select_device()
        self.model = self._build_model().to(self.device)

    def _select_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: {device}")
            return device
        print("Use CPU")
        return torch.device("cpu")

    def _build_model(self):
        if self.args.model != "gpt2ts":
            raise ValueError(f"Unsupported model `{self.args.model}`. Use `--model gpt2ts`.")
        return gpt2ts.Model(self.args).float()

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _ensure_result_dir(self, setting):
        if self.result_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{setting}_features{self.args.features}_{timestamp}"
            self.result_dir = os.path.join("results", run_name)
            os.makedirs(self.result_dir, exist_ok=True)
            print(f"Run artifacts will be saved to: {self.result_dir}")
        return self.result_dir

    def _save_training_log(self, rows):
        if not rows or self.result_dir is None:
            return

        path = os.path.join(self.result_dir, "training_log.csv")
        with open(path, "w", encoding="utf-8") as file:
            file.write("epoch,train_loss,val_loss,time_seconds\n")
            for row in rows:
                file.write(
                    f"{row['epoch']},{row['train_loss']:.10f},"
                    f"{row['val_loss']:.10f},{row['time_seconds']:.4f}\n"
                )

    def _save_test_results(self, setting, metrics, checkpoint_path):
        save_dir = self._ensure_result_dir(setting)
        result_path = os.path.join(save_dir, "result.txt")
        json_path = os.path.join(save_dir, "result.json")

        summary = {
            "setting": setting,
            "data": self.args.data,
            "features": self.args.features,
            "target_col": self.args.target_col,
            "seq_len": self.args.seq_len,
            "pred_len": self.args.pred_len,
            "patch_len": self.args.patch_len,
            "c_in": self.args.c_in,
            "c_out": self.args.c_out,
            "checkpoint": checkpoint_path,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **metrics,
        }

        with open(result_path, "w", encoding="utf-8") as file:
            file.write(f"setting: {setting}\n")
            file.write(f"features: {self.args.features}\n")
            file.write(f"target_col: {self.args.target_col}\n")
            file.write(f"checkpoint: {checkpoint_path}\n")
            file.write(
                "test_loss={test_loss:.6f} | mse={mse:.6f}, mae={mae:.6f}, "
                "rmse={rmse:.6f}, mape={mape:.6f}, mspe={mspe:.6f}\n".format(**metrics)
            )

        import json

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

    def _select_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def _run_epoch(self, loader, optimizer=None):
        is_train = optimizer is not None
        self.model.train(is_train)
        losses = []
        preds, trues = [], []

        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(is_train):
                outputs = self.model(batch_x, batch_y)
                loss = outputs.loss
                if is_train:
                    loss.backward()
                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()

            losses.append(loss.item())
            if not is_train:
                preds.append(outputs.pred.detach().cpu().numpy())
                trues.append(batch_y[:, -self.args.pred_len :, : self.args.c_out].detach().cpu().numpy())

        avg_loss = float(np.mean(losses)) if losses else 0.0
        if is_train:
            return avg_loss, None, None
        return avg_loss, preds, trues

    def _target_channel_index(self):
        if self.args.c_out == 1:
            return 0, self.args.target_col

        csv_path = os.path.join(self.args.root_path, self.args.data_path)
        try:
            with open(csv_path, "r", encoding="utf-8") as file:
                header = file.readline().strip().split(",")
            columns = header[1:] if header and header[0].lower() == "date" else header
            return columns.index(self.args.target_col), self.args.target_col
        except (OSError, ValueError):
            return self.args.c_out - 1, self.args.target_col

    def _save_prediction_plots(self, setting, preds, trues, max_samples=4):
        if preds.size == 0 or trues.size == 0:
            return None

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_dir = self._ensure_result_dir(setting)

        sample_count = min(max_samples, preds.shape[0])
        target_idx, target_name = self._target_channel_index()
        target_idx = min(max(target_idx, 0), preds.shape[-1] - 1)
        horizon = np.arange(preds.shape[1])

        for sample_idx in range(sample_count):
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(horizon, trues[sample_idx, :, target_idx], label="True", linewidth=1.8)
            ax.plot(horizon, preds[sample_idx, :, target_idx], label="Pred", linewidth=1.8)
            ax.set_title(f"Sample {sample_idx} | {target_name}")
            ax.set_xlabel("Prediction step")
            ax.set_ylabel("Scaled value")
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f"prediction_sample_{sample_idx}.png"), dpi=150)
            plt.close(fig)

        return save_dir

    def train(self, setting):
        path = self._ensure_result_dir(setting)
        train_data, train_loader = self._get_data("train")
        vali_data, vali_loader = self._get_data("val")

        optimizer = self._select_optimizer()
        best_loss = float("inf")
        patience_count = 0
        best_path = os.path.join(path, "checkpoint.pth")
        self.best_checkpoint_path = best_path
        training_log = []

        for epoch in range(self.args.train_epochs):
            start = time.time()
            train_loss, _, _ = self._run_epoch(train_loader, optimizer)
            vali_loss, _, _ = self._run_epoch(vali_loader)
            elapsed = time.time() - start
            print(
                f"Epoch {epoch + 1}/{self.args.train_epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={vali_loss:.6f} | "
                f"time={elapsed:.1f}s"
            )
            training_log.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": vali_loss,
                    "time_seconds": elapsed,
                }
            )

            if vali_loss < best_loss:
                best_loss = vali_loss
                patience_count = 0
                torch.save(self.model.state_dict(), best_path)
            else:
                patience_count += 1
                if patience_count >= self.args.patience:
                    print("Early stopping")
                    break

        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            print(f"Best checkpoint saved to: {best_path}")
        self._save_training_log(training_log)
        return self.model

    @torch.no_grad()
    def test(self, setting, checkpoint_path=None, load_checkpoint=True):
        loaded_checkpoint = checkpoint_path
        if load_checkpoint:
            candidate_paths = []
            if checkpoint_path:
                candidate_paths.append(checkpoint_path)
            if self.best_checkpoint_path:
                candidate_paths.append(self.best_checkpoint_path)

            loaded_checkpoint = None
            for candidate_path in candidate_paths:
                if candidate_path and os.path.exists(candidate_path):
                    loaded_checkpoint = candidate_path
                    break

            if loaded_checkpoint:
                print(f"Loading checkpoint: {loaded_checkpoint}")
                self.model.load_state_dict(torch.load(loaded_checkpoint, map_location=self.device))
            else:
                missing_path = checkpoint_path or self.best_checkpoint_path
                print(f"Checkpoint not found, evaluating current weights: {missing_path}")

        test_data, test_loader = self._get_data("test")
        test_loss, preds, trues = self._run_epoch(test_loader)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        metrics = {
            "test_loss": float(test_loss),
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "mspe": float(mspe),
        }
        print(
            f"Test loss={test_loss:.6f} | "
            f"mse={mse:.6f}, mae={mae:.6f}, rmse={rmse:.6f}, mape={mape:.6f}, mspe={mspe:.6f}"
        )
        plot_dir = self._save_prediction_plots(setting, preds, trues)
        self._save_test_results(setting, metrics, loaded_checkpoint)
        if plot_dir:
            print(f"Prediction plots saved to: {plot_dir}")
            print(f"Test results saved to: {plot_dir}")
        return mse, mae
