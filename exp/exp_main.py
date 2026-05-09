import os
import time
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

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
    def _process_one_batch(self, batch, return_output=False):
        if len(batch) < 2:
            raise ValueError("Expected a batch containing at least input and target tensors.")
        batch_x, target = batch[0], batch[1]
        
        batch_x = batch_x.to(dtype=torch.float, device=self.device)
        target = target.to(dtype=torch.float, device=self.device)
        model_target = target if return_output else None
        
        if self.amp_enabled:
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                output = self.model(batch_x, model_target)
        else:
            output = self.model(batch_x, model_target) 
        pred = output.pred if hasattr(output, "pred") else output
        pred, target = self._align_prediction_and_target(pred, target)
        if return_output:
            return pred, target, output
        return pred, target

    # Evaluate a split.
    def vali(self, vali_loader):
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

    def _save_model_checkpoint(self, checkpoint_path):
        trainable_names = {name for name, param in self.model.named_parameters() if param.requires_grad}
        full_state = self.model.state_dict()
        trainable_state = {
            name: tensor.detach().cpu()
            for name, tensor in full_state.items()
            if name in trainable_names
        }
        checkpoint = {
            "model_state_dict": trainable_state,
            "model_config": vars(self.args).copy(),
            "trainable_param_names": sorted(trainable_names),
        }
        torch.save(checkpoint, checkpoint_path)

    def _load_model_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict, strict=False)

    def _train_patch_token_lm(self, checkpoint_dir, vali_loader, test_loader):
        if not hasattr(self.model, "build_lm_training_tensors"):
            return False

        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            print("\tSkipping token LM training: no trainable parameters. Set --lora_r > 0 to enable LoRA.")
            self._save_model_checkpoint(checkpoint_path)
            return False
        if int(self.args.train_epochs) <= 0:
            print("\tSkipping token LM training: train_epochs <= 0.")
            self._save_model_checkpoint(checkpoint_path)
            return False

        input_ids, labels = self.model.build_lm_training_tensors()
        dataset = TensorDataset(input_ids.detach().cpu(), labels.detach().cpu())
        token_batch_size = int(getattr(self.args, "token_batch_size", self.args.batch_size) or self.args.batch_size)
        token_loader = DataLoader(
            dataset,
            batch_size=max(token_batch_size, 1),
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=False,
        )

        model_optim = optim.Adam(trainable_params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        scaler = torch.amp.GradScaler(enabled=self.amp_enabled)
        best_vali_loss = float("inf")
        patience_counter = 0

        print(
            "\tToken LM training windows: {0} | input tokens: {1} | future labels/window: {2}".format(
                len(dataset),
                input_ids.shape[1],
                self.model.future_patch_count,
            )
        )

        for epoch in range(self.args.train_epochs):
            token_losses = []
            self.model.train()
            epoch_time = time.time()

            for batch_input_ids, batch_labels in token_loader:
                batch_input_ids = batch_input_ids.to(device=self.device, dtype=torch.long)
                batch_labels = batch_labels.to(device=self.device, dtype=torch.long)

                model_optim.zero_grad(set_to_none=True)
                if self.amp_enabled:
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                        loss = self.model.token_lm_loss(batch_input_ids, batch_labels)
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss = self.model.token_lm_loss(batch_input_ids, batch_labels)
                    loss.backward()
                    model_optim.step()

                token_losses.append(loss.detach().item())

            train_loss = float(np.mean(token_losses)) if token_losses else float("nan")
            vali_loss, vali_mae = self.vali(vali_loader)
            test_loss, test_mae = self.vali(test_loader)

            if test_loss < self.min_test_loss:
                self.min_test_loss = test_loss
                self.min_test_mae = test_mae
                self.epoch_for_min_test_loss = epoch

            print("Token Epoch {}: cost time: {:.2f} sec".format(epoch + 1, time.time() - epoch_time))
            print(
                "\tEpoch {0}: Steps- {1} | Token Loss: {2:.5f} Vali.MSE: {3:.5f} Vali.MAE: {4:.5f} Test.MSE: {5:.5f} Test.MAE: {6:.5f}".format(
                    epoch + 1, len(token_loader), train_loss, vali_loss, vali_mae, test_loss, test_mae
                )
            )

            if vali_loss < best_vali_loss:
                print(f"\tValidation loss decreased ({best_vali_loss:.6f} --> {vali_loss:.6f}).  Saving model ...")
                best_vali_loss = vali_loss
                patience_counter = 0
                self._save_model_checkpoint(checkpoint_path)
            else:
                patience_counter += 1
                print(f"\tEarlyStopping counter: {patience_counter} out of {self.args.patience}")
                if patience_counter >= self.args.patience:
                    print("\tEarly stopping")
                    break
            if np.isnan(train_loss):
                print("\tStopping: token-loss-nan")
                break

            adjust_learning_rate(model_optim, None, epoch + 1, self.args)

        if os.path.exists(checkpoint_path):
            self._load_model_checkpoint(checkpoint_path)
        return True

    # Run training and early stopping.
    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        _, vali_loader = self._get_data(flag="val")
        _, test_loader = self._get_data(flag="test")

        checkpoint_dir = os.path.join(self.results_dir, setting, self.timestamp, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        if hasattr(self.model, "fit_patch_token_map"):
            if not hasattr(train_data, "data_x"):
                raise ValueError("The patch-token GPT flow requires a dataset with full training data in data_x.")

            fit_start = time.time()
            self.model.eval()
            self.model.fit_patch_token_map(train_data.data_x)

            save_dir = os.path.join(self.results_dir, setting, self.timestamp)
            os.makedirs(save_dir, exist_ok=True)
            self.model.save_patch_token_map(os.path.join(save_dir, "patch_token_map.npz"))
            dropped_points = int(getattr(self.model.dictionary, "last_dropped_points", 0))
            if dropped_points > 0:
                print(f"\tPatchify drop_last: dropped {dropped_points} trailing training point(s).")
            trained_lm = self._train_patch_token_lm(checkpoint_dir, vali_loader, test_loader)
            if not trained_lm:
                self._save_model_checkpoint(os.path.join(checkpoint_dir, "checkpoint.pth"))

            vali_loss, vali_mae = self.vali(vali_loader)
            test_loss, test_mae = self.vali(test_loader)
            if test_loss < self.min_test_loss:
                self.min_test_loss = test_loss
                self.min_test_mae = test_mae
                if self.epoch_for_min_test_loss < 0:
                    self.epoch_for_min_test_loss = 0
            print("Patch-token map fitted and token LM stage finished in {:.2f} sec".format(time.time() - fit_start))
            print(
                "\tPatch-token | Train patches: {0} | Vali.MSE: {1:.5f} Vali.MAE: {2:.5f} Test.MSE: {3:.5f} Test.MAE: {4:.5f}".format(
                    self.model.dictionary.train_patches.shape[0],
                    vali_loss,
                    vali_mae,
                    test_loss,
                    test_mae,
                )
            )
            return self.model

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
                pred, true, output = self._process_one_batch(batch, return_output=True)
                main_loss = criterion(pred, true)
                model_loss = output.loss if hasattr(output, "loss") else None
                loss = main_loss if model_loss is None else main_loss + model_loss

                if self.amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                train_loss.append(loss.detach().item())

            train_loss = float(np.mean(train_loss)) if train_loss else float("nan")
            vali_loss, vali_mae = self.vali(vali_loader)
            test_loss, test_mae = self.vali(test_loader)

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
        self._load_model_checkpoint(best_model_path)
        return self.model

    # Test and save results.
    def _inverse_transform_array(self, data_set, array):
        if not hasattr(data_set, "inverse_transform"):
            return None
        original_shape = array.shape
        flat = array.reshape(-1, original_shape[-1])
        try:
            restored = data_set.inverse_transform(flat)
        except Exception:
            return None
        return restored.reshape(original_shape)

    def _save_batch_curve_view(self, pred_batch, true_batch, save_dir, max_samples=8, max_channels=8, view_name="batch_view"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if torch.is_tensor(pred_batch):
            pred_batch = pred_batch.detach().cpu().numpy()
        if torch.is_tensor(true_batch):
            true_batch = true_batch.detach().cpu().numpy()
        sample_count = min(pred_batch.shape[0], max_samples)
        channel_count = min(pred_batch.shape[-1], max_channels)
        horizon = np.arange(pred_batch.shape[1])

        view_dir = os.path.join(save_dir, view_name)
        os.makedirs(view_dir, exist_ok=True)
        np.save(os.path.join(view_dir, "batch_pred.npy"), pred_batch)
        np.save(os.path.join(view_dir, "batch_true.npy"), true_batch)

        fig, axes = plt.subplots(
            sample_count,
            channel_count,
            figsize=(3.2 * channel_count, 2.2 * sample_count),
            squeeze=False,
        )
        for sample_idx in range(sample_count):
            for channel_idx in range(channel_count):
                ax = axes[sample_idx][channel_idx]
                ax.plot(horizon, true_batch[sample_idx, :, channel_idx], label="true", linewidth=1.2)
                ax.plot(horizon, pred_batch[sample_idx, :, channel_idx], label="pred", linewidth=1.2)
                ax.set_title(f"sample {sample_idx} | channel {channel_idx}", fontsize=8)
                ax.tick_params(axis="both", labelsize=7)
                if sample_idx == 0 and channel_idx == 0:
                    ax.legend(fontsize=7)

        fig.suptitle("Test batch prediction vs true", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(view_dir, "batch_prediction_vs_true.png"), dpi=160)
        plt.close(fig)

    def test(self, setting):
        test_data, test_loader = self._get_data(flag="test")
        criterion = self._select_criterion()
        self.model.eval()
        preds, trues = [], []
        view_pred, view_true = None, None

        with torch.no_grad():
            for batch in test_loader:
                pred, true = self._process_one_batch(batch)
                if view_pred is None:
                    view_pred = pred.detach()
                    view_true = true.detach()
                preds.append(pred.detach())
                trues.append(true.detach())

        preds = torch.cat(preds, dim=0).cpu()
        trues = torch.cat(trues, dim=0).cpu()
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds.numpy(), trues.numpy())
        test_loss = criterion(preds, trues).item()
        print("standardized mse: {}, mae: {}".format(mse, mae))

        raw_metrics = None
        preds_raw = self._inverse_transform_array(test_data, preds.numpy())
        trues_raw = self._inverse_transform_array(test_data, trues.numpy())
        if preds_raw is not None and trues_raw is not None:
            raw_mae, raw_mse, raw_rmse, raw_mape, raw_mspe = metric(preds_raw, trues_raw)
            raw_metrics = {
                "raw_mse": raw_mse,
                "raw_mae": raw_mae,
                "raw_rmse": raw_rmse,
                "raw_mape": raw_mape,
                "raw_mspe": raw_mspe,
            }
            print("raw-scale mse: {}, mae: {}".format(raw_mse, raw_mae))

        save_dir = os.path.join(self.results_dir, setting, self.timestamp)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "metrics.npy"), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(save_dir, "pred.npy"), preds.numpy())
        np.save(os.path.join(save_dir, "true.npy"), trues.numpy())
        view_pred_raw, view_true_raw = None, None
        if raw_metrics is not None:
            np.save(
                os.path.join(save_dir, "metrics_raw.npy"),
                np.array([
                    raw_metrics["raw_mae"],
                    raw_metrics["raw_mse"],
                    raw_metrics["raw_rmse"],
                    raw_metrics["raw_mape"],
                    raw_metrics["raw_mspe"],
                ]),
            )
            np.save(os.path.join(save_dir, "pred_raw.npy"), preds_raw)
            np.save(os.path.join(save_dir, "true_raw.npy"), trues_raw)
            if view_pred is not None and view_true is not None:
                view_pred_raw = self._inverse_transform_array(test_data, view_pred.detach().cpu().numpy())
                view_true_raw = self._inverse_transform_array(test_data, view_true.detach().cpu().numpy())
        if view_pred is not None and view_true is not None:
            self._save_batch_curve_view(view_pred, view_true, save_dir)
        if view_pred_raw is not None and view_true_raw is not None:
            self._save_batch_curve_view(view_pred_raw, view_true_raw, save_dir, view_name="batch_view_raw")

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
            file.write(
                "test_loss={test_loss:.6f} | mse={mse:.6f}, mae={mae:.6f}, "
                "rmse={rmse:.6f}, mape={mape:.6f}, mspe={mspe:.6f}\n".format(**metrics)
            )
            if raw_metrics is not None:
                file.write(
                    "raw_mse={raw_mse:.6f}, raw_mae={raw_mae:.6f}, "
                    "raw_rmse={raw_rmse:.6f}, raw_mape={raw_mape:.6f}, raw_mspe={raw_mspe:.6f}\n".format(
                        **raw_metrics
                    )
                )
            file.write(f"\n")
            
        return mse, mae
