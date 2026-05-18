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
from utils.tools import adjust_learning_rate


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

    # Align prediction horizon and channels.
    def _align_prediction_and_target(self, pred, target):
        pred = pred[:, -self.args.pred_len :, :]
        target = target[:, -self.args.pred_len :, :]
        return pred, target

    # Process one batch.
    def _process_one_batch(self, batch, return_output=False):
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
        if return_output:
            return pred, target, output
        return pred, target

    def _evaluate_loader(self, data_loader, collect_tokens=False):
        self.model.eval()
        preds, trues = [], []
        token_ce_losses = []
        future_token_ids = []

        with torch.no_grad():
            for batch in data_loader:
                pred, true, output = self._process_one_batch(batch, return_output=True)
                preds.append(pred.detach())
                trues.append(true.detach())
                batch_x = batch[0].to(dtype=torch.float, device=self.device)
                target = batch[1].to(dtype=torch.float, device=self.device)
                if hasattr(self.model, "eval_token_ce"):
                    token_ce_losses.append(self.model.eval_token_ce(batch_x, target))
                if collect_tokens:
                    aux = output.aux if hasattr(output, "aux") else None
                    if aux is not None and hasattr(aux, "future_token_ids"):
                        future_token_ids.extend(aux.future_token_ids.detach().cpu().reshape(-1).long().tolist())

            preds = torch.cat(preds, dim=0).cpu()
            trues = torch.cat(trues, dim=0).cpu()
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

            mae, mse, rmse, mape, mspe = metric(preds.numpy(), trues.numpy())
            self.model.train()
            finite_token_ce = [value for value in token_ce_losses if not np.isnan(value)]
            token_ce = float(np.mean(finite_token_ce)) if finite_token_ce else float("nan")
            return {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "mspe": mspe,
                "token_ce": token_ce,
                "preds": preds,
                "trues": trues,
                "future_token_ids": future_token_ids,
            }

    # Evaluate a split.
    def vali(self, vali_loader):
        results = self._evaluate_loader(vali_loader)
        return results["mse"], results["mae"]

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
        if hasattr(self.model, "dictionary") and hasattr(self.model.dictionary, "state_payload"):
            checkpoint["patch_token_dictionary"] = self.model.dictionary.state_payload()
        torch.save(checkpoint, checkpoint_path)

    def _load_model_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        if isinstance(checkpoint, dict) and "patch_token_dictionary" in checkpoint:
            self.model.dictionary.load_state_payload(checkpoint["patch_token_dictionary"])

    def _token_log_path(self, save_dir):
        return os.path.join(save_dir, "token_log.txt")

    def _sorted_token_counts(self, token_ids):
        token_ids = token_ids.detach().cpu().reshape(-1).long()
        if token_ids.numel() == 0:
            return []
        unique_ids, counts = torch.unique(token_ids, return_counts=True)
        pairs = [
            (int(token_id.item()), int(count.item()))
            for token_id, count in zip(unique_ids, counts)
        ]
        return sorted(pairs, key=lambda item: (-item[1], item[0]))

    def _write_train_token_log(self, save_dir):
        if not hasattr(self.model, "dictionary") or not self.model.dictionary.ready:
            return

        dictionary = self.model.dictionary
        token_counts = self._sorted_token_counts(dictionary.train_patch_token_ids)
        allowed_token_count = len(token_counts)
        random_ce = float(np.log(allowed_token_count)) if allowed_token_count > 0 else float("nan")

        with open(self._token_log_path(save_dir), "w", encoding="utf-8") as file:
            file.write("[TRAIN]\n")
            file.write(f"total_train_patches: {int(dictionary.train_patches.shape[0])}\n")
            file.write(f"unique_train_tokens: {allowed_token_count}\n")
            file.write(f"allowed_token_count: {allowed_token_count}\n")
            if hasattr(dictionary, "valid_token_ids"):
                file.write(f"valid_token_ids: {int(dictionary.valid_token_ids.numel())}\n")
            if hasattr(dictionary, "candidate_token_ids"):
                file.write(f"candidate_token_ids: {int(dictionary.candidate_token_ids.numel())}\n")
            if hasattr(dictionary, "motif_token_ids"):
                file.write(f"motif_token_ids: {int(dictionary.motif_token_ids.numel())}\n")
            if hasattr(dictionary, "patch_bank"):
                file.write(f"patch_bank_shape: {tuple(dictionary.patch_bank.shape)}\n")
            if hasattr(dictionary, "last_assignment_method"):
                file.write(f"assignment_method: {dictionary.last_assignment_method}\n")
            file.write(f"random_ce_baseline_log_allowed: {random_ce:.4f}\n")
            file.write(f"patch_len: {int(self.model.patch_len)}\n")
            if hasattr(self.model, "stride"):
                file.write(f"stride: {int(self.model.stride)}\n")
            file.write(f"history_patch_count: {int(self.model.history_patch_count)}\n")
            if hasattr(self.model, "boundary_patch_count"):
                file.write(f"boundary_patch_count: {int(self.model.boundary_patch_count)}\n")
            file.write(f"future_patch_count: {int(self.model.future_patch_count)}\n")
            if hasattr(self.model, "generated_patch_count"):
                file.write(f"generated_patch_count: {int(self.model.generated_patch_count)}\n")
            file.write(f"cluster_num: {int(dictionary.cluster_num)}\n")

    def _write_test_token_log(self, save_dir, future_token_ids):
        if not future_token_ids:
            return

        total_counts = self._sorted_token_counts(torch.tensor(future_token_ids, dtype=torch.long))
        total_generated_count = int(sum(count for _, count in total_counts))
        generated_unique_tokens = len(total_counts)

        with open(self._token_log_path(save_dir), "a", encoding="utf-8") as file:
            file.write("\n[TEST]\n")
            file.write(f"generated_total_count: {total_generated_count}\n")
            file.write(f"generated_unique_tokens: {generated_unique_tokens}\n")
            file.write("total_generated_token:\n")
            file.write("token_id,count\n")
            for token_id, count in total_counts:
                file.write(f"{token_id},{count}\n")

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

        training_tensors = self.model.build_lm_training_tensors()
        dataset = TensorDataset(
            training_tensors.input_ids.detach().cpu(),
            training_tensors.labels.detach().cpu(),
            training_tensors.future_patches.detach().cpu(),
            training_tensors.align_patches.detach().cpu(),
            training_tensors.align_candidate_indices.detach().cpu(),
        )
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
        best_vali_mse = float("inf")
        patience_counter = 0

        print(
                "\tToken LM training windows: {0} | input tokens: {1} | future labels/window: {2}".format(
                    len(dataset),
                    training_tensors.input_ids.shape[1],
                    getattr(self.model, "generated_patch_count", self.model.future_patch_count),
                )
            )

        for epoch in range(self.args.train_epochs):
            loss_sums = {
                "total_loss": 0.0,
                "ce_loss": 0.0,
                "mse_loss": 0.0,
                "align_loss": 0.0,
                "smooth_loss": 0.0,
                "patch_mse": 0.0,
                "sequence_mse": 0.0,
            }
            sample_count = 0
            self.model.train()
            epoch_time = time.time()

            for batch_input_ids, batch_labels, batch_future_patches, batch_align_patches, batch_align_indices in token_loader:
                batch_input_ids = batch_input_ids.to(device=self.device, dtype=torch.long)
                batch_labels = batch_labels.to(device=self.device, dtype=torch.long)
                batch_future_patches = batch_future_patches.to(device=self.device, dtype=torch.float)
                batch_align_patches = batch_align_patches.to(device=self.device, dtype=torch.float)
                batch_align_indices = batch_align_indices.to(device=self.device, dtype=torch.long)

                model_optim.zero_grad(set_to_none=True)
                if self.amp_enabled:
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                        losses = self.model.joint_training_loss(
                            batch_input_ids,
                            batch_labels,
                            batch_future_patches,
                            batch_align_patches,
                            batch_align_indices,
                        )
                    scaler.scale(losses.total_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    losses = self.model.joint_training_loss(
                        batch_input_ids,
                        batch_labels,
                        batch_future_patches,
                        batch_align_patches,
                        batch_align_indices,
                    )
                    losses.total_loss.backward()
                    model_optim.step()

                batch_size = int(batch_input_ids.shape[0])
                sample_count += batch_size
                loss_sums["total_loss"] += float(losses.total_loss.detach().cpu().item()) * batch_size
                loss_sums["ce_loss"] += float(losses.loss_ce.detach().cpu().item()) * batch_size
                loss_sums["mse_loss"] += float(losses.loss_mse.detach().cpu().item()) * batch_size
                loss_sums["align_loss"] += float(losses.loss_align.detach().cpu().item()) * batch_size
                loss_sums["smooth_loss"] += float(losses.loss_smooth.detach().cpu().item()) * batch_size
                loss_sums["patch_mse"] += float(losses.patch_mse.detach().cpu().item()) * batch_size
                loss_sums["sequence_mse"] += float(losses.sequence_mse.detach().cpu().item()) * batch_size

            train_metrics = {
                key: value / max(sample_count, 1)
                for key, value in loss_sums.items()
            }
            vali_metrics = self._evaluate_loader(vali_loader)
            test_metrics = self._evaluate_loader(test_loader)

            if test_metrics["mse"] < self.min_test_loss:
                self.min_test_loss = test_metrics["mse"]
                self.min_test_mae = test_metrics["mae"]
                self.epoch_for_min_test_loss = epoch

            print("Token Epoch {}: cost time: {:.2f} sec".format(epoch + 1, time.time() - epoch_time))
            print(
                "\tEpoch {0}: Steps- {1} | Total: {2:.5f} CE: {3:.5f} MSE: {4:.5f} Align: {5:.5f} Smooth: {6:.5f}".format(
                    epoch + 1,
                    len(token_loader),
                    train_metrics["total_loss"],
                    train_metrics["ce_loss"],
                    train_metrics["mse_loss"],
                    train_metrics["align_loss"],
                    train_metrics["smooth_loss"],
                )
            )
            print(
                "\tPatchMSE: {0:.5f} SeqMSE: {1:.5f} | Vali.CE: {2:.5f} Vali.MSE: {3:.5f} Vali.MAE: {4:.5f} Test.MSE: {5:.5f} Test.MAE: {6:.5f} Best.Vali.MSE: {7:.5f}".format(
                    train_metrics["patch_mse"],
                    train_metrics["sequence_mse"],
                    vali_metrics["token_ce"],
                    vali_metrics["mse"],
                    vali_metrics["mae"],
                    test_metrics["mse"],
                    test_metrics["mae"],
                    min(best_vali_mse, vali_metrics["mse"]),
                )
            )

            if vali_metrics["mse"] < best_vali_mse:
                print(f"\tVali.MSE decreased ({best_vali_mse:.6f} --> {vali_metrics['mse']:.6f}).  Saving model ...")
                best_vali_mse = vali_metrics["mse"]
                patience_counter = 0
                self._save_model_checkpoint(checkpoint_path)
            else:
                patience_counter += 1
                print(f"\tEarlyStopping counter: {patience_counter} out of {self.args.patience}")
                if patience_counter >= self.args.patience:
                    print("\tEarly stopping")
                    break
            if np.isnan(train_metrics["total_loss"]):
                print("\tStopping: total-loss-nan")
                break

            adjust_learning_rate(model_optim, None, epoch + 1, self.args)

        if os.path.exists(checkpoint_path):
            self._load_model_checkpoint(checkpoint_path)
        return True

    # Run training and early stopping.
    def train(self, setting):
        train_data, _ = self._get_data(flag="train")
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
            self._write_train_token_log(save_dir)
            if getattr(self.args, "debug_token_map", False) and hasattr(self.model, "debug_token_map"):
                for line in self.model.debug_token_map():
                    print(f"\t{line}")
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

    # Test and save results.
    def _save_batch_curve_view(
        self,
        pred_batch,
        true_batch,
        save_dir,
        max_samples=8,
        max_channels=8,
        view_name="batch_view",
        file_prefix="batch",
    ):
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
        fig.savefig(os.path.join(view_dir, f"{file_prefix}_prediction_vs_true.png"), dpi=160)
        plt.close(fig)

    def _save_batch_curve_views_from_files(self, save_dir, max_batches=100):
        pred_path = os.path.join(save_dir, "pred.npy")
        true_path = os.path.join(save_dir, "true.npy")
        pred_all = np.load(pred_path)
        true_all = np.load(true_path)
        batch_size = max(int(self.args.batch_size), 1)

        for batch_idx, start in enumerate(range(0, pred_all.shape[0], batch_size)):
            if batch_idx >= max_batches:
                break
            end = min(start + batch_size, pred_all.shape[0])
            self._save_batch_curve_view(
                pred_all[start:end],
                true_all[start:end],
                save_dir,
                view_name="batch_view",
                file_prefix=f"batch_{batch_idx:04d}",
            )

    def test(self, setting):
        _, test_loader = self._get_data(flag="test")
        save_dir = os.path.join(self.results_dir, setting, self.timestamp)
        os.makedirs(save_dir, exist_ok=True)

        results = self._evaluate_loader(test_loader, collect_tokens=True)
        preds = results["preds"]
        trues = results["trues"]
        mae = results["mae"]
        mse = results["mse"]
        rmse = results["rmse"]
        mape = results["mape"]
        mspe = results["mspe"]
        test_loss = mse
        print("standardized mse: {}, mae: {}".format(mse, mae))

        np.save(os.path.join(save_dir, "metrics.npy"), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(save_dir, "pred.npy"), preds.numpy())
        np.save(os.path.join(save_dir, "true.npy"), trues.numpy())
        self._save_batch_curve_views_from_files(save_dir)
        self._write_test_token_log(save_dir, results["future_token_ids"])

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
            file.write(f"\n")
            
        return mse, mae
