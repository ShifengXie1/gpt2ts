import os
import time
from datetime import datetime

import numpy as np
import torch
from torch import optim

from utils.tools import EarlyStopping, adjust_learning_rate
from data_provider.data_factory import data_provider
from models import gpt2ts
from utils.metrics import metric

class TokenLLM_Main:
    def __init__(self, args):
        self.args = args
        self.device = args.gpu if args.use_gpu else torch.device("cpu")
        self.model = self._build_model().to(self.device)
        self.results_dir = args.results_dir if args.results_dir else "./results"

    def _build_model(self):
        return gpt2ts.Model(self.args).float()

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = {'mse': torch.nn.MSELoss(), 'smoothL1': torch.nn.SmoothL1Loss()}
        try:
            return criterion[self.args.loss]
        except KeyError as e:
            raise ValueError(f"Invalid argument: {e} (loss: {self.args.loss})")
    

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

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()        
        preds_mean, trues = [], []

        with torch.no_grad():
            for batch_x, batch_y in vali_loader:
                pred_mean, true = self._process_one_batch(vali_data, batch_x, batch_y, 'vali')
                
                preds_mean.append(pred_mean)
                trues.append(true)

            preds_mean = torch.cat(preds_mean).cpu()
            trues = torch.cat(trues).cpu()
            
            preds_mean = preds_mean.reshape(-1, preds_mean.shape[-2], preds_mean.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            
            mae, mse, rmse, mape, mspe = metric(preds_mean.numpy(), trues.numpy())
            self.model.train()
            return mse, mae
        
    def train(self, setting, optunaTrialReport = None):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        time_now = time.time()        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion() 

        if self.args.use_amp:
            scaler =  torch.amp.GradScaler(init_scale = 1024)
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train() # 将模型设置为训练模式
            epoch_time = time.time()
            
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(train_loader, desc = f'Epoch {epoch + 1}', bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt}")):
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad(set_to_none = True)
                pred_mean, true = self._process_one_batch(train_data, batch_x, batch_y, 'train')
                loss = criterion(pred_mean, true)                
                train_loss.append(loss) #.item())
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch {}: cost time: {:.2f} sec".format(epoch + 1, time.time()-epoch_time))
            train_loss = torch.tensor(train_loss).mean() # np.average(train_loss)
            vali_loss, vali_mae = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mae = self.vali(test_data, test_loader, criterion)

            if test_loss <  self.min_test_loss:
                self.min_test_loss = test_loss
                self.min_test_mae = test_mae
                self.epoch_for_min_test_loss = epoch            
            
            print("\tEpoch {0}: Steps- {1} | Train Loss: {2:.5f} Vali.MSE: {3:.5f} Vali.MAE: {4:.5f} Test.MSE: {5:.5f} Test.MAE: {6:.5f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, vali_mae, test_loss, test_mae))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("\tEarly stopping")
                break
            if torch.isnan(train_loss):
                print("\stopping: train-loss-nan")
                break
            adjust_learning_rate(model_optim, None, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model


    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        criterion =  self._select_criterion() 
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(test_data, batch_x, batch_y,  'test')
                preds.append(pred)
                trues.append(true)

            preds = torch.cat(preds).cpu()
            trues = torch.cat(trues).cpu()
            # result save   
            folder_path = './results/' + setting +'/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
            mae, mse, rmse, mape, mspe = metric(preds.numpy(), trues.numpy())
            print('mse: {}, mae: {}'.format(mse, mae))

            f = open("result_long_term_forecast.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae) + "  \n")

            gflops,params = self.get_gflops()
            f.write(' gflops:{},gparams:{}'.format(gflops,params) + "  \n")


            f.write('datetime:{}'.format(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S  %A')) + "  \n")

            f.write('\n')
            f.write('\n')
            f.close()

            np.save(folder_path + os.sep + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + os.sep + 'pred.npy', preds)
            np.save(folder_path + os.sep + 'true.npy', trues)


            return mse, mae
        
    def _process_one_batch(self, dataset_object, batch_x, target, function):
        batch_x = batch_x.to(dtype = torch.float, device = self.device)
        target =  target.to(dtype = torch.float, device = self.device)
        
        if self.args.use_amp:
            with torch.amp.autocast():
                pred = self.model(batch_x) # 前向传播
        else:
            pred = self.model(batch_x) 
        return pred, target
