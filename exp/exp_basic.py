import os

import numpy as np
import torch


class Exp_Basic(object):
    # 初始化基础实验对象、设备和模型。
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    # 构建模型，具体实现需要由子类重写。
    def _build_model(self):
        raise NotImplementedError
        return None

    # 根据参数选择 CPU 或 GPU 设备。
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    # 获取数据，具体实现预留给子类。
    def _get_data(self):
        pass

    # 执行验证流程，具体实现预留给子类。
    def vali(self):
        pass

    # 执行训练流程，具体实现预留给子类。
    def train(self):
        pass

    # 执行测试流程，具体实现预留给子类。
    def test(self):
        pass
