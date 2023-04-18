#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2022/11/12 上午11:32

import os
import numpy as np

class nyu_dataset(object):

    def __init__(self, data_path, t_size=36, t_pred_size=18, interval=None,
                 noise_level=6, noise_cycle=5, test_ratio=0.15, seed=1024, norm=False,
                 sigma_o=0.1, sigma_s=0.1, beta=50, logger=None, standard=False):

        super(nyu_dataset, self).__init__()

        self.data_path = data_path
        self.t_size = t_size
        self.interval = interval if interval is not None else self.t_size // 2
        self.noise_level = noise_level
        self.test_ratio = test_ratio
        self.seed = seed
        self.norm = norm
        self.t_pred_size = t_pred_size
        self.logger = logger
        self.sigma_o = sigma_o
        self.sigma_s = sigma_s
        self.standard = standard

        if not os.path.exists(self.data_path):
            assert os.path.exists(self.data_path) == False, f'{self.data_path} is invalid data path.'
        else:
            data = np.load(self.data_path)

            self.x_train = data['x_train']
            self.x_test = data['x_test']
            self.y_train_deno = data['y_train_deno']
            self.y_test_deno = data['y_test_deno']
            self.y_train_pred = data['y_train_pred']
            self.y_test_pred = data['y_test_pred']
            self.y_train_max_val = data['y_train_max_val']
            self.y_test_max_val = data['y_test_max_val']
            self.y_train_min_val = data['y_train_min_val']
            self.y_test_min_val = data['y_test_min_val']
            self.train_cls = None
            self.test_cls = None

            self.print(f'Load data from {self.data_path}.')
            hints = f'\t- MPJPE: train = {self.mpjpe(self.x_train, self.y_train_deno)}, ' \
                    f'test = {self.mpjpe(self.x_test, self.y_test_deno)}.'
            self.print(hints)
            hints = f'\t- MPJPE: ' \
                    f'train = {self.mpjpe(self.x_train, self.y_train_deno, self.y_train_max_val, self.y_train_min_val)}, ' \
                    f'test = {self.mpjpe(self.x_test, self.y_test_deno, self.y_test_max_val, self.y_test_min_val)}.'
            self.print(hints)
            hints = f'Done.'
            self.print(hints)

    def print(self, hints):
        print(hints) if self.logger is None else self.logger.info(hints)

    def mpjpe(self, pos1, pos2, max_val=None, min_val=None):

        if max_val is not None and min_val is not None:
            pos1_unnorm = pos1 * (max_val - min_val) + min_val
            pos2_unnorm = pos2 * (max_val - min_val) + min_val

            return np.mean(np.sqrt(np.sum(np.square(pos1_unnorm - pos2_unnorm), axis=-1)))

        else:

            return np.mean(np.sqrt(np.sum(np.square(pos1 - pos2), axis=-1)))

if __name__ == '__main__':
    data_path = './data/NYU/clean/train/joint_data.mat'
    data = nyu_dataset(data_path, t_size=36, t_pred_size=5, noise_level=0.1, sigma_o=0.05, sigma_s=0.05)
