# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/6/15 20:53

import argparse

import os
import time

import shutil
import yaml

import tensorflow as tf

from tools.misc import str2bool


class cmdAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest + '_non_default', True)


class parseArgs(object):

    def __init__(self):
        super(parseArgs, self).__init__()

        self.get_args()
        self.get_config()
        self.merge_config()
        self.check()

    def get_args(self):
        parser = argparse.ArgumentParser()

        # global
        parser.add_argument('--config_file', type=str, default='./config/nyu.yaml', help='config file',
                            action=cmdAction)
        parser.add_argument('--dataset', type=str, default='nyu', help='processor type',
                            action=cmdAction, choices=['nyu', 'shrec'])
        parser.add_argument('--mode', type=str, default='ensemble', help='processor type',
                            action=cmdAction, choices=['sep', 'ensemble'])
        parser.add_argument('--exp_name', type=str, default='default', help='experiment name', action=cmdAction)
        parser.add_argument('--res_tab', type=str, default='res.txt',
                            help='best result log path', action=cmdAction)
        parser.add_argument('--continue_train', type=str2bool, default=False, help='continue to train',
                            action=cmdAction)
        parser.add_argument('--load_best', type=str2bool, default=True,
                            help='continue to train and load the best weight',
                            action=cmdAction)
        parser.add_argument('--vis', type=str2bool, default=True,
                            help='if visualize the results',
                            action=cmdAction)

        # optimizer
        parser.add_argument('--optimizer', type=str, default='sgd', help='optimizers', action=cmdAction,
                            choices=['sgd', 'adam'])
        parser.add_argument('--max_epoch', type=int, default=300, help='# of epochs', action=cmdAction)
        parser.add_argument('--patience', type=int, default=10, help='reduce plateau callback', action=cmdAction)
        parser.add_argument('--base_lr', type=float, default=0.001, help='initial learning rate', action=cmdAction)
        parser.add_argument('--min_lr', type=float, default=0, help='minimum learning rate', action=cmdAction)
        parser.add_argument('--lr_factor', type=float, default=0.5, help='learning rate decay', action=cmdAction)
        parser.add_argument('--deno_weight', type=float, default=1, help='denoising loss weight', action=cmdAction)
        parser.add_argument('--pred_weight', type=float, default=1, help='prediction loss weight', action=cmdAction)

        # model
        parser.add_argument('--phase', type=str, default='train', help='train or test', choices=['train', 'test'],
                            action=cmdAction)
        parser.add_argument('--model', type=str, default='models.ours.stgae', help='model', action=cmdAction)
        parser.add_argument('--t_size', type=int, default=36, help='temporal size, 4x', action=cmdAction)
        parser.add_argument('--t_pred_size', type=int, default=16, help='prediction temporal size, 4x',
                            action=cmdAction)
        parser.add_argument('--bs_train', type=int, default=32, help='training batch size', action=cmdAction)
        parser.add_argument('--bs_test', type=int, default=32, help='testing or validation batch size',
                            action=cmdAction)
        parser.add_argument('--attention', type=str, default='A_B_C', help='attention type',
                            choices=['A', 'AxM', 'A_B', 'A_C', 'B_C', 'A_B_C'], action=cmdAction)
        parser.add_argument('--task', type=str, default='multi',
                            help='tasks, de: denoiseing, pred: prediction, multi: de & pred', action=cmdAction,
                            choices=['de', 'pred', 'multi'])
        parser.add_argument('--freeze_encoder', type=str2bool, default=False, help='freezing trainable params',
                            action=cmdAction)

        # others
        parser.add_argument('--noise', type=str, default='uniform', help='noise type', action=cmdAction)
        parser.add_argument('--strategy', type=str, default='spatial', help='graph type',
                            choices=['spatial', 'wo_indirect', 'wo_direct', 'wo_self'], action=cmdAction)
        parser.add_argument('--device', type=str, default=[0, 1, 2, 3], help='avaliable gpu list', action=cmdAction)
        parser.add_argument('--data_norm', type=str2bool, default=True, help='data normalization', action=cmdAction)
        parser.add_argument('--seed', type=int, default=1024, help='random seed', action=cmdAction)
        parser.add_argument('--sigma_o', type=float, default=0.1, help='occlusion mask', action=cmdAction)
        parser.add_argument('--sigma_s', type=float, default=0.1, help='shift mask', action=cmdAction)

        self.args = parser.parse_args()

    def get_config(self):
        if self.args.dataset == 'nyu':
            self.args.config_file = './config/nyu.yaml'
        elif self.args.dataset == 'shrec':
            self.args.config_file = './config/shrec.yaml'

        with open(self.args.config_file) as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

    def merge_config(self):
        for k, v in self.config.items():
            if k not in vars(self.args).keys():
                setattr(self.args, k, v)
            elif not hasattr(self.args, f'{k}_non_default'):
                setattr(self.args, k, v)

    def check(self):
        # set gpu setting
        self.args.device = [str(i) for i in self.args.device]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.args.device)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        tf.keras.backend.set_floatx = tf.float32

        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # others
        self.args.time_start = time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime())
        self.args.model_args['t_pred_size'] = self.args.t_pred_size

        if self.args.mode == 'sep':
            self.args.pred_decoder_args['t_pred_size'] = self.args.t_pred_size
            self.args.model_args['t_size'] = self.args.t_size
            self.args.pred_decoder_args['t_size'] = self.args.t_size
            self.args.model_args['graph_layout'] = self.args.dataset
            self.args.deno_decoder_args['graph_layout'] = self.args.dataset
            self.args.pred_decoder_args['graph_layout'] = self.args.dataset

        # root path
        if self.args.task != 'de':
            self.args.exp_path = os.path.join(f'./outputs/{self.args.dataset}', self.args.exp_name,
                                              f'{self.args.task}-{self.args.noise}-{self.args.strategy}-{self.args.attention}')
        else:
            self.args.exp_path = os.path.join(f'./outputs/{self.args.dataset}', self.args.exp_name,
                                              f'{self.args.noise}-{self.args.strategy}-{self.args.attention}')

        if not self.args.continue_train and os.path.exists(self.args.exp_path) and self.args.phase == 'train':
            shutil.rmtree(self.args.exp_path)
            print(f'Delete the original {self.args.exp_path}!')

        os.makedirs(self.args.exp_path, exist_ok=True)
        # res tab
        self.args.res_tab = os.path.join(f'./outputs/{self.args.dataset}/res', self.args.res_tab)
        os.makedirs(os.path.dirname(self.args.res_tab), exist_ok=True)
        # saving weight path
        self.args.weight_path = os.path.join(self.args.exp_path, 'weight')
        os.makedirs(self.args.weight_path, exist_ok=True)

        if not hasattr(self.args, 'opt_weight_file'):
            self.args.opt_weight_file = os.path.join(self.args.weight_path, f'best/cp.ckpt')
        self.args.tmp_weight_file = os.path.join(self.args.weight_path, f'tmp/cp.ckpt')
        # saving fig path
        self.args.fig_path = os.path.join(self.args.exp_path, 'figs')
        os.makedirs(self.args.fig_path, exist_ok=True)
        # logger path
        self.args.logger_path = os.path.join(self.args.exp_path, 'logger')
        os.makedirs(self.args.logger_path, exist_ok=True)
        self.args.logger_file = os.path.join(self.args.logger_path, f'{self.args.phase}.log')
        # tf log
        self.args.log_path = os.path.join(self.args.exp_path, 'log')
        if os.path.exists(self.args.log_path):
            shutil.rmtree(self.args.log_path)
        os.makedirs(self.args.log_path, exist_ok=True)
        # tf log
        self.args.his_path = os.path.join(self.args.exp_path, 'history')
        os.makedirs(self.args.his_path, exist_ok=True)
        self.args.his_file = os.path.join(self.args.his_path, f'{self.args.phase}_{self.args.time_start}')


if __name__ == '__main__':
    parser = parseArgs()