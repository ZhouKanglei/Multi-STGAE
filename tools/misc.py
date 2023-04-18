# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/6/15 22:32

import argparse
import random

import numpy as np
import tensorflow as tf

from pydoc import locate

# Multi-processing callback func.
def get_idx(args):
    '''A callback function of multiprocessing'''
    return args


def append_dict(dict1: dict, dict2: dict, replace=False):
    """ append items in dict2 to dict1 """
    for key, value in dict2.items():
        if replace:
            dict1[key] = value
        else:
            if key not in dict1:
                dict1[key] = []
            value = tf.cast(value, dtype=tf.float32)
            dict1[key].append(value.numpy())

def append_avg_dict(dict1: dict, dict2: dict, replace=False):
    """ avg items in dict2 to dict1 """
    for key, value in dict2.items():
        if replace:
            dict1[key] = value
        else:
            if key not in dict1:
                dict1[key] = []
            dict1[key].append(sum(value) / len(value))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    totalParams = trainableParams + nonTrainableParams

    return trainableParams


def import_model(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def import_class(name):
    return locate(name)


def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

def get_flops(model, shape):
    print('TensorFlow:', tf.__version__)

    forward_pass = tf.function(
        model.call,
        input_signature=[tf.TensorSpec(shape=shape)])

    graph_info = profile(forward_pass.get_concrete_function().graph,
                         options=ProfileOptionBuilder.float_operation())

    # The //2 is necessary since `profile` counts multiply and accumulate
    # as two flops, here we report the total number of multiply accumulate ops
    flops = graph_info.total_float_ops // 2
    print('Flops: {:,}'.format(flops))

    return flops