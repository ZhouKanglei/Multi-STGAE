# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/9/7 10:11

import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from einops.layers.tensorflow import Rearrange

from config.graph import Graph

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Activation, Lambda, \
    BatchNormalization, Dropout, Conv2D, UpSampling2D, AveragePooling2D, LeakyReLU


class res_block(Model):

    def __init__(self, in_filters, filters, stride=1, residual=False,
                 padding='same', kernel_size=1, groups=1):
        super(res_block, self).__init__()

        kernel_size = (kernel_size, 1)

        if residual:
            pool = None
            if stride == 1:
                pool = Lambda(lambda x: x)
            elif stride == 2:
                pool = AveragePooling2D(pool_size=(stride, 1))
            elif stride == 0.5:
                pool = UpSampling2D(size=(1 // stride, 1))

            self.conv = Sequential([
                Conv2D(filters=filters, kernel_size=(1, 1), groups=1) if padding == 'same' else \
                    Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, groups=groups),
                BatchNormalization(),
                pool,
            ])
        else:
            self.conv = Lambda(lambda x: 0)

    def call(self, inputs, training=None, mask=None):
        return self.conv(inputs)


class att_adj(Model):

    def __init__(self, filters, k_size=1, bias=False):
        super(att_adj, self).__init__()

        self.k_size = k_size

        self.conv_k = Sequential([
            Conv2D(filters=filters, kernel_size=(1, 1), use_bias=bias),
            Rearrange('n t v (c k) -> n t v k c', k=self.k_size)
        ])
        self.conv_q = Sequential([
            Conv2D(filters=filters, kernel_size=(1, 1), use_bias=bias),
            Rearrange('n t v (c k) -> n t v k c', k=self.k_size)
        ])

    def call(self, inputs, training=None, mask=None):
        q, k = self.conv_q(inputs), self.conv_k(inputs)

        C = tf.einsum('n t v k c, n t u k c -> n k u v', q, k)
        C = tf.nn.softmax(C) / np.sqrt(q.shape[-1])

        return C


class gat_adj(Model):

    def __init__(self, filters, k_size=1, bias=False):
        super(gat_adj, self).__init__()

        self.k_size = k_size

        self.conv = Sequential([
            Conv2D(filters=filters, kernel_size=(1, 1), use_bias=bias),
            Rearrange('n t v (c k) -> n t v k c', k=self.k_size)
        ])

        self.a = Conv2D(filters=1, kernel_size=(1, 1), use_bias=bias)

    def call(self, inputs, training=None, mask=None):
        h = tf.reduce_mean(self.conv(inputs), axis=1)

        n, v, k, c = h.shape
        h_i = tf.tile(h, [1, v, 1, 1])
        h_j = tf.repeat(h, repeats=v, axis=1)
        h_cat = tf.concat([h_i, h_j], axis=-1)

        C = tf.nn.leaky_relu(self.a(h_cat))
        C = tf.reshape(C, [n, k, v, v])
        C = tf.nn.softmax(C)

        return C


class gcn(Model):

    def __init__(self,
                 in_filters,
                 filters,
                 attention_type,
                 groups=1,
                 dropout=0.0,
                 graph_layout='nyu',
                 graph_strategy='spatial'):
        super(gcn, self).__init__()

        self.attention_type = attention_type
        graph = Graph(layout=graph_layout, strategy=graph_strategy)
        self.mask = tf.convert_to_tensor(graph.A)

        self.A = tf.convert_to_tensor(graph.A, dtype=tf.float32)
        self.M = tf.where(self.A == 0, 0, 1)
        self.B = self.add_weight(name='B', shape=self.A.shape, trainable=True)
        self.C = None

        self.k_size = k_size = self.A.shape[0]
        inter_filters = in_filters // 2
        self.att_adj = gat_adj(filters=inter_filters * k_size, k_size=k_size)

        self.conv = Sequential([
            Conv2D(filters=filters * k_size, kernel_size=(1, 1), padding='same', groups=groups),
            BatchNormalization(),
            Activation('ReLU'),
            Dropout(dropout)
        ])

    def graph_adj(self, x):
        adj = None
        self.C = self.att_adj(x)
        if self.attention_type == 'A_B_C':
            adj = self.A + self.B + self.C
        elif self.attention_type == 'A_B':
            adj = self.A + self.B

        return adj

    def call(self, inputs, training=None, mask=None):

        h = self.conv(inputs)

        _, t, v, c = h.shape
        h = tf.reshape(h, [-1, t, v, self.k_size, c // self.k_size])

        adj = self.graph_adj(inputs)

        if len(adj.shape) > 3:
            y = tf.einsum('n t v k c, n k v w -> n t w c', h, adj)
        else:
            y = tf.einsum('n t v k c, k v w -> n t w c', h, adj)

        return y


class tcn(Model):

    def __init__(self, filters, kernel_size, padding='same', dropout=0.0, stride=1, groups=1):
        super(tcn, self).__init__()

        pool = None
        if stride == 1:
            pool = Lambda(lambda x: x)
        elif stride == 2:
            pool = AveragePooling2D(pool_size=(stride, 1))
        elif stride == 0.5:
            pool = UpSampling2D(size=(1 // stride, 1))

        self.tc_block = Sequential([
            Conv2D(filters=filters, kernel_size=(kernel_size, 1), padding=padding, groups=groups),
            Activation('ReLU'),
            BatchNormalization(),
            pool,
            Dropout(dropout)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.tc_block(inputs)


class stgc(Model):

    def __init__(self,
                 in_filters,
                 filters,
                 kernel_size,
                 stride=None,
                 dropout=0,
                 residual=True,
                 attention_type=None,
                 padding='same',
                 graph_layout='nyu',
                 graph_strategy='spatial'):
        super(stgc, self).__init__()

        # groups = 1 if in_filters < 16 or filters < 16 else filters // 8
        groups = 1

        # gcn
        self.gc = gcn(in_filters, filters, attention_type, dropout=dropout, groups=groups,
                      graph_layout=graph_layout, graph_strategy=graph_strategy)
        # tcn
        self.tc = tcn(filters, kernel_size, stride=stride, dropout=dropout, padding=padding, groups=groups)
        # res
        self.res = res_block(in_filters, filters, kernel_size=kernel_size, stride=stride,
                             residual=residual, padding=padding, groups=groups)

    def call(self, inputs, training=None, mask=None):
        h = self.gc(inputs)
        y = self.tc(h) + self.res(inputs)

        return y

class st_att(Model):
    """spatial temporal attention """
    def __init__(self, in_filters):
        super(st_att, self).__init__()

        mid_filters = in_filters // 2
        self.fcn = Sequential([
            Conv2D(mid_filters, kernel_size=(1, 1)),
            BatchNormalization(),
            Activation('relu'),
        ])

        self.conv_v = Conv2D(in_filters, kernel_size=(1, 1))
        self.conv_t = Conv2D(in_filters, kernel_size=(1, 1))

        self.data_bn = BatchNormalization()
        self.act = Activation('relu')

    def call(self, inputs, training=None, mask=None):
        x = inputs

        x_t = tf.reduce_mean(x, axis=1, keepdims=True)
        x_v = tf.reduce_mean(x, axis=2, keepdims=True)

        x_t_att = tf.sigmoid(self.conv_t(self.fcn(x_t)))
        x_v_att = tf.sigmoid(self.conv_v(self.fcn(x_v)))

        att = x_t_att * x_v_att

        out = self.act(self.data_bn(x + x * att))

        return out

class encoder(Model):

    def __init__(self,
                 filters=3,
                 kernel_size=9,
                 attention_type='A_B_C',
                 dropout=0,
                 graph_layout='nyu',
                 graph_strategy='spatial',
                 channels=(64, 128, 256)):
        super(encoder, self).__init__()

        c1, c2, c3 = channels
        self.att = st_att(c3)

        # encoder
        self.enc = [
            stgc(in_filters=filters, filters=c1, kernel_size=kernel_size, stride=1, attention_type=attention_type,
                 dropout=dropout, graph_layout=graph_layout, graph_strategy=graph_strategy),
            stgc(in_filters=c1, filters=c2, kernel_size=kernel_size, stride=1, attention_type=attention_type,
                 dropout=dropout, graph_layout=graph_layout, graph_strategy=graph_strategy),
            stgc(in_filters=c2, filters=c2, kernel_size=kernel_size, stride=1, attention_type=attention_type,
                 dropout=dropout, graph_layout=graph_layout, graph_strategy=graph_strategy),
            stgc(in_filters=c2, filters=c3, kernel_size=kernel_size, stride=1, attention_type=attention_type,
                 dropout=dropout, graph_layout=graph_layout, graph_strategy=graph_strategy),
            stgc(in_filters=c3, filters=c3, kernel_size=kernel_size, stride=1, attention_type=attention_type,
                 dropout=dropout, graph_layout=graph_layout, graph_strategy=graph_strategy),
        ]

    def call(self, inputs, training=None, mask=None):
        x = inputs

        h1 = self.enc[0](x)
        h2 = self.enc[1](h1)
        h3 = self.enc[2](h2)
        h4 = self.enc[3](h3)
        h5 = self.enc[4](h4)
        h5 = self.att(h5)

        return (x, h1, h2, h3, h4, h5)

class deno_decoder(Model):

    def __init__(self,
                 filters=3,
                 kernel_size=9,
                 attention_type='A_B_C',
                 residual=False,
                 dropout=0,
                 graph_layout='nyu',
                 graph_strategy='spatial',
                 channels=(64, 128, 256)):
        super(deno_decoder, self).__init__()

        c1, c2, c3 = channels

        # decoder
        self.dec = [
            stgc(in_filters=c3, filters=c3, kernel_size=kernel_size, stride=1, attention_type=attention_type,
                 dropout=dropout, graph_layout=graph_layout, graph_strategy=graph_strategy),
            stgc(in_filters=c3, filters=c2, kernel_size=kernel_size, stride=1, attention_type=attention_type,
                 dropout=dropout, graph_layout=graph_layout, graph_strategy=graph_strategy),
            stgc(in_filters=c2, filters=c2, kernel_size=kernel_size, stride=1, attention_type=attention_type,
                 dropout=dropout, graph_layout=graph_layout, graph_strategy=graph_strategy),
            stgc(in_filters=c2, filters=c1, kernel_size=kernel_size, stride=1, attention_type=attention_type,
                 dropout=dropout, graph_layout=graph_layout, graph_strategy=graph_strategy),
            stgc(in_filters=c1, filters=filters, kernel_size=kernel_size, stride=1, attention_type=attention_type,
                 dropout=dropout, graph_layout=graph_layout, graph_strategy=graph_strategy),
        ]


    def call(self, inputs, training=None, mask=None):
        x, h1, h2, h3, h4, h5 = inputs

        h6 = self.dec[0](h5)
        h7 = self.dec[1](h6)
        h8 = self.dec[2](h7)
        h9 = self.dec[3](h8)
        y = self.dec[4](h9)

        return y

class pred_decoder(Model):

    def __init__(self,
                 filters=3,
                 kernel_size=9,
                 t_pred_size=9,
                 t_size=36,
                 attention_type='A_B_C',
                 dropout=0,
                 graph_layout='nyu',
                 graph_strategy='spatial',
                 channels=(64, 128, 256)):
        super(pred_decoder, self).__init__()

        self.filters = filters
        self.t_size = t_size
        self.t_pred_size = t_pred_size
        self.kernel_size = t_size - t_pred_size + 1
        c1, c2, c3 = channels

        self.att = st_att(c3)

        # decoder
        self.dec = [
            stgc(in_filters=c3, filters=c3, kernel_size=self.kernel_size, stride=1, attention_type=attention_type,
                 dropout=dropout, graph_layout=graph_layout, graph_strategy=graph_strategy, padding='valid'),
            stgc(in_filters=c3, filters=c2, kernel_size=kernel_size, stride=1, attention_type=attention_type,
                 dropout=dropout, graph_layout=graph_layout, graph_strategy=graph_strategy),
            stgc(in_filters=c2, filters=c2, kernel_size=kernel_size, stride=1, attention_type=attention_type,
                 dropout=dropout, graph_layout=graph_layout, graph_strategy=graph_strategy),
            stgc(in_filters=c2, filters=c1, kernel_size=kernel_size, stride=1, attention_type=attention_type,
                 dropout=dropout, graph_layout=graph_layout, graph_strategy=graph_strategy),
            stgc(in_filters=c1, filters=filters, kernel_size=kernel_size, stride=1, attention_type=attention_type,
                 dropout=dropout, graph_layout=graph_layout, graph_strategy=graph_strategy),
        ]

    def encode(self, max_len, d_model):
        pe = np.zeros([max_len, d_model], dtype=np.float32)
        position = np.expand_dims(np.arange(0, max_len, dtype=np.float32), axis=1)
        div_term = np.exp(np.arange(0.0, float(d_model), 2.0, dtype=np.float32) * -(np.log(10000.0) / float(d_model)))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = tf.convert_to_tensor(pe, dtype=tf.float32)
        pe = tf.expand_dims(pe, axis=0)
        pe = tf.expand_dims(pe, axis=2)
        return pe

    def call(self, inputs, training=None, mask=None):
        x, h1, h2, h3, h4, h5 = inputs

        h5 = self.att(h5)

        h6 = self.dec[0](h5)
        pe = self.encode(self.t_pred_size, h6.shape[-1])
        h7 = self.dec[1](h6 + pe)
        h8 = self.dec[2](h7)
        h9 = self.dec[3](h8)
        y = self.dec[4](h9)

        return y

class stgae(Model):

    def __init__(self,
                 filters=3,
                 kernel_size=9,
                 residual=False,
                 attention_type='A_B_C',
                 t_size=36,
                 t_pred_size=9,
                 dropout=0,
                 graph_layout='nyu',
                 graph_strategy='spatial',
                 channels=(64, 128, 256)):
        super(stgae, self).__init__()

        self.t_size = t_size

        self.enc = encoder(filters=filters, kernel_size=kernel_size, attention_type=attention_type, dropout=dropout,
                           graph_layout=graph_layout, graph_strategy=graph_strategy, channels=channels)
        self.deno = deno_decoder(filters=filters, residual=residual, attention_type=attention_type, dropout=dropout,
                                 graph_layout=graph_layout, graph_strategy=graph_strategy, channels=channels)
        self.pred = pred_decoder(filters=filters, t_size=t_size, t_pred_size=t_pred_size, kernel_size=kernel_size,
                                 dropout=dropout, attention_type=attention_type, graph_layout=graph_layout,
                                 graph_strategy=graph_strategy, channels=channels)

    def call(self, inputs, training=None, mask=None):
        x = inputs

        h = self.enc(inputs)
        y_deno = self.deno(h) + x
        y_pred = self.pred(h) + y_deno[:, -1:, :, :]

        return y_deno, y_pred


if __name__ == '__main__':
    features = np.random.randn(1, 25, 22, 3)

    enc = encoder(filters=3, graph_layout='shrec')
    deno = deno_decoder(filters=3, residual=False, graph_layout='shrec')
    pred = pred_decoder(filters=3, t_pred_size=15, graph_layout='shrec', t_size=features.shape[1])
    # print summary
    h = enc(features[:1, :, :, :])
    y_deno = deno(h)
    y_pred = pred(h)

    print(y_pred.shape)

    from tools.misc import count_params

    print(f'enco: {count_params(enc):,d}')
    print(f'deno: {count_params(deno):,d}')
    print(f'pred: {count_params(pred):,d}')

    model = stgae(filters=3, graph_layout='shrec', t_size=features.shape[1], t_pred_size=15)
    y_deno_hat, y_pred_hat = model(features[:1, :, :, :])
    print(y_deno_hat.shape, y_pred_hat.shape)
