# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/7/22 21:08

from tools.processor_base import *


class processModel(processBase):

    def load_model(self):
        self.logger.info(f'Load model...')
        model = import_class(self.args.model)

        with self.strategy.scope():
            # models
            self.model = model(**self.args.model_args)
            self.encoder = self.model.enc
            # optimizer
            self.learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                [epoch * self.global_train_steps for epoch in self.args.lr_decay_epochs],
                [self.args.base_lr * 0.1 ** i for i in range(len(self.args.lr_decay_epochs) + 1)]
            )
            # self.learning_rate_fn = self.args.base_lr
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_fn)
            self.enco_optimizer = self.optimizer
            # loss
            self.loss_obj = mse(reduction=tf.keras.losses.Reduction.NONE)
            self.metric_obj = jpe(reduction=tf.keras.losses.Reduction.NONE)
            self.adaptive_loss = AutoWeightedLoss(num_tasks=2)

            # checkpoint
            self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
            self.manager = tf.train.CheckpointManager(self.checkpoint,
                                                      directory=os.path.dirname(self.args.tmp_weight_file),
                                                      max_to_keep=3)
            # load model weights
            latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(self.args.opt_weight_file))
            if latest_ckpt is not None and self.args.load_best:
                self.checkpoint.restore(latest_ckpt)
                self.logger.info('Load pre-trained model %s' % latest_ckpt)
            # load model weights
            latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(self.args.tmp_weight_file))
            if latest_ckpt is not None and not self.args.load_best:
                self.checkpoint.restore(latest_ckpt)
                self.logger.info('Load temp model %s' % latest_ckpt)

        # pre-inference
        self.tmp_best_error = np.inf
        if (self.args.phase == 'train' and self.args.continue_train) or self.args.phase == 'test':
            self.init_error = self.test()['error_unnorm'][-1]
            self.tmp_best_error = self.init_error

        if self.args.phase == 'train' and not self.args.continue_train:
            self.init_error = np.inf
            self.test()

        self.logger.info(f'Model (enc) params: {self.model.count_params():,d}')

    @tf.function
    def train_step(self, inputs):
        x_train, y_train_deno, y_train_pred, max_val, min_val = inputs

        result = {}

        with tf.GradientTape(persistent=True) as tape:
            y_deno_hat, y_pred_hat = self.model(x_train, training=True)

            deno_loss = self.mse_loss(y_train_deno, y_deno_hat)
            deno_loss += self.bone_loss(y_train_deno, y_deno_hat, self.direct_pos1_idx, self.direct_pos2_idx)
            # deno_loss += self.bone_loss(y_train_deno, y_deno_hat, self.indirect_pos1_idx, self.indirect_pos2_idx)
            pred_loss = self.mse_loss(y_train_pred, y_pred_hat)
            pred_loss += self.bone_loss(y_train_deno, y_deno_hat, self.direct_pos1_idx, self.direct_pos2_idx)
            # pred_loss += self.bone_loss(y_train_deno, y_deno_hat, self.indirect_pos1_idx, self.indirect_pos2_idx)

            total_loss = self.args.deno_weight * deno_loss + self.args.pred_weight * pred_loss
            # total_loss = self.adaptive_loss([deno_loss, pred_loss])

            result.update({'loss': total_loss, 'deno_loss': deno_loss, 'pred_loss': pred_loss,
                           'deno_weight': self.args.deno_weight,
                           'pred_weight': self.args.pred_weight})

        self.optimizer.minimize(loss=total_loss, tape=tape,
                                var_list=self.model.trainable_variables)

        return result, (y_deno_hat, y_pred_hat)

    @tf.function
    def test_step(self, inputs):
        result = {}
        x_test, y_test_deno, y_test_pred, max_val, min_val = inputs

        y_deno_hat, y_pred_hat = self.model(x_test, training=True)

        deno_error, deno_error_unnorm = self.mpjpe(y_test_deno, y_deno_hat, max_val, min_val)
        pred_error, pred_error_unnorm = self.mpjpe(y_test_pred, y_pred_hat, max_val, min_val)
        deno_loss = self.mse_loss(y_test_deno, y_deno_hat)
        pred_loss = self.mse_loss(y_test_pred, y_pred_hat)
        total_mse = self.mse_loss(tf.concat([y_test_deno, y_test_pred], axis=1),
                                  tf.concat([y_deno_hat, y_pred_hat], axis=1))

        total_error = self.args.deno_weight * deno_error + self.args.pred_weight * pred_error
        total_error_unnorm = self.args.deno_weight * deno_error_unnorm + self.args.pred_weight * pred_error_unnorm

        result.update({'error': total_error, 'deno_error': deno_error, 'pred_error': pred_error,
                       'error_unnorm': total_error_unnorm, 'deno_error_unnorm': deno_error_unnorm,
                       'pred_error_unnorm': pred_error_unnorm, 'deno_mse': deno_loss, 'pred_mse': pred_loss,
                       'total_mse': total_mse})

        return result, (y_deno_hat, y_pred_hat)
