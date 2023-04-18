# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/6/29 10:02

import logging
import pprint
import shutil
import yaml

from tools.tools import *

class processBase(object):

    def __init__(self, args):
        super(processBase, self).__init__()

        self.args = args

        # initialize the log
        self.init_log()

        # save the files
        self.save_files()

        # build graph
        self.build_graph()

        # load data
        self.load_data()

        # load model
        self.load_model()

    def load_model(self):
        self.logger.info(f'Load model...')
        encoder = import_class(self.args.encoder)
        deno_decoder = import_class(self.args.deno_decoder)
        pred_decoder = import_class(self.args.pred_decoder)

        with self.strategy.scope():
            # models
            self.encoder = encoder(**self.args.encoder_args)
            self.deno_decoder = deno_decoder(**self.args.deno_decoder_args)
            self.pred_decoder = pred_decoder(**self.args.pred_decoder_args)
            # optimizer
            self.learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                [epoch * self.global_train_steps for epoch in self.args.lr_decay_epochs],
                [self.args.base_lr * 0.1 ** i for i in range(len(self.args.lr_decay_epochs) + 1)]
            )
            # self.learning_rate_fn = self.args.base_lr
            self.enco_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_fn)
            self.deno_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_fn)
            self.pred_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_fn)

            # loss
            self.loss_obj = mse(reduction=tf.keras.losses.Reduction.NONE)
            self.metric_obj = jpe(reduction=tf.keras.losses.Reduction.NONE)
            self.adaptive_loss = AutoWeightedLoss(num_tasks=2)

            # checkpoint
            self.checkpoint = tf.train.Checkpoint(encoder=self.encoder,
                                                  deno_decoder=self.deno_decoder,
                                                  pred_decoder=self.pred_decoder,
                                                  enco_optimizer=self.enco_optimizer,
                                                  deno_optimizer=self.deno_optimizer,
                                                  pred_optimizer=self.pred_optimizer)
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

        self.logger.info(f'Model (enc) params: {self.encoder.count_params():,d}')
        self.logger.info(f'Model (den) params: {self.deno_decoder.count_params():,d}')
        self.logger.info(f'Model (pre) params: {self.pred_decoder.count_params():,d}')

    def build_graph(self):
        Graph = import_class(self.args.graph)
        graph = Graph(layout=self.args.graph_args['layout'], strategy=self.args.graph_args['strategy'])

        neighbor_direct_link = graph.neighbor_direct_link
        self.direct_pos1_idx, self.direct_pos2_idx = [], []
        for (i, j) in neighbor_direct_link:
            self.direct_pos1_idx.append([i])
            self.direct_pos2_idx.append([j])

        neighbor_indirect_link = graph.neighbor_indirect_link
        self.indirect_pos1_idx, self.indirect_pos2_idx = [], []
        for (i, j) in neighbor_indirect_link:
            self.indirect_pos1_idx.append([i])
            self.indirect_pos2_idx.append([j])


    def bone_loss(self, y_true, y_pred, pos1_idx, pos2_idx):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # true bone length
        y_true_transpose = tf.transpose(y_true, [2, 1, 0, 3])
        pos1 = tf.gather_nd(y_true_transpose, pos1_idx)
        pos1 = tf.transpose(pos1, [2, 1, 0, 3])
        pos2 = tf.gather_nd(y_true_transpose, pos2_idx)
        pos2 = tf.transpose(pos2, [2, 1, 0, 3])

        bone_length_true = tf.sqrt(tf.reduce_sum(tf.square(pos1 - pos2), axis=-1))

        # pred bone length
        y_pred_transpose = tf.transpose(y_pred, [2, 1, 0, 3])
        pos1 = tf.gather_nd(y_pred_transpose, pos1_idx)
        pos1 = tf.transpose(pos1, [2, 1, 0, 3])
        pos2 = tf.gather_nd(y_pred_transpose, pos2_idx)
        pos2 = tf.transpose(pos2, [2, 1, 0, 3])

        bone_length_pred = tf.sqrt(tf.reduce_sum(tf.square(pos1 - pos2), axis=-1))

        # loss
        loss = tf.reduce_sum(self.loss_obj(bone_length_true, bone_length_pred)) * (1. / self.args.bs_train)

        return loss

    def mse_loss(self, y_true, y_pred, max_val=None, min_val=None):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        loss = tf.reduce_sum(self.loss_obj(y_true, y_pred)) * (1. / self.args.bs_train)

        if max_val is not None and min_val is not None:
            max_val = tf.cast(max_val, dtype=tf.float32)
            min_val = tf.cast(min_val, dtype=tf.float32)

            y_true_unnorm = y_true * (max_val - min_val) + min_val
            y_pred_unnorm = y_pred * (max_val - min_val) + min_val
            loss_unnorm = tf.reduce_sum(self.loss_obj(y_true_unnorm, y_pred_unnorm)) * (1. / self.args.bs_train)

            return loss, loss_unnorm

        return loss

    def mpjpe(self, y_true, y_pred, max_val=None, min_val=None):
        '''Mean per joint position error'''
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        error = tf.reduce_sum(self.metric_obj(y_true, y_pred)) / self.args.bs_train

        if max_val is not None and min_val is not None:
            max_val = tf.cast(max_val, dtype=tf.float32)
            min_val = tf.cast(min_val, dtype=tf.float32)

            y_true_unnorm = y_true * (max_val - min_val) + min_val
            y_pred_unnorm = y_pred * (max_val - min_val) + min_val

            error_unnorm = tf.reduce_sum(self.metric_obj(y_true_unnorm, y_pred_unnorm)) / self.args.bs_train

            return error, error_unnorm

        return error

    def load_data(self):
        # load data
        data_loader = import_class(self.args.data_loader)
        data = data_loader(data_path=self.args.data_path, t_size=self.args.t_size, norm=self.args.data_norm,
                           test_ratio=self.args.validation_rate, t_pred_size=self.args.t_pred_size,
                           seed=self.args.seed, logger=self.logger, sigma_o=self.args.sigma_o,
                           sigma_s=self.args.sigma_s, beta=self.args.beta,
                           noise_level=self.args.noise_level,
                           noise_cycle=self.args.noise_cycle)
        self.data = data

        x_train, y_train_deno, y_train_pred = data.x_train, data.y_train_deno, data.y_train_pred
        x_test, y_test_deno, y_test_pred = data.x_test, data.y_test_deno, data.y_test_pred
        y_train_max_val, y_train_min_val = data.y_train_max_val, data.y_train_min_val
        y_test_max_val, y_test_min_val = data.y_test_max_val, data.y_test_min_val

        self.global_train_steps = int(np.ceil(len(x_train) / self.args.bs_train))
        self.global_test_steps = int(np.ceil(len(x_test) / self.args.bs_train))

        # tf data
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        self.strategy = tf.distribute.MirroredStrategy()

        self.dataset = {}
        if self.args.phase == 'train':
            train_data = (x_train, y_train_deno, y_train_pred, y_train_max_val, y_train_min_val)
            self.dataset['train'] = tf.data.Dataset.from_tensor_slices(train_data)
            self.dataset['train'] = self.dataset['train'].shuffle(1024, reshuffle_each_iteration=True)
            self.dataset['train'] = self.dataset['train'].batch(self.args.bs_train)
            self.dataset['train'] = self.strategy.experimental_distribute_dataset(self.dataset['train'])

        # test data
        test_data = (x_test, y_test_deno, y_test_pred, y_test_max_val, y_test_min_val)
        self.dataset['test'] = tf.data.Dataset.from_tensor_slices(test_data)
        self.dataset['eval'] = self.dataset['test'].batch(self.args.bs_train)
        self.dataset['test'] = self.strategy.experimental_distribute_dataset(self.dataset['eval'])

    def save_files(self):
        # save config file
        args_dict = vars(self.args)

        config_dir = os.path.join(self.args.exp_path, 'config')
        os.makedirs(config_dir, exist_ok=True)

        config_file = os.path.join(config_dir, f'{os.path.basename(self.args.config_file)}')
        with open(config_file, 'w') as f:
            yaml.dump(args_dict, f)

        self.logger.info(f'Save args to {config_file}.')

        # save model
        model_dir = os.path.join(self.args.exp_path, 'models')
        os.makedirs(model_dir, exist_ok=True)

        if self.args.phase == 'train':
            shutil.copytree('./models', model_dir, dirs_exist_ok=True)
            self.logger.info(f'Back-up copy models to {model_dir}.')

    def init_log(self):
        # logger: CRITICAL > ERROR > WARNING > INFO > DEBUG
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # stream handler
        log_sh = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(message)s', "%Y-%m-%d %H:%M:%S")
        log_sh.setFormatter(formatter)

        logger.addHandler(log_sh)

        # file handler
        log_fh = logging.FileHandler(self.args.logger_file, mode='a')
        log_fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(message)s', "%Y-%m-%d %H:%M:%S")
        log_fh.setFormatter(formatter)

        logger.addHandler(log_fh)

        self.logger = logger
        self.logger.info(f'Parameters:\n{pprint.pformat(vars(self.args))}\n')

    def save_history(self, history):
        writer = pd.ExcelWriter(self.args.his_file + '.xlsx', engine='xlsxwriter')
        pd.DataFrame(history['train']).to_excel(writer, sheet_name='train')
        pd.DataFrame(history['test']).to_excel(writer, sheet_name='test')
        writer.save()
        self.logger.info("Save history: %s" % self.args.his_file + '.xlsx')

        # plot history
        plot_history(self.args.his_file + '.xlsx')

        # save opt weight
        if min(history['test']['error_unnorm']) < self.init_error:
            self.init_error = min(history['test']['error_unnorm'])

            if os.path.exists(os.path.dirname(self.args.opt_weight_file)):
                shutil.rmtree(os.path.dirname(self.args.opt_weight_file))

            os.rename(os.path.dirname(self.args.tmp_weight_file), os.path.dirname(self.args.opt_weight_file))
            self.logger.info(f'Move {self.args.tmp_weight_file} to {self.args.opt_weight_file}')

    @tf.function
    def train_step(self, inputs):
        x_train, y_train_deno, y_train_pred, max_val, min_val = inputs

        result = {}

        with tf.GradientTape(persistent=True) as tape:
            h = self.encoder(x_train, training=True)
            y_deno_hat = self.deno_decoder(h, training=True)
            y_pred_hat = self.pred_decoder(h, training=True)

            deno_loss = self.mse_loss(y_train_deno, y_deno_hat)
            pred_loss = self.mse_loss(y_train_pred, y_pred_hat)
            total_loss = self.args.deno_weight * deno_loss + self.args.pred_weight * pred_loss

            result.update({'loss': total_loss, 'deno_loss': deno_loss, 'pred_loss': pred_loss})

            deno_loss_1 = self.args.deno_weight * deno_loss
            pred_loss_1 = self.args.pred_weight * pred_loss

        self.deno_optimizer.minimize(loss=deno_loss_1, var_list=self.deno_decoder.trainable_variables, tape=tape)
        self.pred_optimizer.minimize(loss=pred_loss_1, var_list=self.pred_decoder.trainable_variables, tape=tape)
        self.enco_optimizer.minimize(loss=total_loss, var_list=self.encoder.trainable_variables, tape=tape)

        return result, (y_deno_hat, y_pred_hat)

    @tf.function
    def distributed_train_step(self, inputs):
        result, _ = self.strategy.run(self.train_step, args=(inputs,))
        return {
            k: self.strategy.reduce(tf.distribute.ReduceOp.SUM, v, axis=None)
            for k, v in result.items()
        }

    def train(self):
        history = {'train': {}, 'test': {}}

        for epoch in range(self.args.max_epoch):
            self.logger.info(f'+-----------------Epoch {epoch + 1}/{self.args.max_epoch}--------------------+')
            # train
            pbar = tqdm(self.dataset['train'], dynamic_ncols=True, total=self.global_train_steps)
            results = {}

            for batch, batch_inputs in enumerate(pbar):
                result = self.distributed_train_step(batch_inputs)
                append_dict(results, result)

                pbar_desc = '[Train] loss: {:.4f}'
                pbar.set_description(pbar_desc.format(result['loss']))
                pbar.update()

            pbar.close()
            append_avg_dict(history['train'], results)

            self.logger.info(' [Train] avg loss: {:.4f}'.format(history['train']['loss'][-1]))
            self.logger.info('         avg deno: {:.4f}'.format(history['train']['deno_loss'][-1]))
            self.logger.info('         avg pred: {:.4f}'.format(history['train']['pred_loss'][-1]))
            self.logger.info('         deno weight: {:.4f}'.format(history['train']['deno_weight'][-1]))
            self.logger.info('         pred weight: {:.4f}'.format(history['train']['pred_weight'][-1]))
            lr = self.enco_optimizer.lr(self.global_train_steps * (epoch + 1))
            self.logger.info('         lr: {:.4f}'.format(lr))

            # test
            result = self.test()
            append_avg_dict(history['test'], result)

            # save checkpoint
            if history['test']['error_unnorm'][-1] < self.tmp_best_error:
                self.tmp_best_error = history['test']['error_unnorm'][-1]
                self.manager.save(checkpoint_number=epoch + 1)
                ckpt_file_path = tf.train.latest_checkpoint(os.path.dirname(self.args.tmp_weight_file))
                self.logger.info(f'---- Save checkpoint to {ckpt_file_path}. ----')

                self.save_history(history)

        self.save_history(history)
        self.logger.info(f'+-----------------Training done.--------------------+')

        latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(self.args.opt_weight_file))
        if latest_ckpt is not None and self.args.load_best:
            self.checkpoint.restore(latest_ckpt)
            self.logger.info('Load pre-trained model %s' % latest_ckpt)

    @tf.function
    def test_step(self, inputs):
        result = {}
        x_test, y_test_deno, y_test_pred, max_val, min_val = inputs

        h = self.encoder(x_test, training=False)
        y_deno_hat = self.deno_decoder(h, training=False)
        y_pred_hat = self.pred_decoder(h, training=False)

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

    @tf.function
    def distributed_test_step(self, inputs):
        result, _ = self.strategy.run(self.test_step, args=(inputs,))
        return {
            k: self.strategy.reduce(tf.distribute.ReduceOp.SUM, v, axis=None)
            for k, v in result.items()
        }

    def test(self):
        history, results = {}, {}
        pbar = tqdm(self.dataset['test'], dynamic_ncols=True, total=self.global_test_steps)
        results = {}
        for batch, batch_inputs in enumerate(pbar):
            result = self.distributed_test_step(batch_inputs)
            append_dict(results, result)

            pbar_desc = '[Test] error: {:.4f}e-4'
            pbar.set_description(pbar_desc.format(result['error'] * 1e4))
            pbar.update()

        pbar.close()

        append_avg_dict(history, results)

        self.logger.info(' [Eval] mpjpe: {:.4f}\t {:.4f} ({:.4f})'.
                         format(history['error'][-1], history['error_unnorm'][-1], self.tmp_best_error))
        self.logger.info('         deno: {:.4f}\t {:.4f} \t{:.4f}'.
                         format(history['deno_error'][-1], history['deno_error_unnorm'][-1], history['deno_mse'][-1]))
        self.logger.info('         pred: {:.4f}\t {:.4f} \t{:.4f}'.
                         format(history['pred_error'][-1], history['pred_error_unnorm'][-1], history['pred_mse'][-1]))

        return history

    def evaluate(self):
        history, results = {}, {}
        x_test = y_test_deno = y_test_pred = max_val = min_val = None
        y_test_deno_hat = y_test_pred_hat = None

        def concat(x, y):
            return y if x is None else np.concatenate((x, y), axis=0)

        pbar = tqdm(self.dataset['eval'], dynamic_ncols=True, ncols=self.global_test_steps)
        for batch, batch_inputs in enumerate(pbar):
            result, (batch_y_deno_hat, batch_y_pred_hat) = self.test_step(batch_inputs)
            append_dict(results, result)

            x_test = concat(x_test, batch_inputs[0].numpy())
            y_test_deno = concat(y_test_deno, batch_inputs[1].numpy())
            y_test_pred = concat(y_test_pred, batch_inputs[2].numpy())
            max_val = concat(max_val, batch_inputs[3].numpy())
            min_val = concat(min_val, batch_inputs[4].numpy())
            y_test_deno_hat = concat(y_test_deno_hat, batch_y_deno_hat.numpy())
            y_test_pred_hat = concat(y_test_pred_hat, batch_y_pred_hat.numpy())

            pbar.set_description('[Eval]')
            pbar.update()
        pbar.close()

        # print history
        history = {}
        append_avg_dict(history, results)

        self.logger.info(' [Eval] mpjpe: {:.4f}e-4\t {:.4f} ({:.4f})'.
                         format(history['error'][-1] * 1e4, history['error_unnorm'][-1], self.tmp_best_error))
        self.logger.info('         deno: {:.4f}\t {:.4f} \t{:.4f}'.
                         format(history['deno_error'][-1], history['deno_error_unnorm'][-1], history['deno_mse'][-1]))
        self.logger.info('         pred: {:.4f}\t {:.4f} \t{:.4f}'.
                         format(history['pred_error'][-1], history['pred_error_unnorm'][-1], history['pred_mse'][-1]))

        # save result
        self.save_res(x_test, y_test_deno, y_test_pred, y_test_deno_hat, y_test_pred_hat, max_val, min_val, history)

    def save_res(self, x_test, y_test_deno, y_test_pred, y_test_deno_hat, y_test_pred_hat,
                 max_val=None, min_val=None, history=None):

        y_test = np.concatenate((y_test_deno, y_test_pred), axis=1)
        y_test_hat = np.concatenate((y_test_deno_hat, y_test_pred_hat), axis=1)

        # print history using another calculation
        history1 = {}
        history1['error'], history1['error_unnorm'] = calculate_all_mpjpe(y_test, y_test_hat, max_val, min_val)
        history1['deno_error'], history1['deno_error_unnorm'] \
            = calculate_all_mpjpe(y_test_deno, y_test_deno_hat, max_val, min_val)
        history1['pred_error'], history1['pred_error_unnorm'] \
            = calculate_all_mpjpe(y_test_pred, y_test_pred_hat, max_val, min_val)

        history1['deno_mse'] = np.mean(np.square(y_test_deno - y_test_deno_hat))
        history1['pred_mse'] = np.mean(np.square(y_test_pred - y_test_pred_hat))
        history1['total_mse'] = np.mean(np.square(np.concatenate([y_test_deno, y_test_pred], axis=1) -
                                                 np.concatenate([y_test_deno_hat, y_test_pred_hat], axis=1)))

        self.logger.info(' [Eval] mpjpe: {:.4f}e-4\t {:.4f} ({:.4f})'.
                         format(history1['error'] * 1e4, history1['error_unnorm'], self.tmp_best_error))
        self.logger.info('         deno: {:.4f}\t {:.4f} \t{:.4f}'.
                         format(history1['deno_error'], history1['deno_error_unnorm'], history1['deno_mse']))
        self.logger.info('         pred: {:.4f}\t {:.4f} \t{:.4f}'.
                         format(history1['pred_error'], history1['pred_error_unnorm'], history1['pred_mse']))

        # pose mse
        jpe_pose, jpe_pose_unnorm = calculate_all_mpjpe(y_test, y_test_hat, max_val, min_val)
        self.logger.info(f'Hand pose error: {jpe_pose:.6f}, {jpe_pose_unnorm:.4f}')

        # bone length evaluation
        mse_bone_len_direct, mse_bone_len_direct_unnorm = \
            calculate_all_bone_length_error(y_test, y_test_hat, max_val, min_val)
        self.logger.info(f'Bone length error: {mse_bone_len_direct:.6f}, {mse_bone_len_direct_unnorm:.4f}')

        mse_bone_len_direct_deno, mse_bone_len_direct_unnorm_deno = \
            calculate_all_bone_length_error(y_test_deno, y_test_deno_hat, max_val, min_val)
        self.logger.info(f'Bone length error (deno): {mse_bone_len_direct_deno:.6f}, {mse_bone_len_direct_unnorm_deno:.4f}')

        mse_bone_len_direct_pred, mse_bone_len_direct_unnorm_pred = \
            calculate_all_bone_length_error(y_test_pred, y_test_pred_hat, max_val, min_val)
        self.logger.info(f'Bone length error (pred): {mse_bone_len_direct_pred:.6f}, {mse_bone_len_direct_unnorm_pred:.4f}')

        # Symmetrical neighbor length evaluation
        mse_bone_len_indirect, mse_bone_len_indirect_unnorm = \
            calculate_all_bone_length_error(y_test, y_test_hat, max_val, min_val, edge='indirect')
        self.logger.info(f'Symmetrical neighbor error: {mse_bone_len_indirect:.6f}, {mse_bone_len_indirect_unnorm:.4f}')

        mse_bone_len_indirect_deno, mse_bone_len_indirect_unnorm_deno = \
            calculate_all_bone_length_error(y_test_deno, y_test_deno_hat, max_val, min_val, edge='indirect')
        self.logger.info(f'Symmetrical neighbor error (deno): {mse_bone_len_indirect_deno:.6f}, {mse_bone_len_indirect_unnorm_deno:.4f}')

        mse_bone_len_indirect_pred, mse_bone_len_indirect_unnorm_pred = \
            calculate_all_bone_length_error(y_test_pred, y_test_pred_hat, max_val, min_val, edge='indirect')
        self.logger.info(f'Symmetrical neighbor error (pred): {mse_bone_len_indirect_pred:.6f}, {mse_bone_len_indirect_unnorm_pred:.4f}')

        # save loss
        res_info = '\n%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%.2f\t%.2f\n' % (
            self.args.time_start, self.args.exp_name, self.args.phase, self.args.noise, self.args.strategy,
            self.args.attention, self.args.t_size, self.args.t_pred_size, self.args.deno_weight, self.args.pred_weight)

        res_info = res_info + '\ttotal = %.4f (%.6f)\tdeno = %.4f (%.6f)\tpred = %.4f (%.6f)\n' \
                   % (history['error_unnorm'][-1], history['error'][-1], history['deno_error_unnorm'][-1],
                      history['deno_error'][-1], history['pred_error_unnorm'][-1], history['pred_error'][-1])

        res_info = res_info + '\tmpjpe = %.4f (%.6f)\tbone = %.4f (%.6f)\tsymm = %.4f (%.6f)\n' \
                   % (jpe_pose_unnorm, jpe_pose, mse_bone_len_direct_unnorm, mse_bone_len_direct,
                      mse_bone_len_indirect_unnorm, mse_bone_len_indirect)

        res_info = res_info + '\tdeno_mse = %.4f\tpred_mse = %.4f\ttotal_mse = %.4f\n' \
                   % (history['deno_mse'][-1], history['pred_mse'][-1], history['total_mse'][-1])

        res_info = res_info + '\tbone_deno = %.4f (%.6f)\tbone_pred = %.4f (%.6f)\n' \
                   % (mse_bone_len_direct_deno, mse_bone_len_direct_unnorm_deno,
                      mse_bone_len_direct_pred, mse_bone_len_direct_unnorm_pred)

        res_info = res_info + '\tsymm_deno = %.4f (%.6f)\tsymm_pred = %.4f (%.6f)\n' \
                   % (mse_bone_len_indirect_deno, mse_bone_len_indirect_unnorm_deno,
                      mse_bone_len_indirect_pred, mse_bone_len_indirect_unnorm_pred)

        if not os.path.exists(self.args.res_tab):
            with open(self.args.res_tab, 'w', encoding='utf-8') as f:
                f.write(res_info)
                f.close()
        else:
            with open(self.args.res_tab, 'a', encoding='utf-8') as f:
                f.write(res_info)
                f.close()

        self.logger.info(res_info)

        # visualization
        if self.args.vis == True:
            # save adj heat-maps
            A = self.encoder.enc[0].gc.B
            A = np.eye(A.shape[1])  + A
            deal_learnable_adj(A, self.args.fig_path)

            A = self.encoder.enc[0].gc.A
            deal_learnable_adj(A, self.args.fig_path, start=4)

            # visualize sample
            vis_idx = 164
            fig_path = os.path.join(self.args.fig_path, str(vis_idx))
            os.makedirs(fig_path, exist_ok=True)

            deno_mse = np.mean(np.square(y_test[vis_idx, :x_test.shape[1]] - y_test_hat[vis_idx, :x_test.shape[1]]))
            pred_mse = np.mean(np.square(y_test[vis_idx, x_test.shape[1]:] - y_test_hat[vis_idx, x_test.shape[1]:]))
            print(deno_mse, '------- ', pred_mse)

            # save video and figs of result
            if self.args.phase == 'test':
                # plot_save_pred(x_test, y_test, y_test_hat, fig_path, vis_idx)

                # plot_save_single_video(x_test, y_test, y_test_hat, fig_path, vis_idx, mode='gt')
                # plot_save_single_video(x_test, y_test, y_test_hat, fig_path, vis_idx, mode='input')
                # plot_save_single_video(x_test, y_test, y_test_hat, fig_path, vis_idx, mode='output')
                plot_save_single_video(x_test, y_test, y_test_hat, fig_path, vis_idx, mode='all')

                # plot_save_figs(x_test, y_test, y_test_hat, fig_path, vis_idx)
                # plot_save_video(x_test, y_test, y_test_hat, fig_path, vis_idx)
            else:
                fig_path = os.path.join(self.args.fig_path, str(vis_idx))
                os.makedirs(fig_path, exist_ok=True)

                plot_save_video(x_test, y_test, y_test_hat, fig_path, vis_idx)

                plot_save_figs(x_test, y_test, y_test_hat, fig_path, vis_idx)

            # save trajectory
            plot_save_trajectory(x_test, y_test, y_test_hat, fig_path, vis_idx)

            # save mse error
            plot_error(x_test, y_test, y_test_hat, fig_path, vis_idx, max_val, min_val)

    def start(self):

        if self.args.phase == 'train':
            self.train()

        self.evaluate()

        pass