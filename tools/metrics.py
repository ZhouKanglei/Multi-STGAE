import numpy as np
import tensorflow as tf

from config.graph import Graph

class AutoWeightedLoss(tf.keras.models.Model):
    """Auto weighting loss"""

    def __init__(self, num_tasks=2):
        super(AutoWeightedLoss, self).__init__()

        coefs = tf.ones((num_tasks))
        self.coefs = self.add_weight(name='adaptive_loss_weight', shape=coefs.shape,
                                     initializer='ones', trainable=True)


    def call(self, inputs, training=None, mask=None):
        loss = 0
        
        for i in range(len(self.coefs)):
            square = self.coefs[i] ** 2
            loss += inputs[i] / (2 * square) + tf.math.log(1 + square)
            
        return loss

class PiecewiseConstantDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, boundaries, values):
        super().__init__()
        self.boundaries = tf.cast(boundaries, dtype=tf.float32)
        self.values = tf.cast(values, dtype=tf.float32)

    def __call__(self, step):
        for i in range(len(self.boundaries)):
            if self.boundaries[i] >= step:
                return self.values[i]
        else:
            return self.values[-1]


class mae(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        """ return the per sample mean absolute error """
        y_true = tf.cast(y_true, dtype=tf.float64)
        y_pred = tf.cast(y_pred, dtype=tf.float64)
        outputs = tf.abs(y_true - y_pred)
        loss = tf.reduce_mean(outputs, axis=list(range(1, len(y_true.shape))))
        return loss


class mse(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        """ return the per sample mean squared error """
        outputs = tf.square(y_true - y_pred)
        loss = tf.reduce_mean(outputs, axis=list(range(1, len(y_true.shape))))

        return loss


class jpe(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        """ return the per sample joint position error """
        jp_loss = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))
        jp_loss = tf.reduce_mean(jp_loss, axis=list(range(1, len(jp_loss.shape))))

        return jp_loss

############################################################
# Calculate bone length error between two frames
###########################################################
def bone_length_error(x, x_hat, edges):
    mse = 0

    for i, j in edges:
        orignal_bone_len = np.sqrt(np.sum(np.square(x[i, :] - x[j, :])))
        predict_bone_len = np.sqrt(np.sum(np.square(x_hat[i, :] - x_hat[j, :])))

        mse += np.abs(orignal_bone_len - predict_bone_len)

    return mse / len(edges)


def calculate_all_bone_length_error(x, x_hat, max_val=None, min_val=None, edge='direct'):

    if max_val is not None and max_val is not None:
        x_unnorm = x * (max_val - min_val) + min_val
        x_hat_unnorm = x_hat * (max_val - min_val) + min_val
    else:
        x_unnorm, x_hat_unnorm = x, x_hat

    # load direct adjacent matrix
    if x.shape[2] == 22:
        graph = Graph(layout='shrec', strategy='spatial')
    else:
        graph = Graph(layout='nyu', strategy='spatial')

    if edge == 'direct':
        neighbor_link = graph.neighbor_direct_link
    else:
        neighbor_link = graph.neighbor_indirect_link

    # normal
    neighbor_link = np.array(neighbor_link)
    pos1 = x[:, :, neighbor_link[:, 0]]
    pos2 = x[:, :, neighbor_link[:, 1]]
    bone1 = np.sqrt(np.sum(np.square(pos1 - pos2), axis=-1))

    neighbor_link = np.array(neighbor_link)
    pos1 = x_hat[:, :, neighbor_link[:, 0]]
    pos2 = x_hat[:, :, neighbor_link[:, 1]]
    bone2 = np.sqrt(np.sum(np.square(pos1 - pos2), axis=-1))

    bone_len_error_1 = np.mean(np.square(bone1 - bone2))

    # unnorm
    neighbor_link = np.array(neighbor_link)
    pos1 = x_unnorm[:, :, neighbor_link[:, 0]]
    pos2 = x_unnorm[:, :, neighbor_link[:, 1]]
    bone1 = np.sqrt(np.sum(np.square(pos1 - pos2), axis=-1))

    neighbor_link = np.array(neighbor_link)
    pos1 = x_hat_unnorm[:, :, neighbor_link[:, 0]]
    pos2 = x_hat_unnorm[:, :, neighbor_link[:, 1]]
    bone2 = np.sqrt(np.sum(np.square(pos1 - pos2), axis=-1))

    bone_len_error_2 = np.mean(np.square(bone1 - bone2))

    return bone_len_error_1, bone_len_error_2


############################################################
# Calculate pose mse
###########################################################
def calculate_all_mse(x, x_hat, max_val=None, min_val=None):

    mse = np.mean(np.square(x - x_hat))

    if max_val is not None and min_val is not None:
        x_unnorm = x * (max_val - min_val) + min_val
        x_hat_unnorm = x_hat * (max_val - min_val) + min_val
        mse_unnorm = np.mean(np.square(x_unnorm - x_hat_unnorm))
        return mse, mse_unnorm

    return mse

def calculate_all_mpjpe(x, x_hat, max_val=None, min_val=None):
    mpjpe = np.mean(np.sqrt(np.sum(np.square(x - x_hat), axis=-1)))

    if max_val is not None and min_val is not None:
        x_unnorm = x * (max_val - min_val) + min_val
        x_hat_unnorm = x_hat * (max_val - min_val) + min_val
        mpjpe_unnorm = np.mean(np.sqrt(np.sum(np.square(x_unnorm - x_hat_unnorm), axis=-1)))
        return mpjpe, mpjpe_unnorm

    return mpjpe


############################################################
# Custom loss function
###########################################################
def pose_bone_loss(y_true, y_pred):
    l_1 = tf.keras.losses.MSE(y_true, y_pred)
    # l_2 = calculate_all_bone_length_error(y_true, y_pred, edge='direct')
    # l_3 = calculate_all_bone_length_error(y_true, y_pred, edge='indirect')

    return l_1


if __name__ == '__main__':
    x = np.random.randn(1856, 36, 36, 3)
    x_hat = np.random.randn(1856, 36, 36, 3)

    calculate_all_bone_length_error(x, x_hat)