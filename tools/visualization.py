import os

import pandas as pd
import seaborn as sns
import numpy as np

from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib import animation

from sklearn.metrics import mean_squared_error
from tools.nyu import show_joint_skeleton, jnt_color
from tools.metrics import calculate_all_mse

import matplotlib.pyplot as plt

# plt.style.use('ggplot')

plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['savefig.dpi'] = 300
plt.rcParams["savefig.format"] = 'pdf'
plt.rcParams['savefig.bbox'] = 'tight'

############################################################
# Visualization tool funcs
###########################################################
def str2float(arr):
    res = []
    for num in arr:
        if 'str' in str(type(num)):
            if '[' not in num:
                print(num)
                res.append(float(num.split('(')[1].split(',')[0]))
            else:
                res.append(float(num.split('numpy=')[1].split('>')[0]))
        else:
            res.append(num)

    return np.array(res)


# Plot model fit history loss & val_loss
def plot_history(history_excel_path):
    df_train = pd.read_excel(history_excel_path, sheet_name='train')
    df_test = pd.read_excel(history_excel_path, sheet_name='test')
    # loss and val_loss
    loss = df_train['loss'].values
    val_loss = df_test['error'].values

    deno_loss = df_train['deno_loss'].values
    val_deno_loss = df_test['deno_error'].values

    pred_loss = df_train['pred_loss'].values
    val_pred_loss = df_test['pred_error'].values

    loss = str2float(loss)
    val_loss = str2float(val_loss)
    deno_loss = str2float(deno_loss)
    val_deno_loss = str2float(val_deno_loss)
    pred_loss = str2float(pred_loss)
    val_pred_loss = str2float(val_pred_loss)

    if 'lr' in df_train.keys():
        # learning rate
        lr = df_train['lr']

        fig = plt.figure(figsize=(11, 4))
        host = host_subplot(121)  # row=1 col=1 first pic
        host.subplots_adjust(right=0.8)  # adjust the right boundary of the plot window
        par1 = host.twinx()  #
        ax1 = fig.add_subplot(121)
        ax1.plot(range(len(loss)), loss, '-', label="Training loss")
        ax1.plot(range(len(deno_loss)), deno_loss, '-', label="Training deno loss")
        ax1.plot(range(len(pred_loss)), pred_loss, '-', label="Training pred loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.set_title('Training Loss')

        ax1.grid()

        ax2 = fig.add_subplot(122)
        ax2.plot(range(len(val_loss)), val_loss, '-.', label="Validation loss")
        ax2.plot(range(len(val_deno_loss)), val_deno_loss, '-.', label="Validation deno loss")
        ax2.plot(range(len(val_pred_loss)), val_pred_loss, '-.', label="Validation pred loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.set_title('Validation Loss')

        ax2.grid()

        host = host_subplot(121)  # row=1 col=1 first pic
        plt.subplots_adjust(right=0.8)  # adjust the right boundary of the plot window
        par1 = host.twinx()  #

        # set labels
        host.set_xlabel("Epoch")
        host.set_ylabel("Loss")
        par1.set_ylabel("Learning rate")

        # plot curves
        p1, = host.plot(range(len(loss)), loss, '-', label="Training loss")
        host.plot(range(len(deno_loss)), deno_loss, '-', label="Training deno loss")
        host.plot(range(len(pred_loss)), pred_loss, '-', label="Training pred loss")

        host.plot(range(len(val_loss)), val_loss, '-.', label="Validation loss")
        host.plot(range(len(val_deno_loss)), val_deno_loss, '-.', label="Validation deno loss")
        host.plot(range(len(val_pred_loss)), val_pred_loss, '-.', label="Validation pred loss")

        p2, = par1.plot(range(len(lr)), lr, '-.', label="Learning rate")

        host.axis["left"].label.set_color(p1.get_color())
        par1.axis["right"].label.set_color(p2.get_color())
        host.spines["left"].set_edgecolor(p1.get_color())
        par1.spines["right"].set_edgecolor(p2.get_color())
        host.tick_params(axis='y', colors=p1.get_color())
        par1.tick_params(axis='y', colors=p2.get_color())

        host.legend()
        plt.title('Loss \& Learning rate')

        plt.grid()

        # save figure
        fig_name = '%s.pdf' % history_excel_path[:-5]
        fig.savefig(fig_name, dpi=300, bbox_inches='tight')
        print('Save history plot: %s' % fig_name)

    else:
        fig = plt.figure(figsize=(11, 4))
        ax1 = fig.add_subplot(121)
        ax1.plot(range(len(loss)), loss, '-', label="Training loss")
        ax1.plot(range(len(deno_loss)), deno_loss, '-', label="Training deno loss")
        ax1.plot(range(len(pred_loss)), pred_loss, '-', label="Training pred loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.set_title('Training Loss')

        ax1.grid()

        ax2 = fig.add_subplot(122)
        ax2.plot(range(len(val_loss)), val_loss, '-.', label="Validation loss")
        ax2.plot(range(len(val_deno_loss)), val_deno_loss, '-.', label="Validation deno loss")
        ax2.plot(range(len(val_pred_loss)), val_pred_loss, '-.', label="Validation pred loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.set_title('Validation Loss')

        ax2.grid()

        # save figure
        fig_name = '%s.pdf' % history_excel_path[:-5]
        fig.savefig(fig_name, dpi=300, bbox_inches='tight')
        print('Save history plot: %s' % fig_name)

    plt.close()


# Plot model fit history loss & val_loss
def plot_history_0(history_csv_path):
    df = pd.read_excel(history_csv_path)
    # loss and val_loss
    loss = df.iloc[:, 0].values
    val_loss = df.iloc[:, 1].values

    if df.shape[1] == 3:
        # learning rate
        lr = df.iloc[:, 2].values

        fig = plt.figure()
        host = host_subplot(111)  # row=1 col=1 first pic
        plt.subplots_adjust(right=0.8)  # adjust the right boundary of the plot window
        par1 = host.twinx()  #

        # set labels
        host.set_xlabel("Epoch")
        host.set_ylabel("Loss")
        par1.set_ylabel("Learning rate")

        # plot curves
        p1, = host.plot(range(len(loss)), loss, '-', label="Training loss")
        host.plot(range(len(val_loss)), val_loss, '--', label="Validation loss")
        p2, = par1.plot(range(len(lr)), lr, '-.', label="Learning rate")

        host.axis["left"].label.set_color(p1.get_color())
        par1.axis["right"].label.set_color(p2.get_color())
        host.spines["left"].set_edgecolor(p1.get_color())
        par1.spines["right"].set_edgecolor(p2.get_color())
        host.tick_params(axis='y', colors=p1.get_color())
        par1.tick_params(axis='y', colors=p2.get_color())

        host.legend()
        plt.title('Loss \& Learning rate')

    else:
        fig = plt.figure()

        plt.plot(range(len(loss)), loss, '-', label="Training loss")
        plt.plot(range(len(val_loss)), val_loss, '--', label="Validation loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title('Loss')

    plt.grid()
    # plt.show()

    # save figure
    fig_name = '%s.pdf' % history_csv_path[:-4]
    fig.savefig(fig_name, dpi=300)
    print('Save history plot: %s' % fig_name)

    plt.close()


############################################################
# plot test hat and test truth
###########################################################
def plot_save_pred(x_test, y_test, y_test_hat, fig_path, opt_idx):
    plt.rcParams['savefig.bbox'] = 'tight'

    x_test = x_test - x_test[:, :, :1, :]
    y_test = y_test - y_test[:, :, :1, :]
    y_test_hat = y_test_hat - y_test_hat[:, :, :1, :]

    if x_test.shape[2] == 22:
        from tools.shrec import show_joint_skeleton, jnt_color
    else:
        from tools.nyu import show_joint_skeleton, jnt_color
        print('-----')

    # predict & gt
    fig = plt.figure(figsize=(26, 6))
    num_frames = 5

    for i in range(num_frames):
        t = y_test.shape[1] - num_frames + i
        # t = 16 + i * 4
        print(t)

        # save predict
        ax_2 = plt.subplot(1, num_frames, i + 1, projection='3d')
        show_joint_skeleton(ax=ax_2, jnt_xyz=y_test_hat[opt_idx, t, :, :], jnt_color=jnt_color, ticks=False)

    # plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)

    fig_name = os.path.join(fig_path, f'{opt_idx}-pred.pdf')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=-0)

    print('Save file: %s' % fig_name)
    plt.close()

    # predict & gt
    fig = plt.figure(figsize=(26, 6))
    num_frames = 5

    for i in range(num_frames):
        t = y_test.shape[1] - num_frames + i
        # t = 16 + i * 4
        print(t)

        # save predict
        ax_2 = plt.subplot(1, num_frames, i + 1, projection='3d')
        show_joint_skeleton(ax=ax_2, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color, ticks=False)

    # plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)

    fig_name = os.path.join(fig_path, f'{opt_idx}-pred-gt.pdf')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=-0)

    print('Save file: %s' % fig_name)
    plt.close()


def plot_save_figs(x_test, y_test, y_test_hat, fig_path, opt_idx):
    plt.rcParams['savefig.bbox'] = 'tight'

    x_test = x_test - x_test[:, :, :1, :]
    y_test = y_test - y_test[:, :, :1, :]
    y_test_hat = y_test_hat - y_test_hat[:, :, :1, :]

    if x_test.shape[2] == 22:
        from data.SHREC.vis import show_joint_skeleton, jnt_color
    else:
        from data.NYU.nyu import show_joint_skeleton, jnt_color
        print('-----')

    # predict
    fig = plt.figure(figsize=(26, 6))
    num_frames = 6

    for i in range(num_frames):
        t = int(i * np.floor(x_test.shape[1] / (num_frames - 1)))
        # t = 16 + i * 4
        print(t)

        # save predict
        ax_2 = plt.subplot(1, num_frames, i + 1, projection='3d')
        show_joint_skeleton(ax=ax_2, jnt_xyz=y_test_hat[opt_idx, t, :, :], jnt_color=jnt_color, ticks=False)
        # show_joint_skeleton(ax=ax_2, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color, alpha=0.25)

    plt.subplots_adjust(wspace=0, hspace=0)

    fig_name = os.path.join(fig_path, f'{opt_idx}-predict.pdf')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=-0)

    print('Save file: %s' % fig_name)
    plt.close()

    # gt
    fig = plt.figure(figsize=(26, 6))
    num_frames = 6

    for i in range(num_frames):
        t = int(i * np.floor(x_test.shape[1] / (num_frames - 1)))
        # t = 16 + i * 4
        print(t)

        # save predict
        ax_2 = plt.subplot(1, num_frames, i + 1, projection='3d')
        # show_joint_skeleton(ax=ax_2, jnt_xyz=y_test_hat[opt_idx, t, :, :], jnt_color=jnt_color)
        show_joint_skeleton(ax=ax_2, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color, ticks=False)

        ax_2.axis('off')

    # plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)

    fig_name = os.path.join(fig_path, f'{opt_idx}-gt.pdf')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=-0)

    print('Save file: %s' % fig_name)
    plt.close()

    # input
    fig = plt.figure(figsize=(26, 6))
    num_frames = 6

    for i in range(num_frames):
        t = int(i * np.floor(x_test.shape[1] / (num_frames - 1)))
        # t = 16 + i * 4
        print(t)

        # save predict
        ax_2 = plt.subplot(1, num_frames, i + 1, projection='3d')
        # show_joint_skeleton(ax=ax_2, jnt_xyz=y_test_hat[opt_idx, t, :, :], jnt_color=jnt_color)
        show_joint_skeleton(ax=ax_2, jnt_xyz=x_test[opt_idx, t, :, :], jnt_color=jnt_color, ticks=False)

        ax_2.axis('off')

    # plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)

    fig_name = os.path.join(fig_path, f'{opt_idx}-input.pdf')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=-0)

    print('Save file: %s' % fig_name)
    plt.close()

    # predict & gt
    fig = plt.figure(figsize=(26, 6))
    num_frames = 6

    for i in range(num_frames):
        t = int(i * np.floor(x_test.shape[1] / (num_frames - 1)))
        # t = 16 + i * 4
        print(t)

        # save predict
        ax_2 = plt.subplot(1, num_frames, i + 1, projection='3d')
        show_joint_skeleton(ax=ax_2, jnt_xyz=y_test_hat[opt_idx, t, :, :], jnt_color=jnt_color, ticks=False)
        show_joint_skeleton(ax=ax_2, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color, alpha=0.25, ticks=False)

    # plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)

    fig_name = os.path.join(fig_path, f'{opt_idx}-predict_gt.pdf')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=-0)

    print('Save file: %s' % fig_name)
    plt.close()

def plot_save_single_video(x_test, y_test, y_test_hat, fig_path, opt_idx, mode='gt'):
    plt.rcParams['savefig.bbox'] = None

    if x_test.shape[2] == 22:
        from tools.shrec import show_joint_skeleton, jnt_color
    else:
        from tools.nyu import show_joint_skeleton, jnt_color

    num_frames_show = y_test.shape[1]

    if num_frames_show == x_test.shape[1]:
        video_name = os.path.join(fig_path, f'{opt_idx}-{mode}.mp4')
    else:
        video_name = os.path.join(fig_path, f'{opt_idx}-{mode}_dp.mp4')

    # made the video of result
    fig = plt.figure(figsize=(12, 5)) if mode == 'all' else plt.figure(figsize=(5, 5))

    # judge the fig path
    fig_path = os.path.join(fig_path, 'single')
    os.makedirs(fig_path, exist_ok=True)

    def update(t):
        plt.axis('off')
        plt.clf()

        tmp = np.concatenate([y_test[opt_idx], y_test[opt_idx]], axis=0)
        # tmp = np.concatenate([y_test[opt_idx, t, :, :], y_test[opt_idx, t, :, :]], axis=0)

        if mode == 'all':
            ax_1 = plt.subplot(131, projection='3d')
            ax_2 = plt.subplot(132, projection='3d')
            ax_3 = plt.subplot(133, projection='3d')

            if t < x_test.shape[1]:
                show_joint_skeleton(ax=ax_1, jnt_xyz=x_test[opt_idx, t, :, :], jnt_color=jnt_color)
            show_joint_skeleton(ax=ax_1, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color, alpha=0.25)

            if t < x_test.shape[1]:
                ax_1.set_title(
                    'Input ($\mathrm{MSE_{pose}}$ = %.4f)' % (
                        mean_squared_error(x_test[opt_idx, t, :, :], y_test[opt_idx, t, :, :])))

            show_joint_skeleton(ax=ax_2, jnt_xyz=y_test_hat[opt_idx, t, :, :], jnt_color=jnt_color)
            show_joint_skeleton(ax=ax_2, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color, alpha=0.25)
            ax_2.set_title(
                'Output ($\mathrm{MSE_{pose}}$ = %.4f)' % (
                    mean_squared_error(y_test_hat[opt_idx, t, :, :], y_test[opt_idx, t, :, :])))

            show_joint_skeleton(ax=ax_3, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color)
            ax_3.set_title('Ground truth')

            ax_2.set_xlim([tmp[..., 0].min(), tmp[..., 0].max()])
            ax_2.set_ylim([tmp[..., 1].min(), tmp[..., 1].max()])
            ax_2.set_zlim([tmp[..., 2].min(), tmp[..., 2].max()])

            ax_3.set_xlim([tmp[..., 0].min(), tmp[..., 0].max()])
            ax_3.set_ylim([tmp[..., 1].min(), tmp[..., 1].max()])
            ax_3.set_zlim([tmp[..., 2].min(), tmp[..., 2].max()])

        elif mode == 'gt':
            ax_1 = plt.subplot(111, projection='3d')

            show_joint_skeleton(ax=ax_1, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color)
            ax_1.set_title('Ground truth')

        elif mode == 'input':
            ax_1 = plt.subplot(111, projection='3d')

            if t < x_test.shape[1]:
                show_joint_skeleton(ax=ax_1, jnt_xyz=x_test[opt_idx, t, :, :], jnt_color=jnt_color)
            show_joint_skeleton(ax=ax_1, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color, alpha=0.25)

            if t < x_test.shape[1]:
                ax_1.set_title(
                    'Input ($\mathrm{MSE_{pose}}$ = %.4f)' % (
                        mean_squared_error(x_test[opt_idx, t, :, :], y_test[opt_idx, t, :, :])))
            else:
                ax_1.set_title(f'Prediction phase ({t}-th)', color='red')

        elif mode == 'output':
            ax_1 = plt.subplot(111, projection='3d')

            show_joint_skeleton(ax=ax_1, jnt_xyz=y_test_hat[opt_idx, t, :, :], jnt_color=jnt_color)
            show_joint_skeleton(ax=ax_1, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color, alpha=0.25)
            ax_1.set_title(
                'Output ($\mathrm{MSE_{pose}}$ = %.4f)' % (
                    mean_squared_error(y_test_hat[opt_idx, t, :, :], y_test[opt_idx, t, :, :])))

        ax_1.set_xlim([tmp[..., 0].min(), tmp[..., 0].max()])
        ax_1.set_ylim([tmp[..., 1].min(), tmp[..., 1].max()])
        ax_1.set_zlim([tmp[..., 2].min(), tmp[..., 2].max()])

    ani = animation.FuncAnimation(fig=fig, func=update, frames=num_frames_show)
    ani.save(filename=video_name, writer='ffmpeg', fps=10)
    plt.close()
    print('Save video: %s' % video_name)

def plot_save_video(x_test, y_test, y_test_hat, fig_path, opt_idx):

    if x_test.shape[2] == 22:
        from tools.shrec import show_joint_skeleton, jnt_color
    else:
        from tools.nyu import show_joint_skeleton, jnt_color

    video_name = os.path.join(fig_path, '%d-tremor_result.mp4' % opt_idx)

    # made the video of result
    fig = plt.figure(figsize=(12, 5))

    # judge the fig path
    fig_path = os.path.join(fig_path, 'single')
    os.makedirs(fig_path, exist_ok=True)

    def update(t):
        plt.axis('off')
        plt.clf()
        ax_1 = plt.subplot(131, projection='3d')
        ax_2 = plt.subplot(132, projection='3d')
        ax_3 = plt.subplot(133, projection='3d')

        if t < x_test.shape[1]:
            show_joint_skeleton(ax=ax_1, jnt_xyz=x_test[opt_idx, t, :, :], jnt_color=jnt_color)
        show_joint_skeleton(ax=ax_1, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color, alpha=0.25)

        if t < x_test.shape[1]:
            ax_1.set_title(
                'Input ($\mathrm{MSE_{pose}}$ = %.4f)' % (
                    mean_squared_error(x_test[opt_idx, t, :, :], y_test[opt_idx, t, :, :])))

        show_joint_skeleton(ax=ax_2, jnt_xyz=y_test_hat[opt_idx, t, :, :], jnt_color=jnt_color)
        show_joint_skeleton(ax=ax_2, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color, alpha=0.25)
        ax_2.set_title(
            'Output ($\mathrm{MSE_{pose}}$ = %.4f)' % (
                mean_squared_error(y_test_hat[opt_idx, t, :, :], y_test[opt_idx, t, :, :])))

        show_joint_skeleton(ax=ax_3, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color)
        ax_3.set_title('Ground truth')

        tmp = np.concatenate([y_test[opt_idx], y_test[opt_idx]], axis=0)
        # tmp = np.concatenate([y_test[opt_idx, t, :, :], y_test[opt_idx, t, :, :]], axis=0)
        ax_1.set_xlim([tmp[..., 0].min(), tmp[..., 0].max()])
        ax_1.set_ylim([tmp[..., 1].min(), tmp[..., 1].max()])
        ax_1.set_zlim([tmp[..., 2].min(), tmp[..., 2].max()])

        ax_2.set_xlim([tmp[..., 0].min(), tmp[..., 0].max()])
        ax_2.set_ylim([tmp[..., 1].min(), tmp[..., 1].max()])
        ax_2.set_zlim([tmp[..., 2].min(), tmp[..., 2].max()])

        ax_3.set_xlim([tmp[..., 0].min(), tmp[..., 0].max()])
        ax_3.set_ylim([tmp[..., 1].min(), tmp[..., 1].max()])
        ax_3.set_zlim([tmp[..., 2].min(), tmp[..., 2].max()])

        fig_name = os.path.join(fig_path, '%d-%d.pdf' % (opt_idx, t))
        fig.savefig(fig_name, bbox_inches='tight')
        print('Save file: %s' % fig_name)

        # ax_1.axis('off')
        # ax_2.axis('off')
        # ax_3.axis('off')

    ani = animation.FuncAnimation(fig=fig, func=update, frames=y_test.shape[1])
    ani.save(filename=video_name, writer='ffmpeg', fps=10)
    plt.close()
    print('Save video: %s' % video_name)

    # save every fig in single plot
    for t in range(0):
    # for t in range(y_test.shape[1]):
        # save input
        fig = plt.figure()
        ax_1 = plt.subplot(111, projection='3d')
        if t < x_test.shape[1]:
            show_joint_skeleton(ax=ax_1, jnt_xyz=x_test[opt_idx, t, :, :], jnt_color=jnt_color)
        show_joint_skeleton(ax=ax_1, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color, alpha=0.25)

        if t < x_test.shape[1]:
            ax_1.set_title('Input ($\mathrm{MSE_{pose}}$ = %.4f)' % (
                mean_squared_error(x_test[opt_idx, t, :, :], y_test[opt_idx, t, :, :])))

        fig_name = os.path.join(fig_path, '%d-%d-input.pdf' % (opt_idx, t))
        fig.savefig(fig_name, bbox_inches='tight')

        print('Save file: %s' % fig_name)
        plt.close()

        # save predict
        fig = plt.figure()
        ax_2 = plt.subplot(111, projection='3d')

        show_joint_skeleton(ax=ax_2, jnt_xyz=y_test_hat[opt_idx, t, :, :], jnt_color=jnt_color)
        show_joint_skeleton(ax=ax_2, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color, alpha=0.25)
        ax_2.set_title(
            'Output ($\mathrm{MSE_{pose}}$ = %.4f)' % (
                mean_squared_error(y_test_hat[opt_idx, t, :, :], y_test[opt_idx, t, :, :])))

        fig_name = os.path.join(fig_path, '%d-%d-predict.pdf' % (opt_idx, t))
        fig.savefig(fig_name, bbox_inches='tight')

        print('Save file: %s' % fig_name)
        plt.close()

        # save ground truth
        fig = plt.figure()
        ax_3 = plt.subplot(111, projection='3d')

        show_joint_skeleton(ax=ax_3, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color)
        ax_3.set_title('Ground truth')

        fig_name = os.path.join(fig_path, '%d-%d-gt.pdf' % (opt_idx, t))
        fig.savefig(fig_name, bbox_inches='tight')

        print('Save file: %s' % fig_name)
        plt.close()


############################################################
# plot and save the trajectory
###########################################################
def plot_save_trajectory(x_test, y_test, y_test_hat, fig_path, opt_idx):
    # plot x-axis position of index finger tip
    index_finger_idx = 18
    num_axis = 3
    axis_name = ['x', 'y', 'z']
    x = [i for i in range(y_test.shape[1])]

    # make folder
    fig_axis_path = os.path.join(fig_path, 'axis')
    if os.path.exists(fig_axis_path) == False:
        os.makedirs(fig_axis_path)

    # save entire figure
    fig = plt.figure(figsize=(4.5 * num_axis, 3))

    for num in range(num_axis):
        y_test_index_fingtip = y_test[opt_idx, :, index_finger_idx, num]
        y_test_hat_index_fingtip = y_test_hat[opt_idx, :, index_finger_idx, num]
        x_test_index_fingtip = x_test[opt_idx, :, index_finger_idx, num]

        ax_1 = plt.subplot(1, num_axis, num + 1)
        ax_1.plot(x, y_test_index_fingtip, label='Ground truth')
        ax_1.plot(x_test_index_fingtip, label='Input')
        ax_1.plot(x, y_test_hat_index_fingtip, '--', label='Output')

        ax_1.set_xlabel('Frame')
        ax_1.set_ylabel('$%s$-axis' % axis_name[num])
        # ax_1.set_title('Trajectory of $%s$-axis' % axis_name[num])

        ax_1.legend()

    fig_name = os.path.join(fig_path, 'axis/%d-axis.pdf' % opt_idx)
    if os.path.exists(os.path.dirname(fig_name)) == False:
        os.makedirs(os.path.dirname(fig_name))

    fig.savefig(fig_name, bbox_inches='tight')
    print('Save trajectory: %s' % fig_name)

    # save entire figure separately
    for num in range(num_axis):
        fig = plt.figure()

        y_test_index_fingtip = y_test[opt_idx, :, index_finger_idx, num]
        y_test_hat_index_fingtip = y_test_hat[opt_idx, :, index_finger_idx, num]
        x_test_index_fingtip = x_test[opt_idx, :, index_finger_idx, num]

        ax_1 = plt.subplot(1, 1, 1)
        ax_1.plot(x, y_test_index_fingtip, label='Ground truth')
        ax_1.plot(x_test_index_fingtip, label='Input')
        ax_1.plot(x, y_test_hat_index_fingtip, '--', label='Output')

        ax_1.set_xlabel('Frame')
        ax_1.set_ylabel('$%s$-axis' % axis_name[num])
        # ax_1.set_title('Trajectory of $%s$-axis' % axis_name[num])

        ax_1.legend()

        fig_name = os.path.join(fig_path, 'axis/%d-axis-%s.pdf' % (opt_idx, axis_name[num]))
        if os.path.exists(os.path.dirname(fig_name)) == False:
            os.makedirs(os.path.dirname(fig_name))

        fig.savefig(fig_name, bbox_inches='tight')
        print('Save trajectory: %s' % fig_name)

        plt.close()

    # save single figure
    for num in range(num_axis):
        y_test_index_fingtip = y_test[opt_idx, :, index_finger_idx, num]
        y_test_hat_index_fingtip = y_test_hat[opt_idx, :, index_finger_idx, num]
        x_test_index_fingtip = x_test[opt_idx, :, index_finger_idx, num]

        # Ground truth
        fig = plt.figure()

        plt.plot(x, y_test_index_fingtip, 'r', label='Ground truth')

        plt.xlabel('Frame')
        plt.ylabel('$%s$-axis' % axis_name[num])
        # plt.set_title('Trajectory of $%s$-axis' % axis_name[num])

        plt.legend()

        fig_name = os.path.join(fig_path, 'axis/%d-axis-%s-gt.pdf' % (opt_idx, axis_name[num]))
        fig.savefig(fig_name, bbox_inches='tight')
        print('Save trajectory: %s' % fig_name)

        # Input
        fig = plt.figure()

        plt.plot(x_test_index_fingtip, label='Input')
        plt.plot(x, y_test_index_fingtip, label='Ground truth')
        if len(x) == len(x_test_index_fingtip):
            plt.fill_between(x, x_test_index_fingtip, y_test_index_fingtip, facecolor='g', alpha=0.25)

        plt.xlabel('Frame')
        plt.ylabel('$%s$-axis' % axis_name[num])
        # plt.title('Trajectory of $%s$-axis' % axis_name[num])

        plt.legend()

        fig_name = os.path.join(fig_path, 'axis/%d-axis-%s-input.pdf' % (opt_idx, axis_name[num]))
        fig.savefig(fig_name, bbox_inches='tight')
        print('Save trajectory: %s' % fig_name)

        # predict
        fig = plt.figure()

        plt.plot(x, y_test_hat_index_fingtip, '--', label='Output')
        plt.plot(x, y_test_index_fingtip, label='Ground truth')
        plt.fill_between(x, y_test_index_fingtip, y_test_hat_index_fingtip, facecolor='b', alpha=0.25)

        plt.xlabel('Frame')
        plt.ylabel('$%s$-axis' % axis_name[num])
        # plt.title('Trajectory of $%s$-axis' % axis_name[num])

        plt.legend()

        fig_name = os.path.join(fig_path, 'axis/%d-axis-%s-predict.pdf' % (opt_idx, axis_name[num]))
        fig.savefig(fig_name, bbox_inches='tight')
        print('Save trajectory: %s' % fig_name)

        plt.close()


############################################################
# plot and save the learnable adjacency matrix
###########################################################
def plot_adj(adj, fig_path, num=0):
    fig = plt.figure(figsize=(6, 4.5))
    ax = plt.subplot(111)

    sns.heatmap(adj, fmt="d", cmap=plt.cm.bwr, ax=ax)

    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360, horizontalalignment='right')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')

    ax.set_xlabel('Hand joint')
    ax.set_ylabel('Hand joint')
    if num != 0:
        # ax.set_title('$\mathbf{A}_%d \odot \mathbf{M}_%d$' % (num, num))
        fig_name = os.path.join(fig_path, 'adj_heatmap_%d.pdf' % num)
    else:
        # ax.set_title('$\sum_k^K (\mathbf{A}_k \odot \mathbf{M}_k)$')
        fig_name = os.path.join(fig_path, 'adj_heatmap.pdf')

    fig.savefig(fig_name, bbox_inches='tight')

    print('Save heatmap: %s' % fig_name)


############################################################
# plot and save error curve
###########################################################
def plot_error(x_test, y_test, y_test_hat, fig_path, opt_idx, max_val, min_val):
    y_test_opt = y_test[opt_idx]
    y_test_hat_opt = y_test_hat[opt_idx]
    x_test_opt = x_test[opt_idx]
    max_val_opt = max_val[opt_idx]
    min_val_opt = min_val[opt_idx]

    err_input, err_unnorm_input = [], []
    err_test, err_unnorm_test = [], []
    for i in range(x_test.shape[1]):
        err, err_unnorm = calculate_all_mse(x_test_opt[i], y_test_opt[i], max_val_opt[0], min_val_opt[0])
        err_input.append(err)
        err_unnorm_input.append(err_unnorm)

    for i in range(y_test.shape[1]):
        err, err_unnorm = calculate_all_mse(y_test_hat_opt[i], y_test_opt[i], max_val_opt[0], min_val_opt[0])
        err_test.append(err)
        err_unnorm_test.append(err_unnorm)

    # plot
    x = [i for i in range(y_test.shape[1])]

    def plot_err(input, test, affix=None):
        plt.figure()
        plt.plot(input, label='Input')
        plt.plot(x, test, '--', label='Output')

        plt.xlabel('Frame')
        plt.ylabel('$E_{\mathrm{pos}}$')
        # plt.title('$\mathrm{MPJPE_{pose}}$ Curve')

        plt.legend()
        # plt.ylim([-20, 1050])
        
        fig_name = os.path.join(fig_path, '%d-error_curve.pdf' % (opt_idx)) if affix is None else \
            os.path.join(fig_path, '%d-%s_error_curve.pdf' % (opt_idx, affix))
        plt.savefig(fig_name, bbox_inches='tight')

        print('Save error curve plot: %s' % fig_name)

    plot_err(err_input, err_test)
    plot_err(err_unnorm_input, err_unnorm_test, 'unnorm')
