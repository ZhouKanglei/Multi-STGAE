
import multiprocessing

from tqdm import tqdm

from tools.metrics import *
from tools.misc import *
from tools.visualization import *

def data_norm(data, xyz_max, xyz_min):
    data_norm = data.copy()
    data_norm[:, :, :, 0] = (data[:, :, :, 0] - xyz_min[0]) / (xyz_max[0] - xyz_min[0])
    data_norm[:, :, :, 1] = (data[:, :, :, 1] - xyz_min[1]) / (xyz_max[1] - xyz_min[1])
    data_norm[:, :, :, 2] = (data[:, :, :, 2] - xyz_min[2]) / (xyz_max[2] - xyz_min[2])

    return data_norm

def de_norm(data_norm, xyz_max, xyz_min):
    data = data_norm.copy()
    data[:, :, :, 0] = data_norm[:, :, :, 0] * (xyz_max[0] - xyz_min[0]) + xyz_min[0]
    data[:, :, :, 1] = data_norm[:, :, :, 1] * (xyz_max[1] - xyz_min[1]) + xyz_min[1]
    data[:, :, :, 2] = data_norm[:, :, :, 2] * (xyz_max[2] - xyz_min[2]) + xyz_min[2]

    return data

############################################################
# search file from folder
###########################################################
def search_file_from_folder(folder_path, file_key):
    files_selected = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.index(file_key) != -1:
                files_selected.append(file)

    return files_selected


############################################################
# search optimal model and delete the extra model
###########################################################
def search_acceptable_model(y_test, y_test_hat, max_val=None, min_val=None):
    # search an acceptable loss
    error_all = np.empty(shape=(y_test.shape[0], 1))
    error_unnorm_all = np.empty(shape=(y_test.shape[0], 1))

    p = multiprocessing.Pool(64)
    args = [i for i in range(y_test.shape[0])]
    pbar = tqdm(range(y_test.shape[0]))
    for i in p.imap(get_idx, args):
        pbar.update()

        error, error_unnorm,  = calculate_all_mpjpe(y_test[i], y_test_hat[i], max_val[i], min_val[i])
        error_all[i] = error
        error_unnorm_all[i] = error_unnorm

        pbar.set_description('Loss %d: %.4f' % (i, error))

        if error < 0:
            pbar.close()
            break

    p.close()
    p.join()

    error = min(error_all)
    i = list(error_all).index(error)

    return i, error, error_all, error_unnorm_all


############################################################
#  Plot training history
###########################################################
def plot_hat(x_test, y_test, y_test_hat, fig_path, log):
    i, loss, loss_all, _ = search_acceptable_model(y_test, y_test_hat)
    # judge
    if i == x_test.shape[0] - 1:
        opt_idx = i
    else:
        opt_idx = i

    log.info('Acceptable loss %d: %.6fe-4' % (i, loss * 1e4))
    log.info('Selected %d, loss = %.6fe-4' % (opt_idx, loss_all[opt_idx] * 1e4))

    fig_path = os.path.join(fig_path, str(opt_idx))
    if os.path.exists(fig_path) == False:
        os.mkdir(fig_path)

    # delete the remaining files
    video_name = os.path.join(fig_path, '%d-tremor_result.mp4' % opt_idx)
    if os.path.exists(video_name) == False:
        pass

    # save video and figs of result
    plot_save_video(x_test, y_test, y_test_hat, fig_path, opt_idx)

    # save trajectory
    plot_save_trajectory(x_test, y_test, y_test_hat, fig_path, opt_idx)

    # save mse error
    plot_error(x_test, y_test, y_test_hat, fig_path, opt_idx)


############################################################
# Dealing with the test phase: saving figs and search optimal model
###########################################################
def deal_learnable_adj(A, fig_path, start=0):
    num_show = 30 if A.shape[1] != 22 else 22
    for A_num in range(A.shape[0]):
        A_ = A[A_num, :num_show, :num_show]
        plot_adj(adj=A_, fig_path=fig_path, num=start + A_num + 1)

    # only using the top 30 joints of NYU hand model
    A_ = np.zeros(shape=(num_show, num_show))
    for A_num in range(A.shape[0]):
        A_ = A_ + A[A_num, :num_show, :num_show]
    plot_adj(adj=A_, fig_path=fig_path, num=start + 0)
