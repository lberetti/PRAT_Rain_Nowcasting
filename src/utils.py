import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def get_date_from_file_name(filename):

    date_infos = [int(val[1:]) for val in filename.split('/')[-1].split('.')[0].split('-')]
    return date_infos


def weighted_mse_loss(output, target, weight_mask):
    return torch.sum(torch.multiply(weight_mask, (output - target) ** 2))

def weighted_mae_loss(output, target, weight_mask):
    return torch.sum(torch.multiply(weight_mask, torch.abs(output - target)))

def compute_weight_mask(target):

    """
    threshold = [0, 2, 5, 10, 30, 1000]
    weights = [1., 2., 5., 10., 30.]

    # To solve
    mask = torch.ones(target.size(), dtype=torch.double).cuda()

    for k in range(len(weights)):
        mask = torch.where((threshold[k] <= target) & (target < threshold[k+1]), weights[k], mask)
    """

    ### Fix for small gpu below
    return torch.where((0 <= target) & (target < 2), 1., 0.) \
    + torch.where((2 <= target) & (target < 5), 2., 0.) \
    + torch.where((5 <= target) & (target < 10), 5., 0.) \
    + torch.where((10 <= target) & (target < 30), 10., 0.) \
    + torch.where((30 <= target), 30., 0.)

    #return mask


def compute_confusion_matrix_on_batch(output, target, threshold):

    # Computing this tensor helps to compute the confusion matrix much more easily.
    difference = 2*torch.where(output >= threshold, 1.0, 0.0) - torch.where(target >= threshold, 1.0, 0.0)
    # True positive
    true_positive = torch.sum(difference==1, dim=(0, 2, 3)).tolist()
    # True negative
    true_negative = torch.sum(difference==0, dim=(0, 2, 3)).tolist()
    # False positive
    false_positive = torch.sum(difference==2, dim=(0, 2, 3)).tolist()
    # False negative
    false_negative = torch.sum(difference==-1, dim=(0, 2, 3)).tolist()

    return {'true_positive' : true_positive, 'true_negative' : true_negative,
    'false_positive' : false_positive, 'false_negative' : false_negative}


def save_pred_images(network, dataset, n_plots, output_dir, device):

    for k in range(n_plots):
        idx = np.random.randint(0, len(dataset))
        data = dataset.__getitem__(idx)
        input = data['input']
        target = data['target']
        pred = network.forward(input.unsqueeze(0).to(device=device))
        plot_output_gt(pred[0], target, k, output_dir)


def plot_output_gt(output, target, index, output_dir):

    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    fig, axs = plt.subplots(2, output.shape[0], figsize=(15, 6))
    for k in range(output.shape[0]):
        axs[0][k].imshow(output[k], cmap='gray')
        axs[1][k].imshow(target[k], cmap='gray')
        axs[0][k].title.set_text('Pred at t + {}'.format(5*(k+1)))
        axs[1][k].title.set_text('GT at t + {}'.format(5*(k+1)))
    plt.savefig(output_dir + str(index))


def add_confusion_matrix_on_batch(confusion_matrix, confusion_matrix_on_batch, threshold):

    for key in confusion_matrix_on_batch:
        for time_step in range(len(confusion_matrix_on_batch[key])):
            confusion_matrix[str(threshold)][key][time_step] += confusion_matrix_on_batch[key][time_step]
    return confusion_matrix


def model_evaluation(confusion_matrix):
    scores = {}
    for thresh_key in confusion_matrix:
        scores[thresh_key] = {}
        scores[thresh_key]['f1_score'] = {}
        scores[thresh_key]['ts_score'] = {}
        scores[thresh_key]['bias_score'] = {}
        #scores[thresh_key]['CSI_score'] = {}
        for time_step in range(len(confusion_matrix[thresh_key]['true_positive'])):
            scores[thresh_key]['f1_score']["t+"+str(time_step+1)] = compute_f1_score(confusion_matrix[thresh_key], time_step)
            scores[thresh_key]['ts_score']["t+"+str(time_step+1)] = compute_ts_score(confusion_matrix[thresh_key], time_step)
            scores[thresh_key]['bias_score']["t+"+str(time_step+1)] = compute_bias_score(confusion_matrix[thresh_key], time_step)
            #scores[thresh_key]['CSI_score'][time_step] = compute_csi_score(confusion_matrix[thresh_key], time_step)
    return scores

def compute_f1_score(conf_mat, time_step):

    precision = conf_mat['true_positive'][time_step] / (
                                conf_mat['true_positive'][time_step] + conf_mat['false_positive'][time_step])
    recall = conf_mat['true_positive'][time_step] / (
                                conf_mat['true_positive'][time_step] + conf_mat['false_negative'][time_step])
    metric_score = 2 * precision * recall / (precision + recall)

    return round(metric_score, 3)


def compute_ts_score(conf_mat, time_step):

    metric_score = conf_mat['true_positive'][time_step] / (
                        conf_mat['true_positive'][time_step] + conf_mat['false_positive'][time_step] + conf_mat['false_negative'][time_step])

    return round(metric_score, 3)


def compute_bias_score(conf_mat, time_step):

    metric_score = conf_mat['true_positive'][time_step] + conf_mat['false_positive'][time_step] / (
                                conf_mat['true_positive'][time_step] + conf_mat['false_negative'][time_step])

    return round(metric_score, 3)


def compute_csi_score(conf_mat, time_step):

    metric_score = conf_mat['true_positive'][time_step] / (
                                conf_mat['true_positive'][time_step] + conf_mat['false_negative'][time_step] + conf_mat['false_positive'][time_step])

    return round(metric_score, 3)
