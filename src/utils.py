import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

def get_date_from_file_name(filename):

    date_infos = [int(val[1:]) for val in filename.split('/')[-1].split('.')[0].split('-')]
    return date_infos

def filter_one_week_over_two_for_eval(idx):

    samples_per_week = 12*24*7
    return (idx // samples_per_week) % 2

def missing_file_in_sequence(files_names):

    for k in range(len(files_names)-1):
        month_1, day_1, hour_1, min_1 = get_date_from_file_name(files_names[k])[1:]
        month_2, day_2, hour_2, min_2 = get_date_from_file_name(files_names[k+1])[1:]

        if (min_1 + 5) % 60 != min_2:
            #print("Min gap : ", files_names, "\n")
            return True
        if (hour_1 + 1) % 24 != hour_2 and (min_1 == 55 and min_2 == 0):
            #print("Hour gap : ", files_names, "\n")
            return True
        if day_1 != day_2 and day_1 + 1 != day_2 and not ((day_1 == 30 and day_2 == 1) or (day_1 == 31 and day_2 == 1) or (month_1 == 2 and (day_1 == 28 or day_1 == 29) and day_2 == 1)):
            #print("Day gap : ", files_names, "\n")
            return True


    return False

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
    return torch.where((0 <= target) & (target < 0.1), 1., 0.) \
    + torch.where((0.1 <= target) & (target < 1), 2., 0.) \
    + torch.where((1 <= target) & (target < 2.5), 4., 0.) \
    + torch.where((2.5 <= target), 8., 0.)

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
        plot_output_gt(pred[0], target, input, k, output_dir)


def plot_output_gt(output, target, input, index, output_dir):

    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    input = input.cpu().detach().numpy()

    #fig, axs = plt.subplots(2, output.shape[0], figsize=(15, 6))
    fig, axs = plt.subplots(3, 5, figsize=(15, 6))
    for k in range(5):
        axs[0][k].imshow(input[7+k], cmap='gray')
        axs[0][k].title.set_text('Input at t - {}'.format(5*(4-k)))
    #for k in range(output.shape[0]):
    for k in range(5):
        axs[1][k].imshow(output[2*k], cmap='gray')
        axs[2][k].imshow(target[2*k], cmap='gray')
        axs[1][k].title.set_text('Pred at t + {}'.format(5*(2*k+2)))
        axs[2][k].title.set_text('GT at t + {}'.format(5*(2*k+2)))
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

    metric_score = (conf_mat['true_positive'][time_step] + conf_mat['false_positive'][time_step]) / (
                                conf_mat['true_positive'][time_step] + conf_mat['false_negative'][time_step])

    return round(metric_score, 3)


def compute_csi_score(conf_mat, time_step):

    metric_score = conf_mat['true_positive'][time_step] / (
                                conf_mat['true_positive'][time_step] + conf_mat['false_negative'][time_step] + conf_mat['false_positive'][time_step])

    return round(metric_score, 3)


class CustomSampler(Sampler):
    """
    Draws all element of indices one time and in the given order
    """

    def __init__(self, alist, dataset):
        """
        Parameters
        ----------
        alist : list
            Composed of True False for keep or reject position.
        """
        self.__alist___ = alist
        self.indices = [k for k in range(len(alist)) if alist[k]]
        self.dataset = dataset

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def indices_except_undefined_sampler(dataset):

    samples_weight = []

    for i in tqdm(range(len(dataset))):
        condition_meet = True

        dataset_item_i = dataset.__getitem__(i)
        inputs, targets = dataset_item_i["input"], dataset_item_i["target"]

        # If the last image of the input sequence contains no rain, we don't take into account the sequence
        if torch.max(inputs[-1]).item() == 0:
            condition_meet = False

        if torch.min(inputs).item() < 0 or torch.min(targets).item() < 0:
            condition_meet = False

        if condition_meet:
            samples_weight.append(i)

    return samples_weight
