import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
from PIL import Image

def get_date_from_file_name(filename):

    date_infos = [int(val[1:]) for val in filename.split('/')[-1].split('.')[0].split('-')]
    return date_infos

def filter_year(filename, dataset):

    date_infos = get_date_from_file_name(filename)
    year = date_infos[0]

    if dataset == 'train':
        return year == 2016 or year == 2017
    elif dataset == 'valid' or dataset == 'test':
        return year == 2018


def keep_wind_when_rainmap_exists(rainmap_list, U_wind_list, V_wind_list):

    rain_map_new_L, U_wind_new_L, V_wind_new_L = [], [], []

    for k in tqdm(range(len(rainmap_list))):
        if rainmap_list[k] in U_wind_list and rainmap_list[k] in V_wind_list:

            # All repertories have the same file names.
            rain_map_new_L.append(rainmap_list[k])
            U_wind_new_L.append(rainmap_list[k])
            V_wind_new_L.append(rainmap_list[k])

    return rain_map_new_L, U_wind_new_L, V_wind_new_L


def sanity_check(rain_files_names, U_wind_files_names, V_wind_files_names):

    """if not (len(rain_files_names) == len(U_wind_files_names) and len(rain_files_names) == len(V_wind_files_names)):
         print("Error : dimension mismatch")
         return"""

    print(len(rain_files_names))
    print(len(U_wind_files_names))
    print(len(V_wind_files_names))

    for k in range(len(rain_files_names)):
        date_infos_rain = get_date_from_file_name(rain_files_names[k])
        U_wind_infos_rain = get_date_from_file_name(U_wind_files_names[k])
        V_wind_infos_rain = get_date_from_file_name(V_wind_files_names[k])

        if not (date_infos_rain == U_wind_infos_rain and date_infos_rain == V_wind_infos_rain):
            print("Error")
            print(rain_files_names[k])
            print(U_wind_files_names[k])
            print(V_wind_files_names[k])
            return


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
    + torch.where((0.1 <= target) & (target < 1), 1.5, 0.) \
    + torch.where((1 <= target) & (target < 2.5), 2., 0.) \
    + torch.where((2.5 <= target), 3., 0.)

    #return mask


def compute_confusion_matrix_on_batch(output, target, threshold):

    # Case of a RNN
    if len(target.shape) == 5:
        # sum on batch_size, channels(1), height, width
        dimension_reshape = (0, 2, 3, 4)
    else:
        dimension_reshape = (0, 2, 3)

    # Computing this tensor helps to compute the confusion matrix much more easily.
    difference = 2*torch.where(output >= threshold, 1.0, 0.0) - torch.where(target >= threshold, 1.0, 0.0)
    # True positive
    true_positive = torch.sum(difference==1, dim=dimension_reshape).tolist()
    # True negative
    true_negative = torch.sum(difference==0, dim=dimension_reshape).tolist()
    # False positive
    false_positive = torch.sum(difference==2, dim=dimension_reshape).tolist()
    # False negative
    false_negative = torch.sum(difference==-1, dim=dimension_reshape).tolist()

    return {'true_positive' : true_positive, 'true_negative' : true_negative,
    'false_positive' : false_positive, 'false_negative' : false_negative}


def plot_output_gt(output, target, input, index, output_dir):

    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    input = input.cpu().detach().numpy()

    if len(target.shape) == 4:
        output = output.squeeze(1)
        target = target.squeeze(1)
        input = input.squeeze(1)


    min_value = min(np.min(output), np.min(input[7:]), np.min(target))
    max_value = max(np.max(output), np.max(input[7:]), np.max(target))

    fig, axs = plt.subplots(3, 5, figsize=(15, 9))
    for k in range(5):
        im = axs[0][k].imshow(input[7+k], cmap='gray', vmin=min_value, vmax=max_value)
        axs[0][k].title.set_text('Input at t - {}'.format(5*(4-k)))

    for k in range(5):
        axs[1][k].imshow(output[2*k], cmap='gray', vmin=min_value, vmax=max_value)
        axs[2][k].imshow(target[2*k], cmap='gray', vmin=min_value, vmax=max_value)
        axs[1][k].title.set_text('Pred at t + {}'.format(5*(2*k+2)))
        axs[2][k].title.set_text('GT at t + {}'.format(5*(2*k+2)))

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(output_dir + str(index))

def save_gif(single_seq, fname):
    if len(single_seq.shape) == 4:
        single_seq = single_seq.permute((0, 2, 3, 1))
        single_seq = torch.squeeze(single_seq, -1)
    single_seq = single_seq.cpu().detach().numpy()
    single_seq_masked = rain_map_thresholded(single_seq)
    img_seq = [Image.fromarray((img*255).astype(np.uint8), 'RGB') for img in single_seq_masked]
    img = img_seq[0]
    img.save(fname, save_all=True, append_images=img_seq[1:], duration=500, loop=0)


def rain_map_thresholded(single_seq):
    single_seq_masked = np.zeros((single_seq.shape[0], single_seq.shape[1], single_seq.shape[2], 3))
    single_seq_masked[:, :, :, 1] = np.where((0.1 < single_seq) & (single_seq < 1.), 1., 0.)
    single_seq_masked[:, :, :, 2] = np.where((1. < single_seq) & (single_seq < 2.5), 1., 0.)
    single_seq_masked[:, :, :, 0] = np.where((2.5 < single_seq), 1., 0.)
    return single_seq_masked

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
        self.indices = alist
        self.dataset = dataset

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def indices_except_undefined_sampler(dataset):

    """ Currently adapted for recurrent nn with wind """
    # TODO : adapt it to all configs

    samples_weight = []

    for i in tqdm(range(len(dataset))):
        condition_meet = True

        dataset_item_i = dataset.__getitem__(i)
        inputs, targets = dataset_item_i["input"], dataset_item_i["target"]

        # If the last image of the input sequence contains no rain, we don't take into account the sequence
        if torch.max(inputs[-1, 0]).item() < 0.001:
            condition_meet = False

        if torch.min(inputs[:, 0, :, :]).item() < 0 or torch.min(targets).item() < 0:
            condition_meet = False

        if condition_meet:
            samples_weight.append(i)

    return samples_weight
