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


def compute_confusion_matrix(output, target, threshold):

    # Computing this tensor helps to compute the confusion matrix much more easily.
    difference = 2*torch.where(output >= threshold, 1.0, 0.0) - torch.where(target >= threshold, 1.0, 0.0)
    # True positive
    true_positive = torch.sum(difference==1).item()
    # True negative
    true_negative = torch.sum(difference==0).item()
    # False positive
    false_positive = torch.sum(difference==2).item()
    # False negative
    false_negative = torch.sum(difference==-1).item()

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


def round_dictionnary_values(dictionnary, decimal_n):
    for key in dictionnary:
        dictionnary[key].round(decimal_n)
    return dictionnary


class CustomMetric():

    def on_epoch_begin(self):
        self.seen = 0
        self.total = 0

    def on_batch_end(self, metric_score):
        self.seen += 1
        self.total += metric_score

    def on_epoch_end(self):
        if self.seen > 0:
            self.total = self.total / self.seen

    def round(self, decimal_n):
        self.total = round(self.total, decimal_n)


class CSI_Score(CustomMetric):
    def __init__(self, threshold):
        self.threshold = threshold

    def compute_metric(self, output, target):
        max_value = max(torch.max(output).item(), torch.max(target).item())
        if max_value > self.threshold:
            conf_mat = compute_confusion_matrix(output, target, self.threshold)
            metric_score = conf_mat['true_positive'] / (
                                        conf_mat['true_positive'] + conf_mat['false_negative'] + conf_mat['false_positive'])
            self.on_batch_end(metric_score)


def build_metrics_dict(thresholds):
    metrics = {'train' : {}, 'val': {}}

    for thresh in thresholds:
        metrics['train']['csi_score_' + str(thresh)] = CSI_Score(thresh)
        metrics['val']['csi_score_' + str(thresh)] = CSI_Score(thresh)

    return metrics
