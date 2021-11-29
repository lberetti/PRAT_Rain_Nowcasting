import os
import torch

def get_date_from_file_name(filename):

    date_infos = [int(val[1:]) for val in filename.split('/')[-1].split('.')[0].split('-')]
    return date_infos


def weighted_mse_loss(input, target):

    threshold = [0, 2, 5, 10, 30, 1000]
    weights = [1, 2, 5, 10, 30]
    assert len(threshold) == len(weights) + 1
    loss = torch.Tensor([0])

    for k in range(len(weights)):
        mask = ((threshold[k] <= target) & (target < threshold[k+1])).float()
        loss += torch.sum(weights[k] * ((input*mask - target*mask)) ** 2)

    return loss
