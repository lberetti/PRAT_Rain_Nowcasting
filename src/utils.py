import os
import torch
import matplotlib.pyplot as plt

def get_date_from_file_name(filename):

    date_infos = [int(val[1:]) for val in filename.split('/')[-1].split('.')[0].split('-')]
    return date_infos


def mse_rainfall(output, target):

    weight_mask = compute_weight_mask(target)
    loss = weighted_mse_loss(output, target, weight_mask)

    return loss


def weighted_mse_loss(output, target, weight_mask):
    return torch.sum(torch.multiply(weight_mask, (output - target) ** 2))


def compute_weight_mask(target):

    """
    threshold = [0, 2, 5, 10, 30, 1000]
    weights = [1., 2., 5., 10., 30.]

    # To solve
    mask = torch.ones(target.size(), dtype=torch.double).cuda()

    for k in range(len(weights)):
        mask = torch.where((threshold[k] <= target) & (target < threshold[k+1]), weights[k], mask)
    """

    return torch.where((0 <= target) & (target < 2), 1., 0.) \
    + torch.where((2 <= target) & (target < 5), 2., 0.) \
    + torch.where((5 <= target) & (target < 10), 5., 0.) \
    + torch.where((10 <= target) & (target < 30), 10., 0.) \
    + torch.where((30 <= target), 30., 0.)


def plot_output_gt(output, target):

    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(output, cmap='gray')
    axs[1].imshow(target, cmap='gray')
    axs[0].title.set_text('Output of the NN')
    axs[1].title.set_text('Ground Truth')
    plt.show()
