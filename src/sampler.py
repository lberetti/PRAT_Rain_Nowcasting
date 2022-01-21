from torch.utils.data.sampler import Sampler
from tqdm import tqdm
import torch

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
    # TODO : adapt it to all configs (without non recurrent network)

    samples_weight = []

    for i in tqdm(range(len(dataset))):
        condition_meet = True

        dataset_item_i = dataset.__getitem__(i)
        inputs, targets = dataset_item_i["input"], dataset_item_i["target"]

        # If the last image of the input sequence contains no rain, we don't take into account the sequence
        if torch.max(inputs[-1, 0]).item() == 0.0:
            condition_meet = False

        if torch.min(inputs[:, 0, :, :]).item() < 0 or torch.min(targets).item() < 0:
            condition_meet = False

        if condition_meet:
            samples_weight.append(i)

    return samples_weight
