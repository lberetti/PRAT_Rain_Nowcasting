import torch
from torch.utils.data import Dataset
import os
import numpy as np

from utils import get_date_from_file_name, filter_one_week_over_two_for_eval, missing_file_in_sequence



class MeteoDataset(Dataset):

    def __init__(self, rain_dir, input_length,  output_length, temporal_stride, dataset):

        self.rain_dir = rain_dir
        self.input_length = input_length
        self.output_length = output_length
        self.temporal_stride = temporal_stride

        self.files_names = [f for f in os.listdir(rain_dir) if os.path.isfile(os.path.join(rain_dir, f))]
        self.files_names = sorted(self.files_names, key=lambda x: get_date_from_file_name(x))[:2000]

        if dataset == 'valid':
            self.files_names = [val for (idx, val) in enumerate(self.files_names) if filter_one_week_over_two_for_eval(idx) == 0]

        if dataset == 'test':
            self.files_names = [val for (idx, val) in enumerate(self.files_names) if filter_one_week_over_two_for_eval(idx) == 1]

        self.normalization = 100/12

        self.indices = []
        for i in range((len(self.files_names)  - self.input_length - self.output_length) // self.temporal_stride + 1):
            sequence = self.files_names[self.temporal_stride * i : self.temporal_stride * i + self.input_length + self.output_length]
            if not missing_file_in_sequence(sequence):
                self.indices += [i]



    def __len__(self):

        return len(self.indices)

    def __getitem__(self, i):

        idx = self.indices[i]
        files_names_i = self.files_names[self.temporal_stride * idx : self.temporal_stride * idx + self.input_length + self.output_length]
        path_files = [os.path.join(self.rain_dir, file_name) for file_name in files_names_i]

        # Create a sequence of input rain maps.
        rain_map = np.load(path_files[0])
        rain_map = rain_map[rain_map.files[0]] / self.normalization
        rain_map = torch.unsqueeze(torch.from_numpy(rain_map).float(), dim=0)[None, :]
        rain_sequence_data = rain_map
        for k in range(1, self.input_length):
            rain_map = np.load(path_files[k])
            rain_map = rain_map[rain_map.files[0]] / self.normalization
            rain_map = torch.unsqueeze(torch.from_numpy(rain_map).float(), dim=0)[None, :]
            rain_sequence_data = torch.cat((rain_sequence_data, rain_map), dim=0)

        # Create a sequence of target rain maps.
        rain_map = np.load(path_files[self.output_length])
        rain_map = rain_map[rain_map.files[0]] / self.normalization
        rain_map = torch.unsqueeze(torch.from_numpy(rain_map).float(), dim=0)[None, :]
        rain_sequence_target = rain_map
        for k in range(self.input_length + 1, self.output_length + self.input_length):
            rain_map = np.load(path_files[k])
            rain_map = rain_map[rain_map.files[0]] / self.normalization
            rain_map = torch.unsqueeze(torch.from_numpy(rain_map).float(), dim=0)[None, :]
            rain_sequence_target = torch.cat((rain_sequence_target, rain_map), dim=0)

        return {"input" : rain_sequence_data, "target" : rain_sequence_target}
