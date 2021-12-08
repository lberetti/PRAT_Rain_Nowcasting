import torch
from torch.utils.data import Dataset
import os
import numpy as np

from utils import get_date_from_file_name



class MeteoDataset(Dataset):

    def __init__(self, rain_dir, input_length,  output_length):

        self.rain_dir = rain_dir
        self.input_length = input_length
        self.output_length = output_length

        self.files_names = [f for f in os.listdir(rain_dir)[:50] if os.path.isfile(os.path.join(rain_dir, f))]
        self.files_names = sorted(self.files_names, key=lambda x: get_date_from_file_name(x))

        self.normalization = 12
        #print(self.files_names)


    def __len__(self):

        return len(self.files_names) - self.input_length - self.output_length


    def __getitem__(self, i):

        files_names_i = self.files_names[i : i + self.input_length + self.output_length]
        path_files = [os.path.join(self.rain_dir, file_name) for file_name in files_names_i]

        # Create a sequence of input rain maps.
        rain_map = np.load(path_files[0])
        rain_map = rain_map[rain_map.files[0]] / self.normalization
        rain_map = torch.unsqueeze(torch.from_numpy(rain_map).float(), dim=0)
        rain_sequence_data = rain_map
        for k in range(1, self.input_length):
            rain_map = np.load(path_files[k])
            rain_map = rain_map[rain_map.files[0]] / self.normalization
            rain_map = torch.unsqueeze(torch.from_numpy(rain_map).float(), dim=0)
            rain_sequence_data = torch.cat((rain_sequence_data, rain_map), dim=0)

        # Create a sequence of target rain maps.
        rain_map = np.load(path_files[self.output_length])
        rain_map = rain_map[rain_map.files[0]] / self.normalization
        rain_map = torch.unsqueeze(torch.from_numpy(rain_map).float(), dim=0)
        rain_sequence_target = rain_map
        for k in range(self.input_length + 1, self.output_length + self.input_length):
            rain_map = np.load(path_files[k])
            rain_map = rain_map[rain_map.files[0]] / self.normalization
            rain_map = torch.unsqueeze(torch.from_numpy(rain_map).float(), dim=0)
            rain_sequence_target = torch.cat((rain_sequence_target, rain_map), dim=0)

        return {"input" : rain_sequence_data, "target" : rain_sequence_target}
