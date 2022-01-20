import torch
from torch.utils.data import Dataset
import os
import numpy as np

from utils import *



class MeteoDataset(Dataset):

    def __init__(self, rain_dir, input_length,  output_length, temporal_stride, dataset, recurrent_nn=False, wind_dir=None):

        self.rain_dir = rain_dir
        self.input_length = input_length
        self.output_length = output_length
        self.temporal_stride = temporal_stride
        self.recurrent_nn = recurrent_nn
        self.wind_dir = wind_dir

        self.mean_u = 159.88
        self.std_u = 451.17
        self.mean_v = 19.11
        self.std_v = 417.70

        # Get and filter rain data
        self.rain_files_names = [f for f in os.listdir(rain_dir)]
        self.rain_files_names = sorted(self.rain_files_names, key=lambda x: get_date_from_file_name(x))

        if dataset == 'valid':
            self.rain_files_names = [val for (idx, val) in enumerate(self.rain_files_names) if filter_one_week_over_two_for_eval(idx) == 0]

        if dataset == 'test':
            self.rain_files_names = [val for (idx, val) in enumerate(self.rain_files_names) if filter_one_week_over_two_for_eval(idx) == 1]

        # Get and filter wind data
        if wind_dir != None:
            self.U_wind_files_names = [f for f in os.listdir(wind_dir + '/U')]
            self.U_wind_files_names = [val for val in self.U_wind_files_names if filter_year(val, dataset)]
            #self.U_wind_files_names = sorted(self.U_wind_files_names, key=lambda x: get_date_from_file_name(x))

            self.V_wind_files_names = [f for f in os.listdir(wind_dir + '/V')]
            self.V_wind_files_names = [val for val in self.V_wind_files_names if filter_year(val, dataset)]
            #self.V_wind_files_names = sorted(self.V_wind_files_names, key=lambda x: get_date_from_file_name(x))

        self.normalization = 100/12


        if wind_dir != None:
            self.rain_files_names, self.U_wind_files_names, self.V_wind_files_names = keep_wind_when_rainmap_exists(self.rain_files_names,
                                                                                                                    self.U_wind_files_names,
                                                                                                                    self.V_wind_files_names)

        self.indices = []
        for i in range((len(self.rain_files_names)  - self.input_length - self.output_length) // self.temporal_stride + 1):
            sequence = self.rain_files_names[self.temporal_stride * i : self.temporal_stride * i + self.input_length + self.output_length]
            if not missing_file_in_sequence(sequence):
                self.indices += [i]


    def __len__(self):

        return len(self.indices)

    def __getitem__(self, i):

        idx = self.indices[i]

        rain_files_names_i = self.rain_files_names[self.temporal_stride * idx : self.temporal_stride * idx + self.input_length + self.output_length]
        if self.wind_dir != None:
            U_wind_files_names_i = self.U_wind_files_names[self.temporal_stride * idx : self.temporal_stride * idx + self.input_length + self.output_length]
            V_wind_files_names_i = self.V_wind_files_names[self.temporal_stride * idx : self.temporal_stride * idx + self.input_length + self.output_length]

        rain_path_files = [os.path.join(self.rain_dir, file_name) for file_name in rain_files_names_i]
        if self.wind_dir != None:
            U_wind_path_files = [os.path.join(self.wind_dir + '/U', u_wind) for u_wind in U_wind_files_names_i]
            V_wind_path_files = [os.path.join(self.wind_dir + '/V', v_wind) for v_wind in V_wind_files_names_i]

        # Create a sequence of input maps (rain (and wind)).
        data_map = np.load(rain_path_files[0])
        data_map = data_map[data_map.files[0]] / self.normalization
        if self.wind_dir != None:
            u_wind_map = np.load(U_wind_path_files[0])
            u_wind_map = (u_wind_map[u_wind_map.files[0]] - self.mean_u) / self.std_u
            v_wind_map = np.load(V_wind_path_files[0])
            v_wind_map = (v_wind_map[v_wind_map.files[0]] - self.mean_v) / self.std_v

        if self.recurrent_nn:
            data_map = torch.unsqueeze(torch.from_numpy(data_map).float(), dim=0)[None, :]
            if self.wind_dir != None:
                data_map = torch.cat((data_map, torch.from_numpy(u_wind_map).float()[None, None, :]), dim=1)
                data_map = torch.cat((data_map, torch.from_numpy(v_wind_map).float()[None, None, :]), dim=1)
        else:
            data_map = torch.unsqueeze(torch.from_numpy(data_map).float(), dim=0)
            if self.wind_dir != None:
                data_map = torch.cat((data_map, torch.from_numpy(u_wind_map).float()[None, :]), dim=0)
                data_map = torch.cat((data_map, torch.from_numpy(v_wind_map).float()[None, :]), dim=0)

        sequence_data = data_map

        for k in range(1, self.input_length):
            data_map = np.load(rain_path_files[k])
            data_map = data_map[data_map.files[0]] / self.normalization
            if self.wind_dir != None:
                u_wind_map = np.load(U_wind_path_files[k])
                u_wind_map = (u_wind_map[u_wind_map.files[0]] - self.mean_u) / self.std_u
                v_wind_map = np.load(V_wind_path_files[k])
                v_wind_map = (v_wind_map[v_wind_map.files[0]] - self.mean_v) / self.std_v


            if self.recurrent_nn:
                data_map = torch.unsqueeze(torch.from_numpy(data_map).float(), dim=0)[None, :]
                if self.wind_dir != None:
                    data_map = torch.cat((data_map, torch.from_numpy(u_wind_map).float()[None, None, :]), dim=1)
                    data_map = torch.cat((data_map, torch.from_numpy(v_wind_map).float()[None, None, :]), dim=1)
            else:
                data_map = torch.unsqueeze(torch.from_numpy(data_map).float(), dim=0)
                if self.wind_dir != None:
                    data_map = torch.cat((data_map, torch.from_numpy(u_wind_map).float()[None, :]), dim=0)
                    data_map = torch.cat((data_map, torch.from_numpy(v_wind_map).float()[None, :]), dim=0)

            sequence_data = torch.cat((sequence_data, data_map), dim=0)


        # Create a sequence of target maps (rain (and wind)).
        data_map = np.load(rain_path_files[self.input_length])
        data_map = data_map[data_map.files[0]] / self.normalization

        if self.recurrent_nn:
            data_map = torch.unsqueeze(torch.from_numpy(data_map).float(), dim=0)[None, :]
        else:
            data_map = torch.unsqueeze(torch.from_numpy(data_map).float(), dim=0)

        data_sequence_target = data_map

        for k in range(self.input_length + 1, self.output_length + self.input_length):
            data_map = np.load(rain_path_files[k])
            data_map = data_map[data_map.files[0]] / self.normalization

            if self.recurrent_nn:
                data_map = torch.unsqueeze(torch.from_numpy(data_map).float(), dim=0)[None, :]
            else:
                data_map = torch.unsqueeze(torch.from_numpy(data_map).float(), dim=0)

            data_sequence_target = torch.cat((data_sequence_target, data_map), dim=0)

        return {"input" : sequence_data, "target" : data_sequence_target}
