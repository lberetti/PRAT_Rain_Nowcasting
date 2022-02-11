import os
import numpy as np
from tqdm import tqdm
from utils import *

rain_dir = '../../Data/MeteoNet-Brest/rainmap/train'
wind_dir = '../../Data/MeteoNet-Brest/wind'

# Get and filter rain data
rain_files_names = [f for f in os.listdir(rain_dir)]
rain_files_names = sorted(rain_files_names, key=lambda x: get_date_from_file_name(x))
# Get and filter rain data
U_wind_files_names = [f for f in os.listdir(wind_dir + '/U')]
U_wind_files_names = [val for val in U_wind_files_names if filter_year(val, 'train')]

V_wind_files_names = [f for f in os.listdir(wind_dir + '/V')]
V_wind_files_names = [val for val in V_wind_files_names if filter_year(val, 'train')]



files_names = keep_wind_when_rainmap_exists(rain_files_names,
                                            U_wind_files_names,
                                            V_wind_files_names)


U_wind_files_names = [os.path.join(wind_dir + '/U', u_wind) for u_wind in files_names]
V_wind_files_names = [os.path.join(wind_dir + '/V', v_wind) for v_wind in files_names]
u_wind = []
v_wind = []
print("Loading files ...")
for k in tqdm(range(len(U_wind_files_names))):
    u_wind_map = np.load(U_wind_files_names[k])
    u_wind_map = u_wind_map[u_wind_map.files[0]]
    v_wind_map = np.load(V_wind_files_names[k])
    v_wind_map = v_wind_map[v_wind_map.files[0]]
    u_wind.append(u_wind_map)
    v_wind.append(v_wind_map)
u_wind = np.array(u_wind)
v_wind = np.array(v_wind)
print("Mean U (std): {}(+-{})".format(np.mean(u_wind), np.std(u_wind)))
print("Mean V (std): {}(+-{})".format(np.mean(v_wind), np.std(v_wind)))
