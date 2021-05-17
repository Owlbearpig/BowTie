import numpy as np
import pandas
from pathlib import Path
import os

def load_material_data(path):
    df = pandas.read_csv(path)

    freq_dict_key = [key for key in df.keys() if 'freq' in key][0]
    ref_ind_key = [key for key in df.keys() if 'ref_ind' in key][0]
    alpha_key = [key for key in df.keys() if 'alpha' in key][0]
    alpha_err_key = [key for key in df.keys() if 'delta_A' in key][0]
    ref_ind_err_key = [key for key in df.keys() if "delta_N" in key][0]

    frequencies = np.array(df[freq_dict_key])
    ref_ind = np.array(df[ref_ind_key])
    alpha = np.array(df[alpha_key])
    alha_err = np.array(df[alpha_err_key])
    ref_ind_err = np.array(df[ref_ind_err_key])

    data = {'freq': frequencies, 'ref_ind': ref_ind, 'alpha': alpha, 'alpha_err': alha_err, 'ref_ind_err': ref_ind_err}

    return data


base = Path('CeramicTeraLyzerResult')

# find result files
resultfiles = [os.path.join(root, name)
               for root, dirs, files in os.walk(base)
               for name in files
               if name.endswith('.csv')]

df = pandas.read_csv(Path('DavidNVA/MUT 1-1.csv'))

freq_dict_key = [key for key in df.keys() if "freq" in key][0]
eps_mat_r_key = [key for key in df.keys() if "epsilon_r" in key][0]
eps_mat_i_key = [key for key in df.keys() if "epsilon_i" in key][0]

frequencies = np.array(df[freq_dict_key])

MUT1_eps_r = np.array(df[eps_mat_r_key])

MUT1_eps_i = np.array(df[eps_mat_i_key])




