import numpy as np
import pandas
from pathlib import Path
from numpy import sqrt
import matplotlib.pyplot as plt
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

TeralyzerResults = ['HHI HIPS TeraLyzer/HIPS MUT1 HHI_D=2054.csv', 'BT HIPS TeraLyzer/HIPS MUT1 2mm BT_D=2130.csv', 'BT HIPS TeraLyzer/HIPS MUT1 BT closer2emitter_D=2090.csv']

df = pandas.read_csv(Path('DavidNVA/MUT 1-1.csv'))

freq_dict_key = [key for key in df.keys() if "freq" in key][0]
eps_mat_r_key = [key for key in df.keys() if "epsilon_r" in key][0]
eps_mat_i_key = [key for key in df.keys() if "epsilon_i" in key][0]

frequencies = np.array(df[freq_dict_key])

eps_mat = np.array(df[eps_mat_r_key]) + 1j*np.array(df[eps_mat_i_key])

n_MUT1_nva = sqrt(np.abs(eps_mat) + eps_mat.real) / sqrt(2)

plt.plot(frequencies, n_MUT1_nva, label='David NVA')
for teralyzer_result in TeralyzerResults:
    data = load_material_data(teralyzer_result)
    plt.plot(data['freq'], data['ref_ind'], label=str(teralyzer_result))

plt.ylabel('Ref. ind')
plt.xlabel('Frequency (THz)')
plt.ylim((1, 2))
plt.legend()
plt.show()


