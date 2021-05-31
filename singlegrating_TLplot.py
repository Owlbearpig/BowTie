import os
from pathlib import Path
import numpy as np
import pandas
import matplotlib.pylab as plt
import pandas as pd
import pathlib
from pathlib import Path

def fft(t, y):
    delta = t[1] - t[0]
    Y = np.fft.rfft(y * len(y), axis=0)
    freqs = np.fft.rfftfreq(len(t), delta)
    ft = 10 * np.log10(np.abs(Y))
    #plt.plot(t, y)
    #plt.plot(freqs[1:], ft[1:])
    #print(np.mean(np.abs(Y))/np.sqrt(np.var(np.abs(Y))))
    #plt.xlim((0, 0.75))
    #plt.show()

    return freqs[1:], Y[1:]

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def export_csv(data, path):
    df = pd.DataFrame(data=data)
    df.to_csv(path)

def ref_ind_err(ref_ind, d):
    dd = 5
    return dd * (ref_ind - 1) / (d + dd)

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


base = Path('SingleGratings/singlegratingPhatCorrectAngles/tl_result')

# find result files
resultfiles = [os.path.join(root, name)
               for root, dirs, files in os.walk(base)
               for name in files
               if name.endswith('.csv') and '7900' not in str(name)]

for resultfile in resultfiles:
    tl_data = load_material_data(resultfile)
    freq, alpha, alpha_err = tl_data['freq'], tl_data['alpha'], tl_data['alpha_err']
    plt.plot(freq*10**-12, alpha, label=str(resultfile))
    plt.fill_between(freq*10**-12, alpha - alpha_err, alpha + alpha_err, alpha=0.5)

plt.xlabel('Frequency (THz)')
plt.ylabel(r'$\alpha \ (cm^{-1})$')
plt.legend()
plt.show()

fig = plt.figure()

data_export = {'freq': freq}
for resultfile in resultfiles:
    tl_data = load_material_data(resultfile)
    freq, ref_ind, dn = tl_data['freq'], tl_data['ref_ind'], tl_data['ref_ind_err']
    deg = str(resultfile).split(' ')[-1].split('_')[0]
    plt.plot(freq*10**-12, ref_ind, label=str(resultfile))
    plt.fill_between(freq*10**-12, ref_ind - dn, ref_ind + dn, alpha=0.5)
    data_export[deg] = ref_ind

export_csv(data_export, 'singlegratingPhat_measured_ri.csv')
plt.legend()
plt.xlabel('Frequency (THz)')
plt.ylabel('Refractive index')
plt.show()
