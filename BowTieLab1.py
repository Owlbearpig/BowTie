import os
from pathlib import Path
import numpy as np
import pandas
import matplotlib.pylab as plt
import pandas as pd
import pathlib
from pathlib import Path


def export_csv(data, path):
    df = pd.DataFrame(data=data)
    df.to_csv(path)


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


base = Path('SiWaferTeralyzerResult')

# find result files
resultfiles = [os.path.join(root, name)
               for root, dirs, files in os.walk(base)
               for name in files
               if name.endswith('.csv') and 'Test2' in str(name)]

materials = ['SiWafer']

d = {'SiWafer': 460}

def ref_ind_err(ref_ind, d):
    dd = 5
    return dd * (ref_ind - 1) / (d + dd)

material = 'SiWafer'
for resultfile in resultfiles:
    if material in str(resultfile):
        #deg = str(resultfile).split('_')[-2]
        SiWafer_res = load_material_data(resultfile)
        freq, alpha, alpha_err = SiWafer_res['freq'], SiWafer_res['alpha'], SiWafer_res['alpha_err']

        plt.plot(freq*10**-12, alpha, label=f'{material}')
        plt.fill_between(freq*10**-12, alpha - alpha_err, alpha + alpha_err, alpha=0.5)

plt.xlabel('Frequency (THz)')
plt.ylabel(r'$\alpha \ (cm^{-1})$')
plt.legend()
plt.show()


fig = plt.figure()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

for i, material in enumerate(materials):
    export_data = {}
    for resultfile in resultfiles:
        if material in str(resultfile):
            #deg = str(resultfile).split('_')[-2]
            res_data = load_material_data(resultfile)
            freq, ref_ind, alpha = res_data['freq'], res_data['ref_ind'], res_data['alpha']

            #ref_ind = moving_average(ref_ind, 10)

            dn = res_data['ref_ind_err']
            #dn = ref_ind_err(ref_ind, d[material]) # 'error from thickness uncertainty'

            #plt.subplot(2, 5, i+1)
            plt.plot(freq*10**-12, ref_ind, label=f'{material}')
            plt.fill_between(freq*10**-12, ref_ind - dn, ref_ind + dn, alpha=0.5)

    plt.legend()
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Refractive index')

plt.show()

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

datapath = Path('BowTieSetup/SiWaferTest')

# find result files
reffiles = [os.path.join(root, name)
               for root, dirs, files in os.walk(datapath)
               for name in files
               if name.endswith('.txt') and 'Ref' in str(name)]

ref1 = np.loadtxt(reffiles[0])
data_pnts = len(ref1[:, 0]) // 2
freqs, Y = fft(ref1[:,0], ref1[:,1])
spectra = np.zeros((len(reffiles), data_pnts), dtype=complex)

for i, reffile in enumerate(reffiles):
    ref = np.loadtxt(reffile)
    t, y = ref[:,0], ref[:,1]
    freqs, Y = fft(t, y)
    spectra[i, :] = Y

SNR = np.zeros(data_pnts)
for i in range(data_pnts):
    freq_ampl = np.abs(spectra[:, i])
    SNR[i] = np.mean(freq_ampl)/np.sqrt(np.var(freq_ampl))

plt.plot(freqs, 10*np.log10(SNR))
plt.xlim((0, 2))
plt.xlabel('Frequency (THz)')
plt.ylabel('SNR (dB)')
plt.show()



