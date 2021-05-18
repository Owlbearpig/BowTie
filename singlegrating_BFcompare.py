import os
from pathlib import Path
import numpy as np
import pandas
import matplotlib.pylab as plt
import pandas as pd
import pathlib
from pathlib import Path
from scipy.constants import c as c0
from numpy import power, sqrt, sin, cos, outer, abs, pi

THz = 10**12
m_um = 10**6

def form_birefringence(a, b, eps_mat1, eps_mat2, wls):
    """
    :return: array with length of frequency, frequency resolved [ns, np, ks, kp]
    """

    l_mat1, l_mat2 = a, b

    a = (1 / 3) * power(outer(1 / wls, (l_mat1 * l_mat2 * pi) / (l_mat1 + l_mat2)), 2)

    # first order s and p
    wp_eps_s_1 = outer((eps_mat2 * eps_mat1), (l_mat2 + l_mat1)) / (
            outer(eps_mat2, l_mat1) + outer(eps_mat1, l_mat2))

    wp_eps_p_1 = outer(eps_mat1, l_mat1 / (l_mat2 + l_mat1)) + outer(eps_mat2, l_mat2 / (l_mat2 + l_mat1))

    # 2nd order
    wp_eps_s_2 = wp_eps_s_1 + (a * power(wp_eps_s_1, 3) * wp_eps_p_1 * power((1 / eps_mat1 - 1 / eps_mat2), 2))
    wp_eps_p_2 = wp_eps_p_1 + (a * power((eps_mat1 - eps_mat2), 2))

    # returns
    n_p, n_s = (
        sqrt(abs(wp_eps_p_2) + wp_eps_p_2.real) / sqrt(2),
        sqrt(abs(wp_eps_s_2) + wp_eps_s_2.real) / sqrt(2)
    )
    k_p, k_s = (
        sqrt(abs(wp_eps_p_2) - wp_eps_p_2.real) / sqrt(2),
        sqrt(abs(wp_eps_s_2) - wp_eps_s_2.real) / sqrt(2)
    )

    return [n_s, n_p, k_s, k_p]

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

def load_material_data(tl_file_path):
    df = pandas.read_csv(tl_file_path)

    freq_dict_key = [key for key in df.keys() if 'freq' in key][0]
    ref_ind_key = [key for key in df.keys() if 'ref_ind' in key][0]
    alpha_key = [key for key in df.keys() if 'alpha' in key][0]
    alpha_err_key = [key for key in df.keys() if 'delta_A' in key][0]
    ref_ind_err_key = [key for key in df.keys() if "delta_N" in key][0]
    epsilon_r_key = [key for key in df.keys() if "epsilon_r" in key][0]
    epsilon_i_key = [key for key in df.keys() if "epsilon_i" in key][0]

    frequencies = np.array(df[freq_dict_key])
    ref_ind = np.array(df[ref_ind_key])
    alpha = np.array(df[alpha_key])
    alha_err = np.array(df[alpha_err_key])
    ref_ind_err = np.array(df[ref_ind_err_key])
    eps_r = np.array(df[epsilon_r_key])
    
    if not epsilon_i_key:
        eps_i = np.zeros_like(eps_r)
    else:
        eps_i = np.array(df[epsilon_i_key])
    
    data = {'freq': frequencies, 'ref_ind': ref_ind, 'alpha': alpha, 'alpha_err': alha_err, 'ref_ind_err': ref_ind_err,
            'eps_r': eps_r, 'eps_i': eps_i}

    return data

def fbf_from_tl(tl_file_path, a=500, b=500, f_min=0, f_max=np.inf, resolution=1):
    
    tl_data = load_material_data(tl_file_path)

    frequencies = tl_data['freq']

    data_slice = np.where((frequencies > f_min) &
                          (frequencies < f_max))
    data_slice = data_slice[0][::resolution]

    m = len(data_slice)

    frequencies = frequencies[data_slice].reshape(m, 1)

    wls = (c0 / frequencies) * m_um # wl in um

    eps1_r = tl_data['eps_r'][data_slice]
    eps1_i = tl_data['eps_i'][data_slice]
    

    eps1 = (eps1_r + eps1_i * 1j).reshape(m, 1) # a, material
    eps2 = np.ones_like(eps1) # b, air gaps

    n_s, n_p, k_s, k_p = form_birefringence(a, b, eps1, eps2, wls)

    return frequencies, n_s, n_p, k_s, k_p

fig = plt.figure()

# plot real data
base = Path('singlegratingSlimcorrect/tl_result')

# find result files
resultfiles = [os.path.join(root, name)
               for root, dirs, files in os.walk(base)
               for name in files
               if name.endswith('.csv')]

for resultfile in resultfiles:
    tl_data = load_material_data(resultfile)
    freq, ref_ind, dn = tl_data['freq'], tl_data['ref_ind'], tl_data['ref_ind_err']
    if '90deg' in resultfile:
        freq_90deg, ref_ind_90deg = tl_data['freq'], tl_data['ref_ind']
    if ' 0deg' in resultfile:
        freq_0deg, ref_ind_0deg = tl_data['freq'], tl_data['ref_ind']

    plt.plot(freq*10**-12, ref_ind, label=str(resultfile))
    plt.fill_between(freq*10**-12, ref_ind - dn, ref_ind + dn, alpha=0.5)

# plot calculated fbf
HIPS_MUT1_TL_RES_PATH = Path('BT HIPS TeraLyzer/HIPS MUT1 BT closer2emitter_D=2090.csv')
frequencies, n_s, n_p, _, _ = fbf_from_tl(HIPS_MUT1_TL_RES_PATH, a=500, b=500)
plt.plot(frequencies*10**-12, n_s, label='n_s')
plt.plot(frequencies*10**-12, n_p, label='n_p')

plt.legend()
plt.xlabel('Frequency (THz)')
plt.ylabel('Refractive index')
plt.show()

np.save('measured_bf_BowTie_90deg', ref_ind_90deg)
np.save('measured_bf_BowTie_0deg', ref_ind_0deg)
np.save('measured_bf_BowTie_freqs', freq_90deg)
np.save('measured_bf_BowTie', ref_ind_0deg-ref_ind_90deg)

plt.plot(freq_90deg*10**-12, ref_ind_0deg-ref_ind_90deg, label='ref_ind 0deg-90deg (measured)')
plt.plot(frequencies*10**-12, n_p-n_s, label='n_p-n_s')
plt.legend()
plt.xlabel('Frequency (THz)')
plt.ylabel('Birefringence')
plt.show()
