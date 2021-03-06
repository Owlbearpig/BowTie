import numpy as np
from numpy import (power, outer, sqrt, exp, sin, cos, conj, dot, pi,
                   einsum, arctan, array, arccos, conjugate, flip, angle, tan, arctan2)
import pandas
from pathlib import Path, PureWindowsPath
import scipy
from scipy.constants import c as c0
from scipy.optimize import basinhopping
from scipy.optimize import OptimizeResult
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from numpy.linalg import solve
import string
import matplotlib.pyplot as plt
import sys
from consts import *
import pandas as pd

def export_csv(data, path):
    df = pd.DataFrame(data=data)
    df.to_csv(path)

def read_tlcsv_file(path):
    df = pandas.read_csv(path)

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
    eps_i = np.array(df[epsilon_i_key])

    data = {'freq': frequencies, 'ref_ind': ref_ind, 'alpha': alpha, 'alpha_err': alha_err, 'ref_ind_err': ref_ind_err,
            'eps_r': eps_r, 'eps_i': eps_i}

    return data

def fbf_from_tl(tl_file_path, a=500, b=500, f_min=0, f_max=np.inf, resolution=1):
    tl_data = read_tlcsv_file(tl_file_path)

    frequencies = tl_data['freq']

    data_slice = np.where((frequencies > f_min) &
                          (frequencies < f_max))
    data_slice = data_slice[0][::resolution]

    m = len(data_slice)

    frequencies = frequencies[data_slice].reshape(m, 1)

    wls = (c0 / frequencies) * m_um  # wl in um

    eps1_r = tl_data['eps_r'][data_slice]
    eps1_i = tl_data['eps_i'][data_slice]

    eps1 = (eps1_r + eps1_i * 1j).reshape(m, 1)  # a, material
    eps2 = np.ones_like(eps1)  # b, air gaps

    n_s, n_p, k_s, k_p = form_birefringence((a, b), wls, eps1, eps2)

    return frequencies, n_s, n_p, k_s, k_p

def load_material_data(mat_name, f_min=0, f_max=np.inf, resolution=1):
    mat_paths = {
        'ceramic_slow': Path('material_data/Sample1_000deg_1825ps_0m-2Grad_D=3000.csv'),
        'ceramic_fast': Path('material_data/Sample1_090deg_1825ps_0m88Grad_D=3000.csv'),
        'HIPS_MUT_1_1': Path('DavidNVA/MUT 1-1.csv'),
        'HIPS_MUT_1_2': Path('material_data/MUT 1-2.csv'),
        'HIPS_MUT_1_3': Path('material_data/MUT 1-3.csv'),
        'HIPS_MUT_2_1': Path('material_data/MUT 2-1.csv'),
        'HIPS_MUT_2_2': Path('material_data/MUT 2-2.csv'),
        'HIPS_MUT_2_3': Path('material_data/MUT 2-3.csv'),
        'Fused_4eck': Path('material_data/4Eck_D=2042.csv'),
        'quartz_m_slow': Path('material_data/quartz_m_slow.csv'),
        'quartz_m_fast': Path('material_data/quartz_m_fast.csv'),
        'quartz_sellmeier_slow': Path('material_data/sellmeier_quartz_slow.csv'),
        'quartz_sellmeier_fast': Path('material_data/sellmeier_quartz_fast.csv'),
        'quartz_full_slow': Path('material_data/abs_slow_grisch1990_fit.csv'),
        'quartz_full_fast': Path('material_data/abs_fast_grisch1990_fit.csv'),
        'HIPS_HHI': Path('material_data/2mmHIPS_D=2000.csv'),
        'HIPS_HHI_MUT1_2090um': Path('HHI HIPS TeraLyzer/HHI HIPS 2mm MUT1_D=2090.csv'),
        'HIPS_BT_MUT1_2090um': Path('BT HIPS TeraLyzer/HIPS MUT1 BT closer2emitter_D=2090.csv'),

    }

    df = pandas.read_csv(mat_paths[mat_name])

    freq_dict_key = [key for key in df.keys() if "freq" in key][0]
    eps_mat_r_key = [key for key in df.keys() if "epsilon_r" in key][0]
    eps_mat_i_key = [key for key in df.keys() if "epsilon_i" in key][0]

    frequencies = np.array(df[freq_dict_key])

    data_slice = np.where((frequencies > f_min) &
                          (frequencies < f_max))
    data_slice = data_slice[0][::resolution]
    m = len(data_slice)

    frequencies = frequencies[data_slice].reshape(m, 1)

    wls = (c0 / frequencies) * m_um

    eps_mat_r = np.array(df[eps_mat_r_key])[data_slice]

    if not eps_mat_i_key:
        eps_mat_i = np.zeros_like(eps_mat_r)
    else:
        eps_mat_i = np.array(df[eps_mat_i_key])[data_slice]

    eps_mat1 = (eps_mat_r + eps_mat_i * 1j).reshape(m, 1)

    return eps_mat1, frequencies, wls, m


def get_einsum(m, n):
    # setup einsum_str
    s0 = string.ascii_lowercase + string.ascii_uppercase

    einsum_str = ''
    for i in range(n):
        einsum_str += s0[n + 2] + s0[i] + s0[i + 1] + ','

    # remove last comma
    einsum_str = einsum_str[:-1]
    einsum_str += '->' + s0[n + 2] + s0[0] + s0[n]

    # einsum path
    test_array = np.zeros((n, m, 2, 2))
    einsum_path = np.einsum_path(einsum_str, *test_array, optimize='greedy')

    # calc matrix chain from m_n_2_2 tensor
    # (If n > 2x alphabet length, einsum breaks -> split into k parts... n/k)
    """
    1_n_(2x2)*...*1_3_(2x2)*1_2_(2x2)*1_1_(2x2) -> (2x2)_1
    2_n_(2x2)*...*2_3_(2x2)*2_2_(2x2)*2_1_(2x2) -> (2x2)_2
    .
    .
    m_n_(2x2)*...*m_3_(2x2)*m_2_(2x2)*m_1_(2x2) -> (2x2)_m
    """

    def matrix_chain_calc(matrix_array):
        return

    return einsum_str, einsum_path


def form_birefringence(stripes, wls, eps_mat1, eps_mat2):
    """
    :return: array with length of frequency, frequency resolved [ns, np, ks, kp]
    """

    l_mat1, l_mat2 = stripes

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

    n_s = n_s#-(np.linspace(0, 0.023, n_s.shape[0]).reshape(n_s.shape))

    return np.array([n_s, n_p, k_s, k_p])


def j_stack(x, m, n, wls, n_s, n_p, k_s, k_p, einsum_str, einsum_path):
    j = np.zeros((m, n, 2, 2), dtype=complex)

    angles, d = x[0:n], x[n:2 * n]

    phi_s, phi_p = (2 * n_s * pi / wls) * d.T, (2 * n_p * pi / wls) * d.T
    alpha_s, alpha_p = -(2 * pi * k_s / wls) * d.T, -(2 * pi * k_p / wls) * d.T
    alpha_s, alpha_p = np.zeros_like(wls), -(2 * pi * (k_p - k_s) / wls) * d.T
    #"""
    x, y = 1j * phi_s + alpha_s, 1j * phi_p + alpha_p
    angles = np.tile(angles, (m, 1))

    j[:, :, 0, 0] = exp(y) * sin(angles) ** 2 + exp(x) * cos(angles) ** 2
    j[:, :, 0, 1] = 0.5 * sin(2 * angles) * (exp(x) - exp(y))
    j[:, :, 1, 0] = j[:, :, 0, 1]
    j[:, :, 1, 1] = exp(x) * sin(angles) ** 2 + exp(y) * cos(angles) ** 2
    j = np.einsum('ijnm,ij->ijnm',j,exp(-(x+y)/2))
    """
    delta = (phi_s-phi_p)/2
    sd = 1j * sin(delta)

    sdca = sd * cos(2 * angles)

    j[:, :, 0, 0] = j[:, :, 1, 1] = cos(delta)
    j[:, :, 0, 1] = j[:, :, 1, 0] = sd * sin(2 * angles)
    j[:, :, 0, 0] += sdca
    j[:, :, 1, 1] -= sdca
    """
    np.einsum(einsum_str, *j.transpose((1, 0, 2, 3)), out=j[:, 0], optimize=einsum_path[0])

    j = j[:, 0]

    return j


def wp_cnt(settings):
    if settings['bf'] == 'intrinsic':
        n = len(settings['x'])//2
    else:
        n = (len(settings['x'])-2) // 2
    return n


def min_thickness(settings):
    eps_mat1, eps_mat2, n_s, n_p, k_s, k_p, f, wls, m = material_values(settings, return_vals=True)
    bf = n_s - n_p
    argmax_wls = np.argmax(wls)
    print(bf[argmax_wls])
    print(wls[argmax_wls])
    print(0.25*wls[argmax_wls]/bf[argmax_wls])

def thickness_for_1thz(settings):
    eps_mat1, eps_mat2, n_s, n_p, k_s, k_p, f, wls, m = material_values(settings, return_vals=True)
    bf = n_s - n_p
    index_1thz = np.argmin(np.abs(1*THz-f))
    print('bf @ 1 THz:', bf[index_1thz])
    print('wl @ 1 THz:', wls[index_1thz])
    print('thickness req.:', 0.25*wls[index_1thz]/bf[index_1thz])

def loss(j):
    #d, angles = x[0:n], x[n:2*n]
    #d, angles = x[0:n], np.deg2rad(x[n:2*n])
    #angles = np.deg2rad(x[0:n])

    # TODO add optimization(fix bounds especially thickness bounds)

    #delta_equiv = 2*arccos(0.5*np.abs(j[:, 0, 0]+conjugate(j[:, 0, 0])))

    # hwp 1 int opt
    #L = (1 / m) * (1 - j[:, 1, 0] * conj(j[:, 1, 0])) ** 2 + (j[:, 0, 0] * conj(j[:, 0, 0])) ** 2

    # hwp 2 int opt
    #L = (1 / m) * (1 - j[:, 1, 0].real)**2 + (j[:, 1, 0].imag) ** 2 + (j[:, 0, 0] * conj(j[:, 0, 0])) ** 2

    # hwp 3 mat opt
    #print(((np.angle(j[:,1,0])-np.angle(j[:,0,1]))**2))
    #print((j[:, 1, 0].imag - j[:, 0, 1].imag) ** 2)
    #print()
    """
    L = (1 / m) * np.absolute(j[:,0,0])**2+np.absolute(j[:,1,1])**2+
                        #(1-np.abs(j[:,1,0].real))**2+(1-np.abs(j[:,0,1].real))**2)
                        (1-j[:,1,0].real)+(1-j[:,0,1].real)+
                        (j[:,1,0].imag)**2+(j[:,0,1].imag)**2
    """

    # hwp 4 mat opt back to start
    #L = np.absolute(j[:, 0, 0]) ** 2 + np.absolute(j[:, 1, 1]) ** 2 \
    #+ (1-j[:, 0, 1].imag) ** 2 + (1-j[:, 1, 0].imag) ** 2

    # qwp state opt
    norm = 1#j[:, 0, 0] * conjugate(j[:, 0, 0]) + j[:, 1, 0] * conjugate(j[:, 1, 0])
    A, C = j[:, 0, 0]/norm, j[:, 1, 0]/norm
    q = A/C
    L = q.real ** 2 + (q.imag - 1) ** 2
    #L = (j[:, 1, 0] * conj(j[:, 1, 0]) - j[:, 0, 0] * conj(j[:, 0, 0])) ** 2

    # qwp state opt 2.
    #a, b = j[:, 0, 0], j[:, 1, 0]
    #phi = angle(a)-angle(b)
    #L = (np.abs(b)-np.abs(a))**2+(phi-pi/2)**2

    # Masson ret. opt.
    #A, B = j[:, 0, 0], j[:, 0, 1]
    #delta_equiv = 2 * arctan(sqrt((A.imag ** 2 + B.imag ** 2) / (A.real ** 2 + B.real ** 2)))
    #L = (1/m)*(delta_equiv-pi)**2

    return L

def material_values(settings, return_vals=False):
    if return_vals:
        resolution = 1
        f_min, f_max = 0.0*THz, 2.5*THz
    else:
        resolution = settings['resolution']
        f_min, f_max = settings['f_range']

    if settings['bf'] == 'intrinsic':
        eps_mat1, f, wls, m = load_material_data(settings['mat_name'][0], f_min, f_max, resolution)
        eps_mat2, _, _, _ = load_material_data(settings['mat_name'][1], f_min, f_max, resolution)
        n_s, n_p = sqrt(np.abs(eps_mat1) + eps_mat1.real) / sqrt(2), sqrt(np.abs(eps_mat2) + eps_mat2.real) / sqrt(2)
        k_s, k_p = sqrt(np.abs(eps_mat1) - eps_mat1.real) / sqrt(2), sqrt(np.abs(eps_mat2) - eps_mat2.real) / sqrt(2)
    else:
        eps_mat1, f, wls, m = load_material_data(settings['mat_name'][0], f_min, f_max, resolution)
        eps_mat2 = np.ones_like(eps_mat1).reshape(m, 1)  # air
        n_s, n_p, k_s, k_p = None, None, None, None,

    return eps_mat1, eps_mat2, n_s, n_p, k_s, k_p, f, wls, m


def setup(settings, return_vals=False):
    eps_mat1, eps_mat2, n_s, n_p, k_s, k_p, f, wls, m = material_values(settings, return_vals=return_vals)
    #print(len(f))
    #exit()
    n = wp_cnt(settings)

    einsum_str, einsum_path = get_einsum(m, n)

    def make_j(x):
        nonlocal n_s, n_p, k_s, k_p

        if settings['bf'] == 'intrinsic':
            j = j_stack(x, m, n, wls, n_s, n_p, k_s, k_p, einsum_str, einsum_path)
        else:
            stripes = x[-2], x[-1]
            n_s, n_p, k_s, k_p = form_birefringence(stripes, wls, eps_mat1, eps_mat2)
            j = j_stack(x, m, n, wls, n_s, n_p, k_s, k_p, einsum_str, einsum_path)
        """
        plt.plot(f, n_s, label='n_s')
        plt.plot(f, n_p, label='n_p')
        plt.show()
        
        from generate_plotdata import export_csv
        export_csv({'freq': f.flatten(), 'n_s': n_s.flatten(), 'n_p': n_p.flatten()}, 'calculated_ref_ind_design_err.csv')
        """
        return j

    def erf(x):
        j = make_j(x)
        L = loss(j)

        return np.sum(L)

    if return_vals:
        j = make_j(settings['x'])
        return j, f, wls
    else:
        return erf

