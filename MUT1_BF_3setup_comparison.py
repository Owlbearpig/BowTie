from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, array
import pandas
from functions import material_values, form_birefringence
from functions import export_csv


materials = ['HIPS_MUT_1_1', 'HIPS_HHI_MUT1_2090um', 'HIPS_BT_MUT1_2090um', ]
for mi, material in enumerate(materials):
    result_HIPS_David = {
            'name': '',
            'comments': '',
            'x': '',
            'bf': 'form',
            'mat_name': (material, '')
    }

    eps_mat1, eps_mat2, _, _, _, _, f, wls, m = material_values(result_HIPS_David, return_vals=True)
    stripes = np.array([734.55, 392.95])
    n_s, n_p, k_s, k_p = form_birefringence(stripes, wls, eps_mat1, eps_mat2)

    export_csv({'freq': f.flatten(), 'birefringence': n_p.flatten()-n_s.flatten()}, f'birefringence_{material}.csv')

    plt.plot(f.flatten(), n_p.flatten()-n_s.flatten(), label=f'bf rytov {material}')


plt.legend()
plt.show()
