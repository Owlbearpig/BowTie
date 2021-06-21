from pathlib import Path
import os
from functions import read_tlcsv_file, fbf_from_tl, export_csv
import matplotlib.pyplot as plt
import numpy as np

datapath = Path('/media/alex/sda2/ProjectsOverflow/BowTie/1DirectionFullPlates/FullPlates')
#datapath = Path('E:\CURPROJECT\BowTie\1DirectionFullPlates\FullPlatesFocus_offsetcorrected\FullPlatesFocus\tlRes')

samplenames = ['MUT1', 'S2', 'S3']
plot_labels = {'MUT1': 'ZigZag pattern, 2 mm', 'S2': 'Same direction, 2 mm', 'S3': 'Same direction, 8 mm'}

datafiles = []
for root, dirs, files in os.walk(datapath):
    for name in files:
        if name.endswith('.csv') and 'noFP' in str(name):
            datafiles.append(os.path.join(root, name))

export_data = {}
for i, samplename in enumerate(samplenames):
    for datafile in datafiles:
        if samplename in str(datafile):
            data = read_tlcsv_file(datafile)

            freq = data['freq']
            freq /= 10**9

            f_slice = np.where(freq > 75)

            freq, ref_ind, alpha = data['freq'][f_slice], data['ref_ind'][f_slice], data['alpha'][f_slice]

            plt.figure('Ref. ind')
            plt.subplot(len(samplenames)//5+1, len(samplenames), i + 1)
            plt.plot(freq, ref_ind, label=f'{Path(datafile).name}')
            if ' 0deg' in str(datafile):
                ref_ind_0deg = ref_ind
            if '90deg' in str(datafile):
                ref_ind_90deg = ref_ind

    plt.legend()
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Ref. index')
    plt.xlim((60, 350))
    plt.figure('Birefringence')

    plt.plot(freq, ref_ind_0deg - ref_ind_90deg, label=plot_labels[samplename])

    try:
        export_data['freq']
    except KeyError:
        export_data['freq'] = freq

    export_data[f'bf_{plot_labels[samplename]}'] = ref_ind_0deg-ref_ind_90deg

export_csv(export_data, 'fullplates_bf.csv')

#plt.figure('Ref. ind')

plt.figure('Birefringence')
plt.xlim((60, 350))
plt.xlabel('Frequency (GHz)')
plt.ylabel('Birefringence')
plt.legend()
plt.show()
