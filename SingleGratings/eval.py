from pathlib import Path
import os
from functions import read_tlcsv_file, fbf_from_tl
import matplotlib.pyplot as plt

#datapath = Path('/media/alex/sda2/ProjectsOverflow/BowTie/SingleGratings')
datapath = Path('E:\CURPROJECT\BowTie\SingleGratings')

samplenames = ['StandingPrint', 'StandingV2', 'SingleGratingSlim', 'SinglegratingPhat', 'StandingNoFPV2']

datafiles = []
for root, dirs, files in os.walk(datapath):
    for name in files:
        if name.endswith('.csv'):
            datafiles.append(os.path.join(root, name))


for i, samplename in enumerate(samplenames):
    for datafile in datafiles:
        if samplename in str(datafile):
            data = read_tlcsv_file(datafile)
            for key in data:
                data[key] = data[key][5:]
            freq, ref_ind, ref_ind_err, alpha = data['freq'], data['ref_ind'], data['ref_ind_err'], data['alpha']
            freq /= 10**9

            plt.figure('Ref. ind')
            plt.subplot(len(samplenames)//5+1, len(samplenames), i + 1)
            plt.plot(freq, ref_ind, label=f'{Path(datafile).name}')
            #plt.fill_between(freq, ref_ind - ref_ind_err, ref_ind + ref_ind_err)
            if ' 0deg' in str(datafile):
                ref_ind_0deg = ref_ind
            if '90deg' in str(datafile):
                ref_ind_90deg = ref_ind

    plt.legend()
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Ref. index')

    plt.figure('Birefringence')
    plt.plot(freq, ref_ind_0deg - ref_ind_90deg, label=samplename)

plt.figure('Ref. ind')
#plt.show()

HIPS_MUT1_TL_RES_PATH = Path('/media/alex/sda2/ProjectsOverflow/BowTie/SingleGratings/HIPS MUT1 BT closer2emitter_D=2090.csv')
HIPS_MUT1_TL_RES_PATH = Path('E:\CURPROJECT\BowTie\SingleGratings/HIPS MUT1 BT closer2emitter_D=2090.csv')

a, b = 800, 550
frequencies, n_s, n_p, k_s, k_p = fbf_from_tl(HIPS_MUT1_TL_RES_PATH, a=a, b=b)
plt.figure('Birefringence')
#plt.title(f'$a={a},\ b={b}$')
#plt.plot(frequencies*10**-12, n_s, label='n_s')
#plt.plot(frequencies*10**-12, n_p, label='n_p')
plt.plot(frequencies/10**9, n_p-n_s, label=f'$Calculated, (a={a},b={b})$')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Birefringence')
plt.legend()
plt.show()
