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

datapath = Path('BowTieSetup/polarisator')


datafiles = [os.path.join(root, name)
               for root, dirs, files in os.walk(datapath)
               for name in files
               if name.endswith('.txt')]

export_data = {}
for i, datafile in enumerate(datafiles):
    data = np.loadtxt(datafile)
    t, y = data[:,0], data[:,1]
    freqs, Y = fft(t, y)
    if i == 0:
        export_data['freqs'] = freqs

    plt.plot(freqs, 10*np.log10(np.abs(Y)), label=str(datafile))

    export_data[str(datafile).split('-')[-3]] = 10*np.log10(np.abs(Y))

export_csv(export_data, 'wiregrid_fft.csv')
plt.legend()
plt.xlim((0, 2))
plt.xlabel('Frequency (THz)')
plt.ylabel('amplitude (dB)')
plt.show()



