import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from functions import export_csv

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

datapath = Path('2mm MUT1 bowtie flipped')

datafiles = []
for root, dirs, files in os.walk(datapath):
    for name in files:
        if name.endswith('.txt'):
            datafiles.append(os.path.join(root, name))

for i, file in enumerate(datafiles):
    if 'Ref' in str(file):
        data = np.loadtxt(file)
        t, y = data[:,0], data[:,1]
        freqs, Y = fft(t, y-np.mean(y))
        plt.plot(t, y, label=file)
        export_csv({'t': t, 'y': y-np.mean(y)}, 'ref_BT_example.csv')
        Y = 20 * np.log10(np.abs(Y)) - max(20 * np.log10(np.abs(Y)))
        freqs = -1*freqs
        export_csv({'freq': freqs, 'Y': Y}, 'ref_BT_example_fft.csv')
        break

plt.legend()
plt.ylabel('Amplitude')
plt.xlabel('Time (ps)')
plt.show()

for i, file in enumerate(datafiles):
    if 'Ref' not in str(file):
        continue
    data = np.loadtxt(file)
    t, y = data[:,0], data[:,1]
    freqs, Y = fft(t, y-np.mean(y))
    freqs = -1*freqs
    #minf, maxf = np.argmin(np.abs(freqs-0.15)), np.argmin(np.abs(freqs-0.25))
    #print(file, sum(np.abs(Y[minf:maxf])))
    #print(file, np.argmin(y), np.argmax(y))
    Y = 20 * np.log10(np.abs(Y)) - max(20 * np.log10(np.abs(Y)))
    plt.plot(freqs, Y, label=str(file))

    break

plt.xlim((0, 1.5))
plt.legend()
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.show()



