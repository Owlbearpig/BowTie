import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt

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

datapath = Path('singlegratingPhat')

# find result files
files = [os.path.join(root, name)
               for root, dirs, files in os.walk(datapath)
               for name in files
               if name.endswith('.txt')]

for i, file in enumerate(files):
    data = np.loadtxt(file)
    t, y = data[:,0], data[:,1]
    freqs, Y = fft(t, y)
    minf, maxf = np.argmin(np.abs(freqs-0.15)), np.argmin(np.abs(freqs-0.25))
    print(file, sum(np.abs(Y[minf:maxf])))
    plt.plot(freqs, np.abs(Y), label=str(file))

plt.xlim((0, 1))
plt.legend()
plt.xlabel('Frequency (THz)')
plt.ylabel('')
plt.show()



