import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt

datapath = Path(r'/media/alex/sda2/ProjectsOverflow/BowTie/1DirectionFullPlates/FullPlates/2 mm MUT1/flipped')

datafiles = []
for root, dirs, files in os.walk(datapath):
    for name in files:
        if name.endswith('.txt'):
            datafiles.append(os.path.join(root, name))

output_dir = Path(r'/media/alex/sda2/ProjectsOverflow/BowTie/1DirectionFullPlates/FullPlates/2 mm MUT1/flipped/noFP')

for datafile in datafiles:
    data = np.loadtxt(datafile)
    data[0:len(data)-840, 1] = 0
    #plt.plot(data[:,0], data[:,1])
    #plt.show()
    np.savetxt(output_dir/Path(datafile).name, data)
