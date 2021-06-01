import pathlib
from pathlib import Path
import numpy as np
import os

def process(datafile):
    data = np.loadtxt(datafile)
    # remove ooffset
    data[:,1] -= np.mean(data[:,1])
    np.savetxt(output_dir / datafile, data)

base = Path('1DirectionFullPlates/FullPlatesFocus')

output_dir = Path('1DirectionFullPlates/FullPlatesFocus_offsetcorrected')
# (Re)Create folder structure for output
for root, dirs, files in os.walk(base):
    for dir_ in dirs:
        new_dir = output_dir / root / dir_
        pathlib.Path(new_dir).mkdir(parents=True, exist_ok=True)

# find all data files
datafiles = [os.path.join(root, name)
     for root, dirs, files in os.walk(base)
        for name in files
            if name.endswith((".txt"))]

for i, datafile in enumerate(datafiles):
    if i % 10 == 0:
        print(i, len(datafiles))
    process(datafile)
