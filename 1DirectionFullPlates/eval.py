from pathlib import Path
import os



samplenames = ['MUT1', 'S1', 'S2', 'S3']

datapath = Path('Al2O3_1_re')

datafiles = []
for root, dirs, files in os.walk(datapath):
    for name in files:
        if name.endswith('.txt'):
            datafiles.append(os.path.join(root, name))