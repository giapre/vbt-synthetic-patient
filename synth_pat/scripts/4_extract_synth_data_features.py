from scripts.analysis_utils import compute_features
import pandas as pd
import numpy as np
import os
from synth_pat.paths import Paths

wes = np.round(np.linspace(0,1,10),2)
means = np.round(np.linspace(0,50,10),2)
stds = np.round(np.linspace(0,5,10),2)

input_dir = f'{Paths.RESULTS}'
type_of_sweep = 'sigma_we'
file = os.path.join(Paths.RESULTS, f'{type_of_sweep}_sweep.npz')

data = np.load(file)
bold_all = data["bold"]
bold_all = bold_all[:,:84,:] # removing midbrain structures (SN, RF, VTA)
params = data["params"]

fc_ut, fcd_ut, zscored_ALFF, fALFF = compute_features(bold_all, 1000, 60, 59)

## SAVE THE REUSLTS
output_name = f'{Paths.RESULTS}/{type_of_sweep}_features.npz'

np.savez(output_name, 
            FC=fc_ut,
            FCD=fcd_ut,
            ALFF=zscored_ALFF,
            fALFF=fALFF,
            params=params)

assert os.path.exists(output_name), "Save failed!"
print(f'Data features from simulations saved at {output_name}')

