from synth_pat.scripts.analysis_utils import make_roi_alff_df, make_roi_fc_mean_df, make_roi_fc_couples_df, reconstruct_fcd, fcd_variance_excluding_overlap
from synth_pat.paths import Paths

import pandas as pd
import numpy as np

type_of_sweep = Paths.TYPE_OF_SWEEP
print(f'Doing {type_of_sweep}')
feat_file = f"{Paths.RESULTS}/{type_of_sweep}_features.npz"

sim_data = np.load(feat_file)
sim_fc = sim_data['FC']
sim_fcd = reconstruct_fcd(sim_data['FCD'], n_windows=221)
sim_alff = sim_data['ALFF']
params = np.round(sim_data['params'],4)
print(params)

sim_gbc = np.mean(sim_fc,axis=0)
sim_var_fcd = fcd_variance_excluding_overlap(sim_fcd, window_length=60, overlap=59) #np.var(sim_fcd,axis=0)

if 'jdopa' in type_of_sweep:
    sim_df = pd.DataFrame({'GBC': sim_gbc, 'VAR_FCD': sim_var_fcd, 'ws': np.round(params[:,0],4), 'njdopa_ctx': np.round(params[:,1],4), 'njdopa_str': np.round(params[:,2],4)})
else:
    sim_df = pd.DataFrame({'GBC': sim_gbc, 'VAR_FCD': sim_var_fcd, 'we': np.round(params[:,0],4), 'wd': np.round(params[:,1],4), 'ws': np.round(params[:,2],4)})#'we': np.round(params[:,0],4), 'sigma': np.round(params[:,1],4)})#'we': np.round(params[:,0],4), 'wd': np.round(params[:,1],4), 'ws': np.round(params[:,2],4)})

fc_regions = ['PU', 'CA', 'HI', 'STG', 'CER', 'CACG', 'RACG', 'IN', 'PCG', 'POP', 'POR', 'PTR']
h_fc_regions = ['L.'+region for region in fc_regions] + ['R.'+region for region in fc_regions]

fc_mean_df = make_roi_fc_mean_df(sim_fc, h_fc_regions)
alff_df = make_roi_alff_df(sim_alff, h_fc_regions)

fc_combinations = [['PU', 'RACG'], 
                   ['PU', 'CACG'],
                   ['PU', 'IN'],
                   ['PU', 'CER'],
                   ['CA', 'RACG'],
                   ['CA', 'CACG'],
                   ['CA', 'IN'],
                   ['CA', 'PCG'],
                   ['CA', 'HI'],
                   ['CA', 'CER'],
                   ['HI', 'IN'],]

h_fc_combinations = []
for combination in fc_combinations:
    for hemi in ['L.', 'R.']:
        r0 = hemi+combination[0]
        r1 = hemi+combination[1]
        h_fc_combinations.append([r0, r1])

fc_couples_df = make_roi_fc_couples_df(sim_fc, h_fc_combinations)

final_df = pd.concat([sim_df, fc_mean_df, fc_couples_df, alff_df], axis=1)
final_df.to_csv(f'{Paths.RESULTS}/{type_of_sweep}_extracted_features.csv')
