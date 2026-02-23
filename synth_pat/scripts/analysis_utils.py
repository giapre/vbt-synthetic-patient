import numpy as np
import pandas as pd
from synth_pat.paths import Paths

def zscore_scale(bold):
    bold = bold - bold.mean(axis=0, keepdims=True)
    std = bold.std(axis=0, keepdims=True, ddof=1)
    bold = bold / std
    return bold

def minmaxscale(signal):
    smin = signal.min(axis=0)
    smax = signal.max(axis=0)
    signal = (signal - smin)/(smax - smin)
    return signal

def zscore_alff_map(alff):
    mean = alff.mean(axis=0, keepdims=True)
    std = alff.std(axis=0, keepdims=True)
    std[std==0] = 1
    return (alff - mean) / std

def compute_fcd(ts, window_length=20, overlap=19):
    n_samples, n_regions = ts.shape
    #    if n_samples < n_regions:
    #        print('ts transposed')
    #        ts=ts.T
    #        n_samples, n_regions = ts.shape

    window_steps_size = window_length - overlap
    n_windows = int(np.floor((n_samples - window_length) / window_steps_size + 1))

    # upper triangle indices
    Isupdiag = np.triu_indices(n_regions, 1)    

    #compute FC for each window
    FC_t = np.zeros((int(n_regions*(n_regions-1)/2),n_windows))
    for i in range(n_windows):
        FCtemp = np.corrcoef(ts[window_steps_size*i:window_length+window_steps_size*i,:].T)
        #FCtemp = np.nan_to_num(FCtemp, nan=0)
        FC_t[:,i] = FCtemp[Isupdiag]


    # compute FCD by correlating the FCs with each other
    FCD = np.corrcoef(FC_t.T)

    return FCD

def fcd_variance_excluding_overlap(sim_fcd, window_length, overlap):
    import numpy as np
    step_size = window_length - overlap
    W = sim_fcd.shape[0]
    min_sep = int(np.ceil(window_length / step_size))
    
    idx = np.arange(W)
    dist = np.abs(idx[:, None] - idx[None, :])
    
    mask = dist >= min_sep
    
    vals = sim_fcd[mask, :]
    
    return np.var(vals, axis=0)


def compute_fc_all(bold):
    # bold: (T, R, S)
    bold = bold - bold.mean(axis=0, keepdims=True)
    std = bold.std(axis=0, keepdims=True, ddof=1)
    bold = bold / std

    T = bold.shape[0]
    fc = np.einsum("trs,tks->rks", bold, bold) / (T)

    return fc  # (R, R, S)

def compute_fcd_all(bold, window_length, overlap):
    S = bold.shape[2]
    fcd_all = []

    for s in range(S):
        fcd_all.append(
            compute_fcd(bold[:, :, s], window_length, overlap)
        )

    return np.stack(fcd_all, axis=-1)


def compute_alff_all(bold, dt):
    # bold: (Timsteps, Regions, Sweep)
    freqs = np.fft.fftfreq(bold.shape[0], dt / 1000)
    fft = np.fft.fft(bold, axis=0)
    ampl = np.abs(fft)

    pos = freqs > 0
    freqs = freqs[pos]
    ampl = ampl[pos, :, :]  # (Freq, R, S)

    LF = (freqs > 0.01) & (freqs < 0.08)
    HF = (freqs > 0) & (freqs < 0.25)

    ALFF = ampl[LF].sum(axis=0)
    fALFF = ALFF / ampl[HF].sum(axis=0)

    return ALFF, fALFF  # (R, S)

import numpy as np

def compute_features(bold, tr, window_length, overlap):
    """
    Compute FC, FCD, ALFF, fALFF features from BOLD simulations saving memory by only storing the upper triangle

    Parameters
    ----------
    bold : np.ndarray
        BOLD time series, shape (T, R, S)
    tr : float
        Repetition time in ms
    window_length : int
        Sliding window length for FCD
    overlap : int
        Sliding window overlap for FCD

    Returns
    -------
    fc_ut : np.ndarray
        Upper-triangle FC, shape (R*(R-1)//2, S)
    fcd_ut : np.ndarray
        Upper-triangle FCD, shape (num_windows*(num_windows-1)//2, S)
    zscored_ALFF : np.ndarray
        Z-scored ALFF, shape (R, S)
    fALFF : np.ndarray
        fALFF, shape (R, S)
    """

    # --- Z-score per region for FC / FCD ---
    zscaled_bold = zscore_scale(bold)  # shape (T, R, S)

    # --- FC ---
    fc = compute_fc_all(zscaled_bold)  # (R, R, S)
    # extract upper triangle indices
    triu_idx = np.triu_indices(fc.shape[0], k=1)
    fc_ut = fc[triu_idx[0][:, None], triu_idx[1][:, None], np.arange(fc.shape[2])]  # (num_edges, S)

    # --- FCD ---
    fcd = compute_fcd_all(zscaled_bold, window_length, overlap)  # (num_windows, num_windows, S)
    triu_idx_fcd = np.triu_indices(fcd.shape[0], k=1)
    fcd_ut = fcd[triu_idx_fcd[0][:, None], triu_idx_fcd[1][:, None], np.arange(fcd.shape[2])]  # (num_pairs, S)

    # --- ALFF / fALFF ---
    ALFF, fALFF = compute_alff_all(bold, tr)  # (R, S)
    zscored_ALFF = zscore_alff_map(ALFF)       # (R, S)

    return fc_ut, fcd_ut, zscored_ALFF, fALFF

def compute_cortical_emp_sim_alff_correlation(emp_res_dir, sim_res_dir):
    """
    Compute the correlation between the cortical empirical and simulated ALFF
    emp_res_dir: file with the computed empirical features
    sim_res_dir: file with the computed features from simulations
    """
    from scripts.utils import get_cortical_labels
    cortical_labels = get_cortical_labels('dk')
    ce_mask = pd.read_csv(f'{Paths.RESOURCES}/Masks/dk_sero_exc_mask.csv', index_col=0)
    used_labels = ce_mask.index.to_list()
    emp_res = np.load(emp_res_dir)
    emp_alff = emp_res['ALFF']
    sim_res = np.load(sim_res_dir)
    sim_alff = sim_res['ALFF']
    cortical_indices = [used_labels.index(lab) for lab in cortical_labels]
    corr = []
    for sim in range(sim_alff.shape[1]):
        corr.append(np.corrcoef(sim_alff[cortical_indices, sim], emp_alff[cortical_indices, 0])[0,1])

    return np.array(corr)

def compute_cortical_emp_sim_falff_correlation(emp_res_dir, sim_res_dir):
    """
    Compute the correlation between the cortical empirical and simulated ALFF
    emp_res_dir: file with the computed empirical features
    sim_res_dir: file with the computed features from simulations
    """
    from scripts.utils import get_cortical_labels
    cortical_labels = get_cortical_labels('dk')
    ce_mask = pd.read_csv(f'{Paths.RESOURCES}/Masks/dk_sero_exc_mask.csv', index_col=0)
    used_labels = ce_mask.index.to_list()
    emp_res = np.load(emp_res_dir)
    emp_alff = emp_res['fALFF']
    sim_res = np.load(sim_res_dir)
    sim_alff = sim_res['fALFF']
    cortical_indices = [used_labels.index(lab) for lab in cortical_labels]
    corr = []
    for sim in range(sim_alff.shape[1]):
        corr.append(np.corrcoef(sim_alff[cortical_indices, sim], emp_alff[cortical_indices, 0])[0,1])

    return np.array(corr)

def reconstruct_fc(fc_ut, n_regions):
    """
    fc_ut: (E, S) upper triangle (k=1)
    n_regions: R

    returns: (R, R, S)
    """
    S = fc_ut.shape[1]
    fc = np.zeros((n_regions, n_regions, S))

    triu_idx = np.triu_indices(n_regions, k=1)

    for s in range(S):
        fc[triu_idx[0], triu_idx[1], s] = fc_ut[:, s]
        fc[triu_idx[1], triu_idx[0], s] = fc_ut[:, s]  # symmetry
        np.fill_diagonal(fc[:, :, s], 1.0)

    return fc

def reconstruct_fcd(fcd_ut, n_windows):
    """
    fcd_ut: (E_fcd, S)
    n_windows: W

    returns: (W, W, S)
    """
    S = fcd_ut.shape[1]
    fcd = np.zeros((n_windows, n_windows, S))

    triu_idx = np.triu_indices(n_windows, k=1)

    for s in range(S):
        fcd[triu_idx[0], triu_idx[1], s] = fcd_ut[:, s]
        fcd[triu_idx[1], triu_idx[0], s] = fcd_ut[:, s]

    return fcd

def make_roi_alff_df(sim_alff, alff_rois):
    """
    Creates a df with the alff values only of specific regions.
    
    :param sim_alff: the dataframe containing the alff values of all regions and all simulations (shape regions, simulations)
    :param alff_rois: list of regions of interest
    """
    ce_mask = pd.read_csv(f'{Paths.RESOURCES}/Masks/dk_sero_exc_mask.csv', index_col=0)
    regions_names = ce_mask.columns.to_list()
    alff_indices = [regions_names.index(region) for region in regions_names if region in alff_rois]
    alff_df = pd.DataFrame(data=sim_alff[alff_indices, :].T, columns=[f'{roi}_ALFF' for roi in alff_rois])

    return alff_df

def make_roi_fc_couples_df(sim_fc, fc_combinations):
    """
    Creates a df with the fc values only of specific couple of regions.
    
    :param sim_fc: the fc extracted from the simulations (shape regions, regions, simulations)
    :param fc_combinations: Descrizione
    """
    FC = reconstruct_fc(sim_fc, 84)
    ce_mask = pd.read_csv(f'{Paths.RESOURCES}/Masks/dk_sero_exc_mask.csv', index_col=0)
    regions_names = ce_mask.columns.to_list()
    fc_dic = {}

    for combination in fc_combinations:
        idx0 = regions_names.index(combination[0])
        idx1 = regions_names.index(combination[1])
        fc = FC[idx0, idx1]
        name = f'{combination[0]}-{combination[1]}'

        fc_dic.update({name:fc})

    fc_df = pd.DataFrame(fc_dic)

    return fc_df

def make_roi_fc_mean_df(sim_fc, fc_regions):
    """
    Creates a df with the mean fc value of a specific region.
    
    :param sim_fc: the fc extracted from the simulations (shape regions, regions, simulations)
    :param fc_combinations: Descrizione
    """
    FC = reconstruct_fc(sim_fc, 84)
    ce_mask = pd.read_csv(f'{Paths.RESOURCES}/Masks/dk_sero_exc_mask.csv', index_col=0)
    regions_names = ce_mask.columns.to_list()
    fc_dic = {}

    for region in fc_regions:
        idx1 = regions_names.index(region)
        fc = (np.mean(FC[idx1], axis=0)) / 2
        name = f'{region}_FC'
        fc_dic.update({name:fc})

    fc_df = pd.DataFrame(fc_dic)

    return fc_df

def drop_high_corr_features(feat_df):
    """
    Drop highly correlated data features 
    
    :param feat_df: the df containing simulations x data features
    """
    import numpy as np

    # Compute absolute correlation matrix
    corr_matrix = feat_df.corr().abs()

    # Keep only upper triangle (avoid duplicates)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find columns to drop
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

    # Drop them
    feat_df_reduced = feat_df.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} features")

    return feat_df_reduced

def do_pca(X):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X_scaled = StandardScaler().fit_transform(X)

    # PCA projection
    pca = PCA(n_components=2)
    X_r = pca.fit_transform(X_scaled)

    return X_r, pca

def pca_feature_importance(X, pca):
    loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=X.columns)

    importance = np.sqrt(
    loadings['PC1']**2 + loadings['PC2']**2)

    importance = importance.sort_values(ascending=False)

    return importance