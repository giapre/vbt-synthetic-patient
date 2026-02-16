import numpy as np
import pandas as pd
from paths import Paths

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
