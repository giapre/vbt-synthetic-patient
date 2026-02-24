import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from synth_pat.scripts.analysis_utils import compute_fcd, fcd_variance_excluding_overlap
from synth_pat.paths import Paths

# ==========================
# Load data
# ==========================

type_of_sweep = Paths.TYPE_OF_SWEEP
bold_file = f"{Paths.RESULTS}/{type_of_sweep}.npz"
feat_file = f"{Paths.RESULTS}/{type_of_sweep}_extracted_features.csv"

feat_df = pd.read_csv(feat_file, index_col=0)

# choose one simulation
we_value = np.unique(feat_df['we'])[6]
wd_value = np.unique(feat_df['wd'])[3]
ws_value = np.unique(feat_df['ws'])[5]

subset = feat_df[
    (feat_df['we'] == we_value) &
    (feat_df['wd'] == wd_value) &
    (feat_df['ws'] == ws_value)
]

idx = subset.index[0]

bolds = np.load(bold_file)
bold = bolds['bold'][:, :84, idx]

# ==========================
# Helper
# ==========================

def minmaxscale(signal):
    smin = signal.min(axis=0)
    smax = signal.max(axis=0)
    return (signal - smin) / (smax - smin + 1e-12)

def stacked_timeseries(ax, signal):
    signal = minmaxscale(signal)
    n_regions = signal.shape[1]
    offset = 3

    ax.plot(
        np.arange(signal.shape[0])[:, None],
        7 * signal + offset * np.arange(n_regions),
        linewidth=0.5
    )
    ax.set_xticks([])
    ax.set_yticks([])

# ==========================
# Compute FC / FCD
# ==========================

fc = np.corrcoef(bold.T)
fc_mean = np.mean(fc[np.triu_indices_from(fc, k=1)])

fcd = compute_fcd(bold, window_length=60, overlap=59)
fcd_var = fcd_variance_excluding_overlap(fcd[..., None], 60, 59)[0]

# ==========================
# Plot
# ==========================

fig, axes = plt.subplots(1, 3, figsize=(12, 6))

ax_bold, ax_fc, ax_fcd = axes

# BOLD
stacked_timeseries(ax_bold, bold)
ax_bold.set_title("BOLD Timeseries")

# FC
im_fc = ax_fc.imshow(fc, origin="upper", aspect="equal")
ax_fc.set_title(f"FC (mean = {fc_mean:.4f})")
fig.colorbar(im_fc, ax=ax_fc, fraction=0.046)

# FCD
im_fcd = ax_fcd.imshow(fcd, origin="upper", aspect="equal")
ax_fcd.set_title(f"FCD (variance = {fcd_var:.4f})")
fig.colorbar(im_fcd, ax=ax_fcd, fraction=0.046)

plt.suptitle(we_value)

plt.tight_layout()
plt.show()