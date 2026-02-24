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
we_array = [0.        , 0.22222222, 0.44444444, 0.66666667, 0.88888889,
       1.11111111, 1.33333333, 1.55555556, 1.77777778, 2.        ]

feat_df = pd.read_csv(feat_file, index_col=0)
we_array = np.unique(feat_df['we'])
we_value = we_array[6]
feat_df = feat_df[feat_df['we']==we_value]
bolds = np.load(bold_file)
bold = bolds['bold']
raw = bolds['raw']

# ==========================
# Helpers
# ==========================

def minmaxscale(signal):
    smin = signal.min(axis=0)
    smax = signal.max(axis=0)
    return (signal - smin) / (smax - smin + 1e-12)

def stacked_timeseries(ax, signal, title):
    ax.clear()
    signal = minmaxscale(signal)
    n_regions = signal.shape[1]
    offset = 3

    ax.plot(
        np.arange(signal.shape[0])[:, None],
        10*signal + offset * np.arange(n_regions),
        linewidth=0.5
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def pivot_metric(metric):
    return feat_df.pivot(columns="wd", index="ws", values=metric)

# ==========================
# Prepare sweep maps
# ==========================

varfcd_map = pivot_metric("VAR_FCD")
meanfc_map = pivot_metric("GBC")
alff_ca_map = pivot_metric("R.CA_ALFF")
alff_stg_map = pivot_metric("R.PU_ALFF")

wd_vals = varfcd_map.columns.values
ws_vals = varfcd_map.index.values

# ==========================
# Figure layout
# ==========================

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(6, 3, width_ratios=[1.2, 1, 1])

ax_raw  = fig.add_subplot(gs[0:3, 0])
ax_bold = fig.add_subplot(gs[3:6, 0])

ax_varfcd = fig.add_subplot(gs[0:2, 1])
ax_meanfc = fig.add_subplot(gs[0:2, 2])

ax_alff_ca  = fig.add_subplot(gs[2:4, 1])
ax_alff_stg = fig.add_subplot(gs[2:4, 2])

ax_fcd = fig.add_subplot(gs[4:6, 1])
ax_fc  = fig.add_subplot(gs[4:6, 2])

# Plot static sweep heatmaps
im1 = ax_varfcd.imshow(varfcd_map.values, origin="lower", aspect="equal")
im2 = ax_meanfc.imshow(meanfc_map.values, origin="lower", aspect="equal")
im3 = ax_alff_ca.imshow(alff_ca_map.values, origin="lower", aspect="equal")
im4 = ax_alff_stg.imshow(alff_stg_map.values, origin="lower", aspect="equal")

ax_varfcd.set_title("Sweep VAR_FCD")
ax_meanfc.set_title("Sweep MEAN_FC")
ax_alff_ca.set_title("Sweep R.CA_ALFF")
ax_alff_stg.set_title("Sweep L.CA_ALFF")

for ax, im in zip(
    [ax_varfcd, ax_meanfc, ax_alff_ca, ax_alff_stg],
    [im1, im2, im3, im4]
):
    fig.colorbar(im, ax=ax, fraction=0.046)

# Add readable x/y ticks showing `we` (columns) and `sigma` (rows) values
xticks = np.arange(len(wd_vals))
yticks = np.arange(len(ws_vals))
xlabels = [f"{v:.3g}" for v in wd_vals]
ylabels = [f"{v:.3g}" for v in ws_vals]
for ax in [ax_varfcd, ax_meanfc, ax_alff_ca, ax_alff_stg]:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, fontsize=8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)

# Create initial, persistent images for FCD and FC so we can add static
# horizontal colorbars below them (these do not need updating on clicks).
im_fcd = ax_fcd.imshow(np.zeros((10, 10)), vmin=-1, vmax=1, origin="lower", aspect="equal")
im_fc  = ax_fc.imshow(np.zeros((10, 10)), vmin=-1, vmax=1,  origin="lower", aspect="equal")

# Horizontal colorbars under the bottom heatmaps
cbar_fcd = fig.colorbar(im_fcd, ax=ax_fcd, orientation='vertical', fraction=0.046, pad=0.08)
cbar_fc  = fig.colorbar(im_fc,  ax=ax_fc,  orientation='vertical', fraction=0.046, pad=0.08)

# ==========================
# Interactive update function
# ==========================

highlight_patches = []

def update_plots(wd_value, ws_value):

    global highlight_patches

    # remove old highlight
    for patch in highlight_patches:
        patch.remove()
    highlight_patches = []

    # find indices
    col_idx = np.where(wd_vals == wd_value)[0][0]
    row_idx = np.where(ws_vals == ws_value)[0][0]

    # highlight on all sweep heatmaps
    for ax in [ax_varfcd, ax_meanfc, ax_alff_ca, ax_alff_stg]:
        rect = plt.Rectangle(
            (col_idx - 0.5, row_idx - 0.5),
            1, 1,
            fill=False,
            edgecolor='red',
            linewidth=3
        )
        ax.add_patch(rect)
        highlight_patches.append(rect)

    # get simulation index
    subset = feat_df[
        (feat_df['we'] == we_value) &
        (feat_df['wd'] == wd_value) &
        (feat_df['ws'] == ws_value)
    ]

    if subset.empty:
        print("No simulation for this combination!")
        return

    idx = subset.index[0]

    bold_sel = bold[:, :84, idx]
    raw_sel = raw[:, :84, idx]

    # update time series
    stacked_timeseries(ax_raw, raw_sel, "Raw")
    stacked_timeseries(ax_bold, bold_sel, "BOLD")

    # compute FC / FCD
    fc = np.corrcoef(bold_sel.T)
    fc_mean = np.mean(fc[np.triu_indices_from(fc, k=1)])

    fcd = compute_fcd(bold_sel, window_length=60, overlap=59)
    fcd_var = fcd_variance_excluding_overlap(fcd[..., None], 60, 59)[0]

    # update matrices
    ax_fcd.clear()
    ax_fc.clear()

    # Initial empty images
    im_fcd = ax_fcd.imshow(np.zeros((10,10)), origin="lower", aspect="equal")
    im_fc  = ax_fc.imshow(np.zeros((10,10)), origin="lower", aspect="equal")

    #cbar_fcd = fig.colorbar(im_fcd, ax=ax_fcd, fraction=0.046)
    #cbar_fc  = fig.colorbar(im_fc, ax=ax_fc, fraction=0.046)

    # Update image data
    im_fcd.set_data(fcd)
    im_fcd.set_clim(vmin=fcd.min(), vmax=fcd.max())
    ax_fcd.set_title(f"FCD (var={fcd_var:.4f})")

    im_fc.set_data(fc)
    im_fc.set_clim(vmin=fc.min(), vmax=fc.max())
    ax_fc.set_title(f"FC (mean={fc_mean:.4f})")

    # Update colorbars
    #cbar_fcd.update_normal(im_fcd)
    #cbar_fc.update_normal(im_fc)

    fig.canvas.draw_idle()

# ==========================
# Click event
# ==========================

def on_click(event):
    if event.inaxes in [ax_varfcd, ax_meanfc, ax_alff_ca, ax_alff_stg]:
        x = int(round(event.xdata))
        y = int(round(event.ydata))

        if 0 <= x < len(wd_vals) and 0 <= y < len(ws_vals):
            wd_value = wd_vals[x]
            ws_value = ws_vals[y]
            update_plots(wd_value, ws_value)

fig.canvas.mpl_connect("button_press_event", on_click)

plt.tight_layout()
plt.show()
