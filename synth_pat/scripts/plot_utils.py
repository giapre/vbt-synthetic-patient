import matplotlib.pyplot as plt
import numpy as np
from synth_pat.paths import Paths

def minmaxscale(signal):
    smin = signal.min(axis=0)
    smax = signal.max(axis=0)
    signal = (signal - smin)/(smax - smin)
    return signal

def plot_bold(bold):
    bold = np.array(bold)
    bold = minmaxscale(bold)
    plt.figure(figsize=(6,12))
    plt.plot(range(bold.shape[1])+3*bold, linewidth=0.5)
    plt.show()

def basic_3d_plot(sweep_df, p1_name, 
                        p2_name, p3_name, var_to_plot,
                        ax=None):

    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    x = sweep_df[p1_name].astype(float)
    y = np.log(sweep_df[p2_name].astype(float))
    z = np.log(sweep_df[p3_name].astype(float))
    c = sweep_df[var_to_plot].astype(float)

    if c.max() == c.min():
        sizes = np.ones_like(c) * 10
    else:
        sizes = 1 + (5*(c - c.min()) / (c.max() - c.min()))**4

    # If no axis is provided, create standalone figure
    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    sc = ax.scatter(x, y, z, c=c, cmap='viridis', s=sizes, alpha=0.4)

    ax.set_xlabel(p1_name)
    ax.set_ylabel(p2_name)
    ax.set_zlabel(p3_name)
    ax.set_title(f'3D Scatter of {var_to_plot}')

    return sc  # return scatter so colorbar can be attached outside

def plot_hist_and_3d(sweep_df, p1_name, p2_name, p3_name, var_to_plot, outpath):

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 6))

    # Left: Histogram
    ax1 = fig.add_subplot(1, 2, 1)
    c = sweep_df[var_to_plot].astype(float)
    ax1.hist(c, bins=30, edgecolor='black', alpha=0.7)
    ax1.set_title(f'Distribution of {var_to_plot}')
    ax1.set_xlabel(var_to_plot)
    ax1.set_ylabel("Count")

    # Right: 3D
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    sc = basic_3d_plot(
        sweep_df, p1_name, p2_name, p3_name, var_to_plot, ax=ax2
    )

    fig.colorbar(sc, ax=ax2, shrink=0.6, label=var_to_plot)

    plt.tight_layout()
    outpath = f'{outpath}/{var_to_plot}_distr.png'
    plt.savefig(outpath)
    plt.close()


def make_tick_formatter(dt=0.5):
    """
    Function for formatting the x-ticks of time expressing them in seconds
    Inputs: the dt of the integration fo the model
    """
    def format_ticks(x, pos):
        return '{:.1f}'.format(x * dt * 1e-3)
    return format_ticks

def plot_eeg(data, channels, dt=0.5):
    """
    Plotting the EEG signals from selected channels
    """
    fig, axes = plt.subplots(len(channels), 1, figsize=(12, len(channels)), sharex=True)

    # Plot each channel's time series in a separate subplot
    for ax_i, ch_i in enumerate(channels):
        axes[ax_i].plot(data[ch_i, :], color='blue', linewidth=0.8)
        axes[ax_i].set_ylabel(f'Ch {ch_i+1}', rotation=0, labelpad=15, fontsize=8)
        axes[ax_i].set_yticks([])
        axes[ax_i].spines['top'].set_visible(False)
        axes[ax_i].spines['right'].set_visible(False)
        axes[ax_i].spines['left'].set_visible(False)
        axes[ax_i].spines['bottom'].set_visible(False)
        axes[ax_i].tick_params(left=False, bottom=False)  # Hide ticks

    # Set common x-label
    axes[-1].xaxis.set_major_formatter(make_tick_formatter(dt))
    axes[-1].set_xlabel("Time")

    plt.tight_layout()
    #plt.savefig(results_path+'first_eeg.png')
    plt.show()

    
def basic_3d_sweep_plot(sweep_df, p1_name, 
    p2_name, p3_name, var_to_plot):
    # Use interactive notebook backend
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    #%matplotlib widget

    # Extract coordinates and variable to plot
    x = sweep_df[p1_name].astype(float)
    y = sweep_df[p2_name].astype(float)
    z = sweep_df[p3_name].astype(float)
    c = sweep_df[var_to_plot].astype(float)

    sizes = 1 + (5*(c - c.min()) / (c.max() - c.min()))**4

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x, y, z, c=c, cmap='viridis', s=sizes, alpha=0.4)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label(var_to_plot)

    ax.set_xlabel(p1_name)
    ax.set_ylabel(p2_name)
    ax.set_zlabel(p3_name)

    plt.title(f'3D Scatter of {var_to_plot}')
    plt.tight_layout()
    plt.show()

def plot_2d_heatmaps(feat_df, title, metrics, columns, index, outpath):
    """
    2d heatmaps created by fixing one value (the pivot) 
    
    :param feat_df: df with data features
    :param metrics: features to be plotted
    :param pivot: the parameter to be fixed in each plot
    :param columns: the param to be used as column of the heatmaps
    :param index: the param to be used as index of the heatmaps
    """

    fig, axes = plt.subplots(
        int(np.sqrt(len(metrics))), int(np.sqrt(len(metrics))),
        figsize=(4 * len(metrics), 4 * len(metrics)),
        sharex=True,
        sharey=True
    )
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        dfplot = feat_df.pivot(
            columns=columns,
            index=index,
            values=metric
        )

        im = ax.imshow(
            dfplot.values,
            aspect="auto",
            origin="lower"
        )

        ax.set_title(metric)
        fig.colorbar(im, ax=ax, fraction=0.046)

        # ticks
        ax.set_xticks(np.arange(len(dfplot.columns)))
        ax.set_xticklabels(dfplot.columns, rotation=90)

        ax.set_yticks(np.arange(len(dfplot.index)))
        ax.set_yticklabels(dfplot.index)

        ax.set_xlabel(columns)

    axes[0].set_ylabel(index)
    fig.suptitle(f"{title}", fontsize=14)

    plt.tight_layout()
    savepath = f'{outpath}/{title}_2d_heatmap.png'
    plt.savefig(savepath)
    plt.close()

def save_feat_and_color_by_param(params, scatter0, scatter1, feat_df, outpath):
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, param in zip(axes, params):

        if param in ['wd', 'ws', 'sigma']:
            c = np.log(feat_df[param].astype(float) )  # inverse log10
        else:
            c = feat_df[param].astype(float)

        sc = ax.scatter(
            scatter0,
            scatter1,
            c=c,
            cmap='viridis',
            alpha=0.7
        )

        ax.set_title(f'Colored by {param}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        fig.colorbar(sc, ax=ax)

    plt.tight_layout()
    savepath = f'{outpath}/pca_scatter.png'
    plt.savefig(savepath)
    #plt.show()

def plot_feat_and_color_by_param(params, scatter0, scatter1, feat_df):
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, param in zip(axes, params):

        if param in ['wd', 'ws', 'sigma']:
            c = np.log(feat_df[param].astype(float) )  # inverse log10
        else:
            c = feat_df[param].astype(float)

        sc = ax.scatter(
            scatter0,
            scatter1,
            c=c,
            cmap='viridis',
            alpha=0.7
        )

        ax.set_title(f'Colored by {param}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        fig.colorbar(sc, ax=ax)

    plt.tight_layout()
    plt.show()

def plot_pca_feat_importance(params, X_r, feat_df, importance):

    pc1_corr = [
        np.corrcoef(X_r[:, 0], feat_df[p])[0, 1]
        for p in params]

    pc2_corr = [
        np.corrcoef(X_r[:, 1], feat_df[p])[0, 1]
        for p in params]

    importance_params = importance.loc[
        importance.index.intersection(params)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    # LEFT: Importance
    axes[0].bar(params, importance_params.values)
    axes[0].set_title("Feature Importance")
    axes[0].set_ylabel("Absolute Loading")

    # RIGHT: Correlation with PC1
    axes[1].bar(params, pc1_corr)
    axes[1].set_title("Correlation with PC1")
    axes[1].set_ylabel("Pearson r")
    # RIGHT: Correlation with PC1
    axes[2].bar(params, pc2_corr)
    axes[2].set_title("Correlation with PC1")
    axes[2].set_ylabel("Pearson r")

    plt.tight_layout()
    plt.show()

    