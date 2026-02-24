from synth_pat.paths import Paths
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 
import os

type_of_sweep =  Paths.TYPE_OF_SWEEP
print(f'doing {type_of_sweep}')
bold_file = f"{Paths.RESULTS}/{type_of_sweep}_sweep.npz"
feat_file = f"{Paths.RESULTS}/{type_of_sweep}_extracted_features.csv"
outpath = f'{Paths.FIGURES}/{type_of_sweep}'
os.mkdir(outpath)

feat_df = pd.read_csv(feat_file, index_col=0)

fig = plt.figure(figsize=(10,10))
sns.heatmap(feat_df.corr(), cmap='coolwarm')
plt.savefig(f"{Paths.FIGURES}/{type_of_sweep}/correlation_heatmap.png")
plt.close()

plt.figure(figsize=(12,4))
plt.subplot(121)
sns.barplot(feat_df.corr()['wd'])
plt.xticks(rotation=90)
plt.title('wd correlation')
plt.subplot(122)
sns.barplot(feat_df.corr()['ws'])
plt.title('ws correlation')
plt.xticks(rotation=90);
plt.savefig(f"{Paths.FIGURES}/{type_of_sweep}/correlation_ws_wd.png")
plt.close()

from synth_pat.scripts.plot_utils import plot_hist_and_3d
p1_name = 'we'
p2_name = 'wd'
p3_name = 'ws'
for var_to_plot in feat_df.columns:
    plot_hist_and_3d(feat_df, p1_name, p2_name, p3_name, var_to_plot, outpath)

from synth_pat.scripts.plot_utils import plot_2d_heatmaps
metrics = ['VAR_FCD',
'GBC',
'L.CA_FC',
'L.CA-L.CER',
'L.CA_ALFF',
'R.CA_ALFF',
'L.PTR_ALFF',
'L.HI_ALFF',
'L.CACG_ALFF']

columns='wd'
index='ws'

for we_value in np.unique(feat_df['we']):
    plot_df = feat_df[feat_df['we']==we_value]
    title=f'we={we_value}'
    plot_2d_heatmaps(plot_df, title, metrics, columns, index, outpath)

# === PCA ===

from synth_pat.scripts.analysis_utils import do_pca

X_r, pca = do_pca(feat_df.dropna())
from synth_pat.scripts.plot_utils import save_feat_and_color_by_param
params = ['we', 'wd', 'ws']
scatter0 = X_r[:,0]
scatter1 = X_r[:,1]
save_feat_and_color_by_param(params, scatter0, scatter1, feat_df.dropna(), outpath)

from synth_pat.scripts.analysis_utils import pca_feature_importance
importance = pca_feature_importance(feat_df, pca)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
params = ['we', 'wd', 'ws']
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
axes[2].set_title("Correlation with PC2")
axes[2].set_ylabel("Pearson r")

plt.tight_layout()
savepath = f'{outpath}/pca_feat_importance.png'
plt.savefig(savepath)


