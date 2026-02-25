import os
import sys
import numpy as np
import pandas as pd
import jax.numpy as jp
import scipy.sparse
import synth_pat.scripts.gast_model as gm

from synth_pat.paths import Paths
from synth_pat.scripts.simulation_utils import (
#    stack_connectomes,
#    setup_delays,
#    setup_ja,
     run_bold_sweep,
#    setup_receptors,
#    adjust_ja_for_midbrain
)

# ------------------------
# Parse Snakemake args
# ------------------------

ws = float(sys.argv[1])
njdopa_ctx = float(sys.argv[2])
njdopa_str = float(sys.argv[3])
type_of_sweep = sys.argv[4]
output_file = sys.argv[5]

RESULTS_DIR = Paths.RESULTS
DATA_DIR = Paths.DATA

# ------------------------
# Load data
# ------------------------

#W = pd.read_csv(os.path.join(DATA_DIR, "averaged_weights_with_sero_and_dopa.csv"), index_col=0)
L = pd.read_csv(os.path.join(DATA_DIR, "averaged_lengths_with_sero_and_dopa.csv"), index_col=0)
#zscores = pd.read_csv(os.path.join(DATA_DIR, "averaged_cortical_zscores.csv"), index_col=0)

regions_names = L.columns.to_list()

# ------------------------
# Model setup
# ------------------------

setup = {
    "Seids": [],
    "idelays": [],
    "params": gm.sigm_d1d2sero_default_theta,
    "v_c": 3.9,
    "horizon": 650,
    "num_item": 1,
    "dt": 0.1,
    "num_skip": 10,
    "num_time": 300000,
    "init_state": jp.array([.01, -55.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(10,1),
    "noise": 0.0631,
}

Ceids = np.load('Ceids.npy')#stack_connectomes(W)
setup["Seids"] = scipy.sparse.csr_matrix(Ceids)
setup["idelays"] = np.load('idelays.npy') #setup_delays(L, Ceids, setup["v_c"], setup["dt"])

#mean_Ja = 12
#std_Ja = 1.2
#pid = 0
#Ja = setup_ja(zscores, W, pid, mean_Ja, std_Ja)
#Ja = adjust_ja_for_midbrain(Ja, regions_names)
Ja = np.load('Ja.npy')

#Rd1, Rd2, Rsero = setup_receptors()
Rd1, Rd2, Rsero = np.load('Receptors.npy')

# ------------------------
# Set parameters
# ------------------------

JJdopa = np.ones(len(regions_names)) * njdopa_ctx
for region in ['L.PU', 'R.PU', 'L.CA', 'R.CA', 'L.PA', 'R.PA', 'L.AC', 'R.AC']:
    idx = regions_names.index(region)
    JJdopa[idx] = njdopa_str
JJdopa = JJdopa[:, None]

theta = gm.sigm_d1d2sero_default_theta._replace(
    I=46.5,
    Ja=Ja,
    Jsa=Ja,
    Jsg=13,
    Jg=0,
    Jdopa=100000*JJdopa,
    Rd1=Rd1,
    Rd2=Rd2,
    Rs=Rsero,
    Sd1=-10.0,
    Sd2=-10.0,
    Ss=-40.0,
    Zd1=0.5,
    Zd2=1.0,
    Zs=0.25,
    we=0.3333,
    wi=1.,
    wd=1,
    ws=ws,
    sigma_V=setup["noise"],
    sigma_u=0.1 * setup["noise"],
)

setup["params"] = theta

# ------------------------
# Run simulation
# ------------------------

bold, raw = run_bold_sweep((theta, setup))

bold = np.asarray(bold)
raw = np.asarray(raw)

os.makedirs(os.path.dirname(output_file), exist_ok=True)

np.savez(output_file, bold=bold, raw=raw)
