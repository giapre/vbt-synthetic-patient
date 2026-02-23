import os
import sys
import time
import numpy as np
import pandas as pd
import jax.numpy as jp
import scipy.sparse

from synth_pat.paths import Paths
from simulation_utils import (
    stack_connectomes,
    setup_delays,
    setup_ja,
    run_bold_sweep,
    setup_receptors
)
import gast_model as gm

# ------------------------
# Paths and inputs
# ------------------------

RESULTS_DIR = Paths.RESULTS
DATA_DIR = Paths.DATA

# ------------------------
# Parameter sweeps
# ------------------------

type_of_sweep = 'restricted_sigma_we'

g_p1, g_p2 = np.mgrid[0:0.5:16j, -2:0:16j]

jp_p1 = jp.array(g_p1.ravel())
jp_p2 = jp.array(10**g_p2.ravel())

# ------------------------
# Load data
# ------------------------

W = pd.read_csv(
    os.path.join(DATA_DIR, "averaged_weights_with_sero_and_dopa.csv"),
    index_col=0
)
L = pd.read_csv(
    os.path.join(DATA_DIR, "averaged_lengths_with_sero_and_dopa.csv"),
    index_col=0
)
zscores = pd.read_csv(
    os.path.join(DATA_DIR, "averaged_cortical_zscores.csv"),
    index_col=0,
)

# ------------------------
# Model setup
# ------------------------

setup = {
    "Seids": [],
    "idelays": [],
    "params": gm.sigm_d1d2sero_default_theta,
    "v_c": 3.9,
    "horizon": 650,
    "num_item": jp_p1.shape[0],
    "dt": 0.1,
    "num_skip": 10,
    "num_time": 300000,
    "init_state": jp.array(
        [.01, -55.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ).reshape(10, 1),
    "noise": jp_p2,
}

Ceids = stack_connectomes(W)
setup["Seids"] = scipy.sparse.csr_matrix(Ceids)
setup["idelays"] = setup_delays(L, Ceids, setup["v_c"], setup["dt"])

mean_Ja = 12
std_Ja = 1.2
pid = 0
Ja = setup_ja(zscores, W, pid, mean_Ja, std_Ja)

Rd1, Rd2, Rsero = setup_receptors()

# ------------------------
# Run sweep
# ------------------------

theta = gm.sigm_d1d2sero_default_theta._replace(
    I=46.5,
    Ja=Ja,
    Jsa=Ja,
    Jsg=13,
    Jg=0,
    Rd1=0,
    Rd2=0,
    Rs=0,
    Sd1=-10.0,
    Sd2=-10.0,
    Ss=-40.0,
    Zd1=0.5,
    Zd2=1.0,
    Zs=0.25,
    we=jp_p1,
    wi=1.,
    wd=0,
    ws=0,
    sigma_V=setup["noise"],
    sigma_u=0.1 * setup["noise"],
)

setup["params"] = theta

tic = time.time()
print(f"Running {type_of_sweep} simulations")
bold, raw = run_bold_sweep((theta, setup))
toc = time.time()

bold = np.asarray(bold)
raw = np.asarray(raw)

output_file = os.path.join(
    RESULTS_DIR,
    f"{type_of_sweep}_sweep"
)

np.savez(output_file, bold=bold, raw=raw, params=np.stack((jp_p1, jp_p2), axis=1))

print(f"Finished in {toc - tic} seconds")
