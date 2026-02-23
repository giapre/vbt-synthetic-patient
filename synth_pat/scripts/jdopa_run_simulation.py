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

type_of_sweep = 'jdopa_ws'

ws = 10**np.linspace(-3,0,10)
njdopa_ctx_arr = np.round(10**np.linspace(-3,2,10),3)
njdopa_str_arr = np.round(10**np.linspace(-3,2,10),3)

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

regions_names = W.columns.to_list()

# ------------------------
# Model setup
# ------------------------

setup = {
    "Seids": [],
    "idelays": [],
    "params": gm.sigm_d1d2sero_default_theta,
    "v_c": 3.9,
    "horizon": 650,
    "num_item": ws.shape[0],
    "dt": 0.1,
    "num_skip": 10,
    "num_time": 300000,
    "init_state": jp.array(
        [.01, -55.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ).reshape(10, 1),
    "noise": 0.025,
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

for njdopa_ctx in njdopa_ctx_arr:
    for njdopa_str in njdopa_str_arr:
        JJdopa = np.ones(len(regions_names)) * njdopa_ctx
        for region in ['L.PU', 'R.PU', 'L.CA', 'R.CA', 'L.PA', 'R.PA', 'L.AC', 'R.AC']:
            idx = regions_names.index(region)
            JJdopa[idx] = njdopa_str

        theta = gm.sigm_d1d2sero_default_theta._replace(
            I=46.5,
            Ja=Ja,
            Jsa=Ja,
            Jsg=13,
            Jg=0,
            Rd1=Rd1,
            Rd2=Rd2,
            Rs=Rsero,
            Sd1=-10.0,
            Sd2=-10.0,
            Ss=-40.0,
            Zd1=0.5,
            Zd2=1.0,
            Zs=0.25,
            we=0.25,
            wi=1.,
            wd=1,
            ws=ws,
            sigma_V=setup["noise"],
            sigma_u=0.1 * setup["noise"],
        )

        setup["params"] = theta

        tic = time.time()
        print(f"Running simulations for njdopa_ctx {njdopa_ctx} and {njdopa_str}")
        bold, raw = run_bold_sweep((theta, setup))
        toc = time.time()

        bold = np.asarray(bold)
        raw = np.asarray(raw)

        output_file = os.path.join(
            RESULTS_DIR,
            f"{type_of_sweep}_sweep/njdopa_ctx={njdopa_ctx}_njdopa_str={njdopa_str}"
        )

        np.savez(output_file, bold=bold, raw=raw)

        print(f"Finished in {toc - tic} seconds")
