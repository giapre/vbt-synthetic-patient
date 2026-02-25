import os
import re
import glob
import numpy as np
from synth_pat.paths import Paths

# With this script I extract the single-stored signals and put them in a single file so I can use the 4, 5, 6 scripts inside scripts

type_of_sweep = 'jdopa_ws_sweep'
results_path = f'{Paths.SNAKEMAKE}/results/{type_of_sweep}'

pattern = re.compile(
    r"ws=(?P<ws>[^_]+)_njdopa_ctx=(?P<ctx>[^_]+)_njdopa_str=(?P<str>.+)\.npz"
)

files = glob.glob(os.path.join(results_path, "*.npz"))

ws = np.empty(len(files))
njdopa_ctx = np.empty(len(files))
njdopa_str = np.empty(len(files))

raws = []
bolds = []

for i, file in enumerate(files):
    match = pattern.search(os.path.basename(file))
    ws[i] = float(match.group("ws"))
    njdopa_ctx[i] = float(match.group("ctx"))
    njdopa_str[i] = float(match.group("str"))

    data = np.load(file, mmap_mode="r") 
    raws.append(data["raw"])
    bolds.append(data["bold"])

raws = np.array(raws)
bolds = np.array(bolds)

raws = raws.squeeze(-1).transpose(1, 2, 0)
bolds = bolds.squeeze(-1).transpose(1, 2, 0)

params = np.vstack((ws, njdopa_ctx, njdopa_str))

# ------------------------
# Save merged file
# ------------------------

save_path = f'{Paths.RESULTS}/{type_of_sweep}.npz'
np.savez(save_path, bold=bolds, raw=raws, params=params)

print("Saved merged file:", save_path)
print(raws.shape)
print(params.shape)

# ------------------------
# Verify file integrity
# ------------------------

test = np.load(save_path)
assert test["bold"].shape == bolds.shape
assert test["raw"].shape == raws.shape
assert test["params"].shape == params.shape

print("Verification successful. Deleting single files...")

# ------------------------
# Delete single files
# ------------------------

for file in files:
    os.remove(file)

print(f"Deleted {len(files)} individual files.")
