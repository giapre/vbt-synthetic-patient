from pathlib import Path

class Paths:
    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "data"
    RESOURCES = ROOT / "resources"
    RESULTS = ROOT / "results"
    FIGURES = ROOT / "figures"

    TYPE_OF_SWEEP = "last_small_we_no_feed_we_wd_ws_noise=0.1585_sweep"

