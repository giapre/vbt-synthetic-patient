from pathlib import Path

class Paths:
    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "data"
    RESOURCES = ROOT / "resources"
    RESULTS = ROOT / "results"
    #FIGURES = ROOT / "figures"

