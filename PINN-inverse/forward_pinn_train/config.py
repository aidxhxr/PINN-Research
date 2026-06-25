import numpy as np
import torch

torch.set_num_threads(48)
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE = dict(
    beta_catenin=0.20,
    APC=1.00,
    HOXA5=0.80,
    HOXA13=0.30,
    MYC=0.30,
    RA=0.60,
    CYP26A1=0.40
)

VAR_NAMES  = ["b", "apc", "h5", "h13", "m", "r", "c"]
VAR_LABELS = [r"$\beta$-catenin", "APC", "HOXA5", "HOXA13",
              "MYC", "RA", "CYP26A1"]
