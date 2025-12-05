"""
Utility functions for reading simulation parameters from CSV
and building Material / HeatPlate objects.
"""

import os
from typing import Tuple

import pandas as pd

from material import Material
from heat_plate import HeatPlate


def load_simulation_from_csv(csv_path: str) -> Tuple[Material, HeatPlate]:
    """
    Read the first row of a CSV file to create Material and HeatPlate objects.

    Expected columns (you can adjust names to match your real file):
        k, rho, c, h, q_gen, T_inf, L, n_nodes, dt, t_final, T_init
    """
    if not os.path.exists(csv_path):
        # Example of file-related exception handling
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Input CSV file is empty.")

    row = df.iloc[0]

    material = Material(
        k=float(row["k"]),
        rho=float(row["rho"]),
        c=float(row["c"]),
        h=float(row["h"]),
        q_gen=float(row["q_gen"]),
        T_inf=float(row["T_inf"]),
        L=float(row["L"]),
    )

    n_nodes = int(row["n_nodes"])
    dt = float(row["dt"])
    t_final = float(row["t_final"])
    T_init = float(row["T_init"]) if "T_init" in row and not pd.isna(row["T_init"]) else None

    plate = HeatPlate(
        material=material,
        n_nodes=n_nodes,
        dt=dt,
        t_final=t_final,
        T_init=T_init,
    )
    return material, plate
