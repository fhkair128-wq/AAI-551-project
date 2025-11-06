# utils.py
"""
Utility functions: file handling, exceptions, and helper tools.
"""
import pandas as pd
import os

def read_input(file_path):
    """Reads simulation parameters from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file {file_path} not found.")
    data = pd.read_csv(file_path)
    return data.to_dict(orient="records")[0]

def generate_temperature(material, n, dt, t_final):
    """Generator function for temperature values over time."""
    from heatplate import HeatPlate
    hp = HeatPlate(material, n, dt, t_final)
    T_data = hp.solve_implicit()
    for step in T_data:
        yield step  # generator for temperature field
