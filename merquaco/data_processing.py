import numpy as np
import pandas as pd
from typing import Union
import tifffile as tiff
from pathlib import Path
import json


def process_input(input_data: Union[str, Path, np.ndarray, pd.DataFrame, dict]):
    """
    Helper function processes and returns relevant format for input data
    """
    if isinstance(input_data, pd.DataFrame):
        if not check_if_none(input_data):
            return input_data
    elif isinstance(input_data, np.ndarray):
        if not check_if_none(input_data):
            return input_data
    elif isinstance(input_data, dict):
        return input_data
    elif isinstance(input_data, str):
        return process_path(input_data)
    elif isinstance(input_data, Path):
        return process_path(str(input_data))
    else:
        raise TypeError("Unsupported input type. Must be a DataFrame, numpy array, dictionary, file path, or JSON.")


def process_path(path: Union[str, Path]) -> Union[pd.DataFrame, np.ndarray]:
    # Determine the type of file based on its extension
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith(('.tif', '.tiff')):
        return tiff.imread(path)
    elif path.endswith('.json'):
        with open(path, 'r') as file:
            return json.load(file)
    else:
        raise ValueError("File type not supported or file not found.")


def check_if_none(*args):
    """
    Check if all provided arguments are None.

    Parameters
    ----------
    *args: Variable number of arguments that can be of any type.

    Returns
    -------
    bool
        True if all arguments are None, False otherwise.
    """
    return all(arg is None for arg in args)
