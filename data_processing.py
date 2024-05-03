import numpy as np
import pandas as pd
from typing import Union
import tifffile as tiff


def process_input(input_data: Union[str, np.ndarray, pd.DataFrame]):
    if isinstance(input_data, pd.DataFrame):
        return process_dataframe(input_data)
    elif isinstance(input_data, np.ndarray):
        return process_image_array(input_data)
    elif isinstance(input_data, str):
        return process_path(input_data)
    else:
        raise TypeError("Unsupported input type. Must be a DataFrame, numpy array, or file path.")

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df

def process_image_array(image_array: np.ndarray) -> np.ndarray:
    return image_array

def process_path(path: str) -> Union[pd.DataFrame, np.ndarray]:
    # Determine the type of file based on its extension
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith(('.png', '.jpg', '.jpeg')):
        return tiff.imread(path)
    else:
        raise ValueError("File type not supported or file not found.")

# # Example usages:
# df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
# result_df = process_input(df)

# image = np.random.rand(100, 100)
# result_img = process_input(image)

# result_from_csv = process_input('data.csv')  # Assuming 'data.csv' is a valid path
# result_from_image = process_input('image.jpg')  # Assuming 'image.jpg' is a valid path
