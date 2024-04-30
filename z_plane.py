import pandas as pd
import numpy as np


def get_transcripts_per_z(transcripts: pd.DataFrame, num_planes: int = None) -> np.ndarray:
    """
    Helper function gets number of transcripts per z plane
    """
    if num_planes is None:
        num_planes = transcripts['global_z'].nunique()

    transcripts_per_z = []
    # Iterate over z-planes to count transcripts
    for z in range(num_planes):
        plane = transcripts[transcripts['global_z'] == z]
        transcripts_per_z.append(len(plane))

    return np.asarray(transcripts_per_z)

def compute_z_ratio(transcripts: pd.DataFrame, num_planes: int = 7, high_plane: int = 6, low_plane: int = 0) -> float:
    """
    Returns ratio of transcript counts between highest and lowest planes
    """
    transcripts_per_z = get_transcripts_per_z(transcripts, num_planes)
    
    if high_plane > num_planes-1:
        raise Exception(f"Specified high plane {high_plane} not in experiment z-plane range")

    z_plane_ratio = np.round(transcripts_per_z[high_plane]/ transcripts_per_z[low_plane], 3)

    return z_plane_ratio
