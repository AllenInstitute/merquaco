import pandas as pd
import numpy as np


def get_transcripts_per_z(transcripts: pd.DataFrame, num_planes: int = None) -> np.ndarray:
    """
    Iterates through z-planes and returns array of transcript counts per plane

    Parameters
    ----------
    transcripts : pd.DataFrame
        Transcripts table
    num_planes : int, optional
        Number of z-planes to count transcripts in. Default is None.

    Returns
    -------
    transcripts_per_z : np.ndarray
        Array of transcript counts per z plane
    """
    if num_planes is None:
        num_planes = transcripts['global_z'].nunique()

    transcripts_per_z = []
    # Iterate over z-planes to count transcripts
    for z in range(num_planes):
        plane = transcripts[transcripts['global_z'] == z]
        transcripts_per_z.append(len(plane))

    return np.asarray(transcripts_per_z)


def compute_z_ratio(transcripts: pd.DataFrame,
                    num_planes: int = None,
                    high_plane: int = 6,
                    low_plane: int = 0) -> float:
    """
    Returns ratio of transcript counts between highest and lowest planes

    Parameters
    ----------
    transcripts : pd.DataFrame
        Transcripts table
    num_planes : int, optional
        Number of z-planes to count transcripts in. Default is None.
    high_plane : int, optional
        Highest z-plane; numerator in z-ratio division. Default is 6.
    low_plane : int, optional
        Lowest z-plane; denominator in z-ratio division. Default is 0.

    Returns
    -------
    z_plane_ratio : float
        Ratio of transcript counts in high/low z-planes

    Raises
    ------
    ValueError
        If `high_plane` is outside of z-plane range for experiment
    """
    transcripts_per_z = get_transcripts_per_z(transcripts, num_planes)

    if high_plane > (num_planes - 1):
        raise ValueError(f"Specified high plane {high_plane} not in experiment z-plane range")

    z_plane_ratio = np.round(transcripts_per_z[high_plane] / transcripts_per_z[low_plane], 3)

    return z_plane_ratio
