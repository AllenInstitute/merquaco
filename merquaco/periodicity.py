import numpy as np
import pandas as pd


def get_periodicity_list(transcripts: pd.DataFrame, num_planes: int = 7, fov_dimensions: tuple = (202, 202)):
    """
    Performs periodicity analysis across z-planes

    Parameters
    ----------
    transcripts : pd.DataFrame
        Transcripts table
    num_planes : int, optional
        Number of planes to get periodicity values for. Default is 7.
    fov_dimensions : tuple, optional
        (x, y) dimensions for FOV tiles. Default is (202, 202).

    Returns
    -------
    periodicity_list : list of tuples
        Periodicity values in (x,y) for each z-plane
    """
    image_dimensions = get_image_dimensions(transcripts)

    periodicity_list = []
    for plane_number in range(num_planes):
        # Subset transcripts by z plane
        plane = transcripts[transcripts['global_z'] == plane_number]
        plane_x = np.asarray(plane['global_x'])
        plane_y = np.asarray(plane['global_y'])
        _, chunk_x = get_chunk_values(plane_x, image_dimensions, fov_dimensions=fov_dimensions[0])
        _, chunk_y = get_chunk_values(plane_y, image_dimensions, fov_dimensions=fov_dimensions[1])
        periodicity_x = np.nanmin(chunk_x) / np.max(chunk_x)
        periodicity_y = np.nanmin(chunk_y) / np.max(chunk_y)
        periodicity_list.append((periodicity_x, periodicity_y))

    return periodicity_list


def get_chunk_values(transcripts: np.ndarray, image_dimensions: int, fov_dimensions: int = 202):
    """
    Calculates histogram and periodicity chunk values for transcripts DataFrame

    Parameters
    ----------
    transcripts : np.ndarray
        Array of transcript x or y coordinates
    image_dimensions : int
        Size of image
    fov_dimensions : int, optional
        Field of view dimensions. Default is 202 (microns).

    Returns
    -------
    tuple of np.ndarray
        periodicity_hist : np.ndarray
            Histogram values
        periodicity_chunk : np.ndarry
            Periodicity metric chunk values
    """
    # Get transcripts histogram
    periodicity_hist, bins = np.histogram(transcripts, bins=np.arange(0, image_dimensions, 1), density=True)
    # Threshold bins to only consider "non-empty" bins
    bins_thr = bins[:-1][periodicity_hist > 0.0001]

    # Extract 'non-empty' histogram values
    hist_thr = periodicity_hist[int(np.amin(bins_thr)):int(np.amax(bins_thr))]

    # Initialize periodicity metric values
    periodicity_chunk = np.zeros(fov_dimensions)
    fovs = int(np.floor(len(hist_thr)/fov_dimensions))

    # Calculate periodicity metric values
    for fov in range(fovs):
        # Chunks are hist_thr values at 1um intervals within an FOV
        fov_chunk = hist_thr[(0+fov)*fov_dimensions:(fov+1)*fov_dimensions]
        # Determine periodicity metric at each 1um within FOV
        periodicity_chunk = periodicity_chunk + fov_chunk/np.mean(fov_chunk)

    return periodicity_hist, periodicity_chunk


def get_image_dimensions(transcripts: pd.DataFrame):
    """
    Helper method to calculate dimensions for image based on max (x,y) values

    Parameters
    ----------
    transcripts : pd.DataFrame
        Transcripts table

    Returns
    -------
    int
        Image dimensions in microns
    """
    max_x = transcripts['global_x'].max()
    max_y = transcripts['global_y'].max()

    return np.round(max(max_x, max_y)) + 1000


def get_periodicity_vals_all_z(transcripts: pd.DataFrame):
    """
    Performs periodicity analysis in all z-planes collapsed

    Parameters
    ----------
    transcripts : pd.DataFrame
        Transcripts table

    Returns
    -------
    tuple
        Periodicity metric in (x, y)
    """
    image_dimensions = get_image_dimensions(transcripts)
    x = np.asarray(transcripts['global_x'])
    y = np.asarray(transcripts['global_y'])

    _, periodicity_chunk_x = get_chunk_values(x, image_dimensions)
    _, periodicity_chunk_y = get_chunk_values(y, image_dimensions)
    periodicity_x = np.nanmin(periodicity_chunk_x) / np.max(periodicity_chunk_x)
    periodicity_y = np.nanmin(periodicity_chunk_y) / np.max(periodicity_chunk_y)
    return (periodicity_x, periodicity_y)
