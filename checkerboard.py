import numpy as np
import pandas as pd

class Checkerboard:
    """
    Class for checkerboard pattern analysis methods
    """
    FOV_DIMENSIONS = 202

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_cb_list(transcripts: pd.DataFrame, num_planes: int = 7):
        """
        Performs checkerboard analysis across z-planes
        """
        pixel_dimensions = Checkerboard.get_image_dimensions(transcripts)

        cb_list = []
        for plane_number in range(num_planes):
            # Subset transcripts by z plane
            plane = transcripts[transcripts['global_z'] == plane_number]
            plane_x = np.asarray(plane['global_x'])
            plane_y = np.asarray(plane['global_y'])
            _, cb_chunk_x = Checkerboard.get_chunk_values(plane_x, pixel_dimensions)
            _, cb_chunk_y = Checkerboard.get_chunk_values(plane_y, pixel_dimensions)
            cb_x = np.min(cb_chunk_x) / np.max(cb_chunk_x)
            cb_y = np.min(cb_chunk_y) / np.max(cb_chunk_y)
            cb_list.append((cb_x, cb_y))

        return cb_list

    @staticmethod
    def get_chunk_values(transcripts: pd.DataFrame, pixel_dimensions: int):
        """
        Calculates histogram and checkerboard chunk values for transcripts DataFrame
        """
        # Get transcripts histogram
        cb_hist, bins = np.histogram(transcripts, bins=np.arange(0, pixel_dimensions, 1), density=True)
        # Threshold bins to only consider "non-empty" bins
        bins_thr = bins[:-1][cb_hist > 0.0001]

        # Extract 'non-empty' histogram values
        hist_thr = cb_hist[int(np.amin(bins_thr)):int(np.amax(bins_thr))]

        # Initialize checkerboard metric values
        cb_chunk = np.zeros(Checkerboard.FOV_DIMENSIONS)
        fovs = int(np.floor(len(hist_thr)/Checkerboard.FOV_DIMENSIONS))

        # Calculate checkerboard metric values
        for fov in range(fovs):
            # Chunks are hist_thr values at 1um intervals within an FOV
            fov_chunk = hist_thr[(0+fov)*Checkerboard.FOV_DIMENSIONS:(fov+1)*Checkerboard.FOV_DIMENSIONS]
            # Determine checkerboard metric at each 1um within FOV
            cb_chunk = cb_chunk + fov_chunk/np.mean(fov_chunk)

        return cb_hist, cb_chunk

    @staticmethod
    def get_image_dimensions(normalized_transcripts: pd.DataFrame):
        """
        Helper method to calculate dimensions for image based on max (x,y) values
        """
        max_x = normalized_transcripts['global_x'].max()
        max_y = normalized_transcripts['global_y'].max()

        return np.round(max(max_x, max_y)) + 1000
    
    @staticmethod
    def get_cb_vals_all_z(transcripts: pd.DataFrame):
        """
        Performs checkerboard analysis in all z-planes collapsed
        """
        pixel_dimensions = Checkerboard.get_image_dimensions(transcripts)
        x = np.asarray(transcripts['global_x'])
        y = np.asarray(transcripts['global_y'])

        _, cb_chunk_x = Checkerboard.get_chunk_values(x, pixel_dimensions)
        _, cb_chunk_y = Checkerboard.get_chunk_values(y, pixel_dimensions)
        cb_x = np.nanmin(cb_chunk_x) / np.max(cb_chunk_x)
        cb_y = np.nanmin(cb_chunk_y) / np.max(cb_chunk_y)

        return (cb_x, cb_y)
