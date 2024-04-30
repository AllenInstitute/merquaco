import pandas as pd
import numpy as np


class ZPlane:
    """
    Class for z-plane specific analysis methods

    Methods
    -------
    get_transcripts_per_z(transcripts, num_planes) -> np.ndarray
        Gets number of transcripts per z plane
    get_z_plane_ratio(transcripts_per_z, high_plane, low_plane) -> float
        Returns ratio of transcript counts in high and low z planes
    compute_z_ratio(transcripts) -> float
        Returns ratio of transcript counts between highest and lowest planes
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_transcripts_per_z(transcripts: pd.DataFrame, num_planes: int = None) -> np.ndarray:
        """
        Helper function gets number of transcripts per z plane

        Parameters
        ----------
        transcripts : pd.DataFrame
            DataFrame of detected transcripts
        num_planes : int, optional
            Number of z-planes, default is None

        Returns
        -------
        transcripts_per_z : np.ndarray
            Array of transcript counts per z-plane

        Examples
        --------
        >>> ts_per_z = ZPlane.get_transcripts_per_z(transcripts)
        >>> ts_per_z
        array([21958813, 22998356, 23800627, 24365519, 23836891, 17464100, 9575010])
        """
        if num_planes is None:
            num_planes = transcripts['global_z'].nunique()

        transcripts_per_z = []
        # Iterate over z-planes to count transcripts
        for z in range(num_planes):
            plane = transcripts[transcripts['global_z'] == z]
            transcripts_per_z.append(len(plane))

        return np.asarray(transcripts_per_z)

    @staticmethod
    def get_z_plane_ratio(transcripts_per_z: np.ndarray, high_plane: int = 6, low_plane: int = 0) -> float:
        """
        Returns ratio of transcript counts between highest and lowest z-planes

        Parameters
        ----------
        transcripts_per_z : np.ndarray
            Array of transcript counts per z-plane. Array index corresponds to z-plane number
        high_plane : int, optional
            Nominator z-plane, default is 6
        low_plane : int, optional
            Denominator z-plane, default is 0

        Returns
        -------
        float
            Ratio of transcript counts between high and low z-planes

        Examples
        --------
        >>> ts_per_z = ZPlane.get_transcripts_per_z(transcripts)
        >>> ts_per_z
        array([21958813, 22998356, 23800627, 24365519, 23836891, 17464100, 9575010])

        Using `high_plane` = 6 and `low_plane` = 0
        >>> z_ratio = ZPlane.get_z_plane_ratio(ts_per_z)
        >>> z_ratio
        0.061

        Using `high_plane` = 4 and `low_plane` = 0
        >>> z_ratio = ZPlane.get_z_plane_ratio(ts_per_z, 4)
        >>> z_ratio
        0.288

        Using `high_plane` = 7 and `low_plane` = 0
        >>> z_ratio = ZPlane.get_z_plane_ratio(ts_per_z, 7)
        ValueError: `high_plane` not in experiment z-plane range

        Notes
        -----
        `high_plane` and `low_plane` default to 6 and 0, respectively, representing the z-span imaged in a 10um
        experiment. Different thickness selections on the MERSCOPE image a different number of z-planes, but these
        parameters can compare any two z-planes, not just the actual highest and lowest.
        """
        highest = len(transcripts_per_z)
        if highest == high_plane + 1:
            return np.round(transcripts_per_z[high_plane] / transcripts_per_z[low_plane], 3)
        else:
            raise ValueError('`high_plane` not in experiment z-plane range')

    @staticmethod
    def compute_z_ratio(transcripts: pd.DataFrame, num_planes: int = 7) -> float:
        """
        Returns ratio of transcript counts between highest and lowest planes

        Parameters
        ----------
        transcripts : pd.DataFrame
            Transcripts DataFrame

        Attributes Set
        --------------
        transcripts_per_z : np.ndarray
            Array of transcript counts per z-plane. Index corresponds to z-plane
        z_plane_ratio : float
            Ratio of transcript counts between highest and lowest z-planes

        Returns
        -------
        z_plane_ratio : float
            Ratio of transcript counts between highest and lowest z-planes

        Examples
        --------
        >>> z_ratio = ZPlane.get_transcripts_per_z(transcripts)
        >>> z_ratio
        0.436
        """
        transcripts_per_z = ZPlane.get_transcripts_per_z(transcripts, num_planes)
        z_plane_ratio = ZPlane.get_z_plane_ratio(transcripts_per_z)
        return z_plane_ratio
