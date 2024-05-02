from typing import Union
from pathlib import Path
import pandas as pd
import numpy as np

class Experiment:

    def __init__(self, transcripts: pd.DataFrame = None, transcripts_path: Union[str, Path] = None):
        """
        Initialize an Experiment instance from transcripts dataframe 
        """
        if transcripts_path is None and transcripts is None:
            raise Exception("Either transcripts_path or transcripts must be provided")
        
        if transcripts_path is not None and transcripts is None:
            # Read transcripts
            transcripts = self.read_transcripts(self.transcripts_path)

        # Adjust (x,y) locations
        self.transcripts = self.scale_transcripts_xy(transcripts)
        # Counts per gene (including blanks)
        self.counts_per_gene = self.transcripts.groupby('gene').size().to_dict()
        # Remove 'Blank' codewords
        self.filtered_transcripts = self.remove_blanks(self.transcripts)
        # self.n_genes = self.filtered_transcripts['gene'].nunique()
        self.genes = [gene for gene in self.filtered_transcripts['gene'].unique()]
        # Total transcript counts, including 'Blank' codewords
        self.total_transcript_counts = len(self.transcripts)
        # Total transcript counts, excluding 'Blank' codewords
        self.filtered_transcripts_count = len(self.filtered_transcripts)
        # Number of z-planes imaged
        self.num_planes = self.filtered_transcripts['global_z'].nunique()
        # DataFrame grouped by FOVs and storing FOV information
        self.fovs = self.get_fovs_dataframe(self.filtered_transcripts)


    @staticmethod
    def read_transcripts(transcripts_path: Union[str, Path]) -> pd.DataFrame:
        """
        Reads transcripts csv and returns DataFrame for given `detected_transcripts.csv`

        Parameters
        ----------
        path : str or Path
            Path to detected_transcripts.csv file

        Returns
        -------
        all_transcripts : pd.DataFrame
            Detected transcripts CSV

        Raises
        ------
        FileNotFoundError
            If file not found at given path
        """
        try:
            transcripts = pd.read_csv(transcripts_path)
            return transcripts
        except FileNotFoundError as e:
            raise FileNotFoundError(f'detected_transcripts.csv not found at {transcripts_path}'
                                    f'Error: {e}')
        
    @staticmethod
    def remove_blanks(transcripts: pd.DataFrame) -> pd.DataFrame:
        """
        Filters transcripts DataFrame to remove 'Blank' codewords

        Parameters
        ----------
        transcripts : pd.DataFrame
            Transcripts DataFrame

        Returns
        -------
        filtered_transcripts : pd.DataFrame
            Transcripts DataFrame excluding 'Blank' codewords
        """
        return transcripts[~transcripts['gene'].str.startswith('Blank')]

    @staticmethod
    def scale_transcripts_xy(transcripts: pd.DataFrame) -> pd.DataFrame:
        """
        Scales transcripts (x,y) locations based on min (x,y) values

        Parameters
        ----------
        transcripts : pd.DataFrame
            Transcripts DataFrame

        Returns
        -------
        transcripts : pd.DataFrame
            Rescaled transcripts DataFrame

        Notes
        -----
        This is especially necessary for generating accurate masks in experiments with multiple regions, where x,y
        transcript locations are on a global scale.
        """
        min_x = transcripts['global_x'].min()
        min_y = transcripts['global_y'].min()

        transcripts.loc[:, 'global_x'] -= min_x
        transcripts.loc[:, 'global_y'] -= min_y

        return transcripts

    @staticmethod
    def find_fovs(transcripts: pd.DataFrame) -> pd.DataFrame:
        """
        Group transcripts by FOV and stores per-FOV information, including coordinates and transcript counts

        Parameters
        ----------
        transcripts : pd.DataFrame
            Transcripts DataFrame

        Returns
        -------
        fovs : pd.DataFrame
            FOVs DataFrame
        """
        # Get min/max coords
        fovs = transcripts[['global_x', 'global_y', 'fov']].groupby('fov').min()
        fovs.rename(columns={'global_x': 'x_min', 'global_y': 'y_min'}, inplace=True)
        fovs[['x_max', 'y_max']] = transcripts[['global_x', 'global_y', 'fov']].groupby('fov').max()

        # Width/height
        fovs['width'] = fovs['x_max'] - fovs['x_min']
        fovs['height'] = fovs['y_max'] - fovs['y_min']

        # Centroids
        fovs['center_x'] = (fovs['x_max'] - fovs['x_min']) / 2 + fovs['x_min']
        fovs['center_y'] = (fovs['y_max'] - fovs['y_min']) / 2 + fovs['y_min']

        # Add transcript counts per FOV
        num_planes = transcripts['global_z'].nunique()
        fovs['transcript_count'] = transcripts.groupby('fov').size()

        # Add counts per z-plane per FOV
        for i in range(num_planes):
            fovs[f'z{i}_count'] = transcripts[transcripts['global_x'] == i].groupby('fov').size()

        # Add z-ratio per z plane
        fovs['z_ratio'] = fovs[f'z{num_planes - 1}_count'] / fovs['z0_count']

        return fovs
    

    @staticmethod
    def get_fov_neighbors(fovs: pd.DataFrame) -> pd.DataFrame:
        """
        Find neighbors for each FOV using grid coordinates

        Uses Euclidean distance between FOV centers to identify closest FOVs in cardinal direction

        Parameters
        ----------
        fovs : pd.DataFrame
            FOVs dataframe containing min/max/width/height/center coords

        Returns
        -------
        fovs : pd.DataFrame
            Modified to contain neighbor information
        """
        # Size of grid square
        max_width = np.max(fovs['width'])
        max_height = np.max(fovs['height'])
        grid_sq_size = max((max_width, max_height))

        centers_array = np.array(fovs[['center_x', 'center_y']])

        neighbors = [[] for i in range(len(fovs))]

        for i in range(len(fovs)):
            fov = fovs.index[i]
            # Find euclidean distance between grid coords of each FOV and all other FOVs
            fov_center = np.broadcast_to(centers_array[i], (len(centers_array), 2))
            euclidian_distances = np.argsort(np.linalg.norm(fov_center - centers_array, axis=1))

            # Neighbor above
            above_fovs = np.unique(np.where((centers_array[:, 1] > fovs.loc[fov, 'y_max']) &
                                            (abs(centers_array[:, 0] - centers_array[i, 0]) <= grid_sq_size / 2) &
                                            (abs(centers_array[:, 1] - centers_array[i, 1]) <= grid_sq_size * 1.5))[0])
            if len(above_fovs) > 0:
                neighbors[i].append(fovs.index[euclidian_distances[np.isin(euclidian_distances, above_fovs)][0]])

            # Neighbor below
            below_fovs = np.unique(np.where((centers_array[:, 1] > fovs.loc[fov, 'y_min']) &
                                            (abs(centers_array[:, 0] - centers_array[i, 0]) <= grid_sq_size / 2) &
                                            (abs(centers_array[:, 1] - centers_array[i, 1]) <= grid_sq_size * 1.5))[0])
            if len(below_fovs) > 0:
                neighbors[i].append(fovs.index[euclidian_distances[np.isin(euclidian_distances, below_fovs)][0]])

            # Neighbor right
            right_fovs = np.unique(np.where((centers_array[:, 0] > fovs.loc[fov, 'x_max']) &
                                            (abs(centers_array[:, 1] - centers_array[i, 1]) <= grid_sq_size / 2) &
                                            (abs(centers_array[:, 0] - centers_array[i, 0]) <= 400))[0])
            if len(right_fovs) > 0:
                neighbors[i].append(fovs.index[euclidian_distances[np.isin(euclidian_distances, right_fovs)][0]])

            # Neighbor left
            left_fovs = np.unique(np.where((centers_array[:, 0] < fovs.loc[fov, 'x_min']) &
                                           (abs(centers_array[:, 1] - centers_array[i, 1]) <= grid_sq_size / 2) &
                                           (abs(centers_array[:, 0] - centers_array[i, 0]) <= 400))[0])
            if len(left_fovs) > 0:
                neighbors[i].append(fovs.index[euclidian_distances[np.isin(euclidian_distances, left_fovs)][0]])

        fovs['neighbors'] = neighbors
        return fovs

    @staticmethod
    def get_fovs_dataframe(transcripts: pd.DataFrame):
        """
        Creates FOVs DataFrame including coordinates, transcript counts, z-ratio, neighbors

        Parameters
        ----------
        transcripts : pd.DataFrame
            Transcripts DataFrame

        Returns
        --------------
        fovs : pd.DataFrame
            DataFrame of FOV coordinates, transcript counts, neighbors
        """
        # Get coordinates, transcript counts, z-plane ratios
        fovs = Experiment.find_fovs(transcripts)
        # Get cardinal neighbors
        fovs = Experiment.get_fov_neighbors(fovs)

        # Counts per gene
        counts_per_gene = transcripts.groupby(['fov'])['gene'].value_counts().unstack(fill_value=0)
        fovs = fovs.merge(counts_per_gene, left_index=True, right_index=True, how='left')
        return fovs
