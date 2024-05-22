from typing import Union
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import merquaco.pixel_classification as pc
import merquaco.data_processing as data_processing
from merquaco.data_loss import FOVDropout, DropoutResult
import merquaco.figures as figures
import merquaco.z_plane_detection as zp
import merquaco.perfusion as perfusion
import merquaco.periodicity as periodicity

# dictionary keys for output metadata
metrics_dict_keys = ["filtered_transcripts_count", "transcript_density_um2", "transcript_density_um2_per_gene",
                     "on_tissue_filtered_transcript_count", "z_ratio", "transcripts_per_z", "periodicity",
                     "periodicity_list", "counts_per_gene", "n_dropped_fovs", "n_dropped_genes", "dropped_fovs_dict",
                     "dropped_genes", "damage_area", "transcripts_area", "detachment_area", "ventricle_area", 
                     "total_area", "damage_percent", "transcripts_percent", "detachment_percent", "ventricle_percent",
                     "transcripts_mask_pixel_path", "transcripts_mask_object_path", "dapi_mask_pixel_path",
                     "dapi_mask_object_path", "ventricle_mask_pixel_path", "ventricle_mask_object_path"]


class Experiment:

    def __init__(self,
                 transcripts_input: Union[pd.DataFrame, str, Path],
                 ilastik_config_input: Union[dict, str, Path] = None,
                 transcripts_image_path: Union[str, Path] = None,
                 transcripts_mask_path: Union[str, Path] = None,
                 dapi_high_res_image_path: Union[str, Path] = None,
                 dapi_image_path: Union[str, Path] = None,
                 dapi_mask_path: Union[str, Path] = None,
                 detachment_mask_path: Union[str, Path] = None,
                 ventricle_image_path: Union[str, Path] = None,
                 ventricle_mask_path: Union[str, Path] = None,
                 damage_mask_path: Union[str, Path] = None,
                 pixel_classification_path: Union[str, Path] = None,
                 codebook_input: Union[pd.DataFrame, str, Path] = None,
                 perfusion_path: Union[str, Path] = None,
                 output_dir: Union[str, Path] = None):
        """
        Initialize an Experiment instance from transcripts table dataframe

        Parameters
        ----------
        transcripts_input : pd.DataFrame, str, or Path
            DataFrame of or path to transcripts table
        ilastik_path_input : dict, str, or Path, optional
            Dictionary of or path to ilastik paths. Default is None.
        transcripts_image_path : str or Path, optional
            Path at which to save transcripts image. Default is None.
        transcripts_mask_path : str or Path, optional
            Path at which to save binary transcripts mask. Default is None.
        dapi_high_res_image_path : str or Path, optional
            Path to high resolution DAPI tiff image. Default is None.
        dapi_image_path : str or Path, optional
            Path at which to save low-res, modified DAPI image. Default is None.
        dapi_mask_path : str or Path, optional
            Path at which to save binary DAPI mask.
        detachment_mask_path : str or Path, optional
            Path at which to save detachment binary mask. Default is None.
        ventricle_image_path : str or Path, optional
            Path at which to save ventricle image. Default is None.
        ventricle_mask_path : str or Path, optional
            Path at which to save binary ventricle mask. Default is None.
        damage_mask_path : str or Path, optional
            Path at which to save binary damage mask. Default is None.
        pixel_classification_path : str or Path, optional
            Path at which to save final pixel classification results. Default is None.
        codebook_input : pd.DataFrame, str, or Path, optional
            DataFrame of or path to codebook table. Default is None.
        perfusion_path : str or Path, optional
            Path to perfusion log file. Default is None.
        output_dir : str or Path, optional
            Directory at which to save QC outputs

        Attributes Set
        --------------
        transcripts_image_path : str or Path
            Path at which to save transcripts image.
        transcripts_mask_path : str or Path
            Path at which to save binary transcripts mask.
        dapi_high_res_image_path : str or Path
            Path to high resolution DAPI tiff image.
        dapi_image_path : str or Path
            Path at which to save low-res, modified DAPI image.
        dapi_mask_path : str or Path
            Path at which to save binary DAPI mask.
        detachment_mask_path : str or Path
            Path at which to save detachment binary mask.
        ventricle_image_path : str or Path
            Path at which to save ventricle image.
        ventricle_mask_path : str or Path
            Path at which to save binary ventricle mask.
        damage_mask_path : str or Path
            Path at which to save binary damage mask.
        pixel_classification_path : str or Path
            Path at which to save final pixel classification results.
        codebook : pd.DataFrame
            Codebook table.
        perfusion_path : str or Path
            Path to perfusion log file.
        output_dir : str or Path
            Directory at which to save QC outputs.
        transcripts : pd.DataFrame
            Transcripts table
        counts_per_gene : dict
            Dictionary of transcript counts pergene
        filtered_transcripts : pd.DataFrame
            Transcripts table excluding blanks
        n_genes : int
            Number of genes imaged
        genes : list
            List of genes
        total_transcripts_count : int
            Total detected transcripts
        filtered_transcripts_count : int
            Total detected transcripts, excluding blanks
        num_planes : int
            Number of z-planes imaged
        fovs_df : pd.DataFrame
            DataFrame of FOV coordinates, transcript counts, neighbors
        """
        # Assign ilastik paths as attributes
        ilastik_config_dict = data_processing.process_input(ilastik_config_input)
        # Unpack dictionary as self attributes
        try:
            for key, val in ilastik_config_dict.items():
                setattr(self, key, val)
        except AttributeError:
            pass

        # Assign parameters as self attributes
        self.transcripts_image_path = transcripts_image_path
        self.transcripts_mask_path = transcripts_mask_path
        self.dapi_high_res_image_path = dapi_high_res_image_path
        self.dapi_image_path = dapi_image_path
        self.dapi_mask_path = dapi_mask_path
        self.detachment_mask_path = detachment_mask_path
        self.ventricle_image_path = ventricle_image_path
        self.ventricle_mask_path = ventricle_mask_path
        self.damage_mask_path = damage_mask_path
        self.pixel_classification_path = pixel_classification_path
        self.perfusion_path = perfusion_path
        self.output_dir = output_dir
        try:
            self.codebook = Experiment.read_codebook(codebook_input)
        except AttributeError:
            self.codebook = codebook_input

        # Transcripts dataframe
        print('Processing transcripts dataframe')
        transcripts = data_processing.process_input(transcripts_input)
        # Adjust (x,y) locations
        self.transcripts = self.scale_transcripts_xy(transcripts)
        # Counts per gene (including blanks)
        self.counts_per_gene = self.transcripts.groupby('gene').size().to_dict()
        # Remove 'Blank' codewords
        print('Filtering Blank codewords')
        self.filtered_transcripts = self.remove_blanks(self.transcripts)
        self.n_genes = self.filtered_transcripts['gene'].nunique()
        self.genes = [gene for gene in self.filtered_transcripts['gene'].unique()]
        # Total transcript counts, including 'Blank' codewords
        self.total_transcripts_counts = len(self.transcripts)
        # Total transcript counts, excluding 'Blank' codewords
        self.filtered_transcripts_count = len(self.filtered_transcripts)
        # Number of z-planes imaged
        self.num_planes = self.filtered_transcripts['global_z'].nunique()
        # DataFrame grouped by FOVs and storing FOV information
        print('Creating FOVs dataframe')
        self.fovs_df = self.get_fovs_dataframe(self.filtered_transcripts)

        # Create transcripts mask if parameters are provided
        try:
            if not data_processing.check_if_none(self.ilastik_program_path,
                                                 self.transcripts_mask_pixel_path,
                                                 self.transcripts_mask_object_path,
                                                 self.transcripts_image_path,
                                                 self.transcripts_mask_path):
    
                self.transcripts_mask = pc.generate_transcripts_mask(self.transcripts_image_path,
                                                                     self.ilastik_program_path,
                                                                     self.transcripts_mask_pixel_path,
                                                                     self.transcripts_mask_object_path,
                                                                     self.filtered_transcripts)
        except AttributeError:
            pass
      

    @staticmethod
    def read_transcripts(transcripts_path: Union[str, Path]) -> pd.DataFrame:
        """
        Reads and returns transcripts table dataframe

        Parameters
        ----------
        transcripts_path : str or Path
            Path to detected_transcripts.csv file

        Returns
        -------
        transcripts : pd.DataFrame
            Transcripts table

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
            Transcripts table

        Returns
        -------
        filtered_transcripts : pd.DataFrame
            Transcripts table excluding 'Blank' codewords
        """
        return transcripts[~transcripts['gene'].str.startswith('Blank')]

    @staticmethod
    def read_codebook(codebook_input: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """
        Reads codebook for use with data loss module

        Parameters
        ----------
        codebook_path : str or Path
            Path at which to read codebook csv
        Returns
        -------
        codebook : pd.DataFrame
            Codebook dataframe, excluding blanks
        """
        codebook = data_processing.process_input(codebook_input)
        codebook = codebook.drop(columns=['id', 'barcodeType'], errors='ignore').set_index('name')
        codebook = codebook[~codebook.index.str.startswith('Blank')]

        return codebook

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
        fovs['transcripts_count'] = transcripts.groupby('fov').size()

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

    @staticmethod
    def get_transcript_density(transcripts_image_input: Union[np.ndarray, str, Path],
                               transcripts_mask_input: Union[np.ndarray, str, Path]):
        """
        Calculates transcript density per on-tissue micron

        Parameters
        ----------
        transcripts_image_input : np.ndarray, str, or Path
            Array of or path to transcripts image.
        transcripts_mask_input : np.ndarray, str, or Path
            Array of or path to binary transcripts mask.

        Returns
        -------
        tuple
            on_tissue_filtered_transcript_count : float
                Number of on-tissue transcripts
            transcript_density_um2 : float
                Number of transcripts per on-tissue micron
        """
        transcripts_image = data_processing.process_input(transcripts_image_input)
        transcripts_mask = data_processing.process_input(transcripts_mask_input)

        on_tissue_filtered_transcript_count = np.sum(transcripts_image[transcripts_mask == 1])
        transcripts_mask_area = np.count_nonzero(transcripts_mask) * 100  # Mask has 10um pixels

        # When issue with Ilastik mask or experiment such that transcript counts are way low
        if transcripts_mask_area > 0:
            transcript_density_um2 = on_tissue_filtered_transcript_count / transcripts_mask_area
        else:
            transcript_density_um2 = np.nan

        return on_tissue_filtered_transcript_count, transcript_density_um2
    
    @staticmethod
    def write_qc_summary(qc_summary_path: Union[str, Path], qc_dict: dict):

        # Check if the file exists and load existing data
        if qc_summary_path.exists():
            with open(qc_summary_path, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = {}

        # Update only the fields that are "NA" in the existing data
        for key, value in qc_dict.items():
            if isinstance(existing_data.get(key), float) and np.isnan(existing_data.get(key)):
                existing_data[key] = value

        # Write the updated data back to the file
        with open(qc_summary_path, 'w') as file:
            json.dump(existing_data, file, indent=4)


    def run_dropout_pipeline(self):
        """
        Runs entire dropout pipeline, including false positive correction

        Attributes Set
        --------------
        fovs : pd.DataFrame
            FOVs dataframe including all dropout information
        """
        self.fovs_df = FOVDropout.find_on_tissue_fovs(self.filtered_transcripts, self.fovs_df,
                                                      self.transcripts_mask_path, self.transcripts_image_path,
                                                      self.ilastik_program_path,
                                                      self.transcripts_mask_pixel_path,
                                                      self.transcripts_mask_object_path)
        self.fovs_df = FOVDropout.detect_dropouts(self.filtered_transcripts, self.fovs_df)
        self.fovs_df = FOVDropout.compare_codebook_fov_genes(self.fovs_df, self.codebook)
        self.fovs_df = FOVDropout.detect_false_positives(self.fovs_df, self.codebook)

        if self.output_dir is not None:
            FOVDropout.save_fov_tsv(self.fovs_df, self.output_dir)

    def run_full_pixel_classification(self):
        """
        Runs entire pixel classification workflow:
            - generates binary masks for transcripts, DAPI, gel lifting, ventricles, and damage
            - resizes and aligns masks
            - calculates pixel percentages over "ideal" tissue area

        Attributes Set
        --------------
        pixel_areas : np.ndarray
            Array of pixel classification area designations in microns: [damage, tissue, gel lifting, ventricles]
        pixel_percentages : np.ndarray
            Array of pixel classification area designations as percentage of "ideal" tissue area
            [damage, tissue, gel lifting, ventricles]
        ideal_tissue_area : float
            Sum of all non-off-tissue pixels
        """
        # Set attributes as None in case there are no ventricle genes
        self.ventricle_mask = None
        self.damage_mask = None

        if self.transcripts_mask is None:
            print("Generating transcript mask...")
            self.transcripts_mask = pc.generate_transcripts_mask(self.transcripts_image_path,
                                                                 self.ilastik_program_path,
                                                                 self.transcripts_mask_pixel_path,
                                                                 self.transcripts_mask_object_path,
                                                                 self.transcripts_mask_path,
                                                                 self.filtered_transcripts)

        print("Generating DAPI mask...")
        self.dapi_mask = pc.generate_dapi_mask(self.dapi_image_path,
                                               self.ilastik_program_path,
                                               self.dapi_mask_pixel_path,
                                               self.dapi_mask_object_path,
                                               self.dapi_high_res_image_path)

        print("Generating lifting mask...")
        self.detachment_mask = pc.generate_detachment_mask(self.transcripts_mask_path,
                                                           self.dapi_mask_path,
                                                           self.detachment_mask_path)

        if any(np.isin(self.genes, self.ventricle_genes_list)):  # If ventricle genes exist
            print("Generating ventricle mask...")
            self.ventricle_mask = pc.generate_ventricle_mask(self.ventricle_image_path,
                                                             self.dapi_mask_path,
                                                             self.transcripts_mask_path,
                                                             self.ilastik_program_path,
                                                             self.ventricle_mask_pixel_path,
                                                             self.ventricle_mask_object_path,
                                                             self.filtered_transcripts,
                                                             self.ventricle_genes_list)

            print("Generating damage mask...")
            self.damage_mask = pc.generate_damage_mask(self.damage_mask_path,
                                                       self.dapi_image_path,
                                                       self.dapi_mask_path,
                                                       self.transcripts_mask_path,
                                                       self.ventricle_mask_path)

            # Resize all masks by transcripts mask
            self.transcripts_mask, self.dapi_mask, self.detachment_mask, \
                self.ventricle_mask, self.damage_mask = pc.resize_all_masks(self.transcripts_mask,
                                                                            self.dapi_mask,
                                                                            self.detachment_mask,
                                                                            self.ventricle_mask,
                                                                            self.damage_mask)

        # Classify each pixel
        print("Classifying pixels...")
        self.pixel_classification = pc.classify_pixels(self.transcripts_mask,
                                                       self.detachment_mask,
                                                       self.ventricle_mask,
                                                       self.damage_mask,
                                                       self.pixel_classification_path)

        # Get pixel areas in microns and as percentage of "ideal" tissue area
        self.damage_area, self.transcripts_area, self.detachment_area, \
            self.ventricle_area, self.total_area = pc.calculate_class_areas(self.pixel_classification)
        self.damage_percent, self.transcripts_percent, self.detachment_percent, \
            self.ventricle_percent = pc.calculate_class_percentages(self.damage_area,
                                                                    self.transcripts_area,
                                                                    self.detachment_area,
                                                                    self.ventricle_area,
                                                                    self.total_area)

        pixel_stats_dict = {'damage_area': self.damage_area,
                            'transcripts_area': self.transcripts_area,
                            'detachment_area': self.detachment_area,
                            'ventricle_area': self.ventricle_area,
                            'damage_percent': self.damage_percent,
                            'transcripts_percent': self.transcripts_percent,
                            'detachment_percent': self.detachment_percent,
                            'ventricle_percent': self.ventricle_percent,
                            'total_area': self.total_area}

        # Convert and write JSON object to file
        if self.output_dir is not None:
            with open(Path(self.output_dir, "pixel_stats.json"), "w") as outfile:
                json.dump(pixel_stats_dict, outfile, indent=4)

    def run_all_qc(self,
                   run_pixel_classification: bool = True,
                   run_dropout: bool = True,
                   run_perfusion: bool = False,
                   plot_figures: bool = True,
                   save_metrics: bool = True):
        """
        Runs all standard QC functions and prints results

        Parameters
        ----------
        run_pixel_classification : bool, optional
            Whether to run pixel classification workflow. Default is True.
        run_dropout : bool, optional
            Whether to run FOV dropout workflow. Default is True.
        run_perfusion : bool, optional
            Whether to run perfusion workflow. Default is False.
        plot_figures : bool, optional
            Whether to plot figures. Default is True.

        Attributes Set
        --------------

        """
        # 1. Run  pixel classification workflow
        if run_pixel_classification:
            self.run_full_pixel_classification()

            if plot_figures:
                figures.plot_full_pixel_fig(self.pixel_classification,
                                            self.dapi_mask,
                                            self.transcripts_mask,
                                            self.detachment_mask,
                                            self.transcripts_percent,
                                            self.detachment_percent,
                                            self.damage_mask,
                                            self.ventricle_mask,
                                            self.damage_percent,
                                            self.ventricle_percent,
                                            Path(self.output_dir, "pixel_classification_plot.png"))
                plt.show()

        # 2. Dropout detection
        self.n_dropped_fovs = np.nan  # Initialize empty for plotting later if dropout is not run
        if run_dropout:
            print('Beginning FOV Dropout Detection')
            self.run_dropout_pipeline()
            dropout = DropoutResult(self.fovs_df)
            # Get raw number of dropped FOVs, genes
            self.n_dropped_fovs = int(dropout.get_dropped_fov_counts())
            self.n_dropped_genes = int(dropout.get_dropped_gene_counts())
            self.dropped_genes = ", ".join(dropout.get_dropped_genes())
            # Get dictionary of dropped genes per FOV
            self.dropped_fovs_dict = dropout.get_dropped_gene_counts(dic=True)
            if plot_figures:
                # Draw and save dropout
                if self.n_dropped_fovs > 0:
                    dropout.draw_genes_dropped_per_fov(out_path=Path(self.output_dir, 'fov_dropout.png'))

        # 3. On-tissue metrics
        self.on_tissue_filtered_transcript_count, \
            self.transcript_density_um2 = Experiment.get_transcript_density(self.transcripts_image_path,
                                                                             self.transcripts_mask)
        self.transcript_density_um2_per_gene = self.transcript_density_um2 / self.n_genes

        # 4. Periodicity
        print('Calculating periodicity')
        self.periodicity_list = periodicity.get_periodicity_list(self.filtered_transcripts, num_planes=self.num_planes)
        self.periodicity = np.round(np.nanmin(self.periodicity_list), 3)
        if plot_figures:
            figures.plot_periodicity_hist(self.transcripts, out_file=Path(self.output_dir, "periodicity_hist.png"))

        # 5. z-plane transcript ratio
        print('Computing z-plane metrics')
        self.z_ratio = zp.compute_z_ratio(self.filtered_transcripts, self.num_planes)
        self.transcripts_per_z = zp.get_transcripts_per_z(self.filtered_transcripts, self.num_planes).tolist()
        if plot_figures:
            figures.plot_every_z_plane(self.transcripts, out_file=Path(self.output_dir, "transcripts_per_z.png"))
            figures.plot_transcripts_per_z(self.transcripts_per_z,
                                           out_file=Path(self.output_dir, "transcript_counts_per_z.png"))

        # 7. Perfusion
        if run_perfusion and self.perfusion_path is not None:
            perfusion_data = perfusion.analyze_flow(self.perfusion_path)
            if plot_figures:
                figures.plot_perfusion_figure(perfusion_data,
                                              out_file=Path(self.output_dir, "perfusion.png"))

        # 8. Save metrics
        metrics_dict = {}
        for key in metrics_dict_keys:
            metrics_dict[key] = getattr(self, key, np.nan)

        if save_metrics:
            Experiment.write_qc_summary(Path(self.output_dir, "qc_summary.json"), metrics_dict)
