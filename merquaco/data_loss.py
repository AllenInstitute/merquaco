import pandas as pd
import numpy as np
from pathlib import Path
import os
import tifffile as tiff
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as color_normalize
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
from typing import Union
import merquaco.pixel_classification as pc


class FOVDropout:
    """
    Class with methods for detecting FOV dropout

    Methods
    -------
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def find_on_tissue_fovs(transcripts: pd.DataFrame, fovs: pd.DataFrame,
                            transcripts_mask_path: Union[str, Path] = None,
                            transcripts_image_path: Union[str, Path] = None,
                            ilastik_program_path: Union[str, Path] = None,
                            pixel_model_path: Union[str, Path] = None,
                            object_model_path: Union[str, Path] = None,
                            force_mask: bool = False) -> pd.DataFrame:
        """
        Use ilastik transcripts mask to detemine on- and off-tissue FOVs
        FOV is considered on-tissue if at least 50% of its area is on-tissue

        Parameters
        ----------
        transcripts : pd.DataFrame
            Transcripts table dataframe
        fovs : pd.DataFrame
            FOVs dataframe
        transcripts_mask_path : np.ndarray or str or Path
            String or Path of path to transcripts mask. Optional,
            required if force_mask is False
        transcripts_image_path : str or Path
            Path to transcripts image to save image for transcripts mask
            generation. Required if force_mask is True
        ilastik_program_path : str or Path
            Path to ilastik program for transcripts mask generation.
            Required if force_mask is True
        pixel_model_path : str or Path
            Path to pixel classification model for transcripts mask
            generation. Required if force_mask is True
        object_model_path : str or Path
            Path to object classification model for transcripts mask
            generation. Required if force_mask is True
        force_mask : bool, optional
            Whether to force the creation of a new transcripts mask

        Returns
        -------
        fovs : pd.DataFrame
            Updated FOVs dataframe with on-tissue column

        Raises
        ------
        ValueError
            If image path is not provided
        RuntimeError
            If no on-tissue FOVs are found

        Notes
        -----
        `pixel_classification.create_transcripts_image()` returns the original dimensions of the transcripts image, but
        saves a transposed version of that image to match the remaining (i.e., DAPI, ventricles) masks. For this reason,
        the mask is transposed after being read in, to match the original transcripts dimensions.
        """

        _, mask_x_bins, mask_y_bins = pc.create_transcripts_image(transcripts)

        if not force_mask and os.path.exists(transcripts_mask_path):
            transcripts_mask = tiff.imread(transcripts_mask_path)
        else:
            if transcripts_image_path is None:
                raise ValueError("transcripts_image_path must be provided")
            if ilastik_program_path is None:
                raise ValueError("ilastik_program_path must be provided")
            if pixel_model_path is None:
                raise ValueError("pixel_model_path must be provided")
            if object_model_path is None:
                raise ValueError("object_model_path must be provided")

            # Create and save mask
            _, mask_x_bins, mask_y_bins = pc.create_transcript_image(transcripts, transcripts_image_path)
            transcripts_mask = pc.generate_transcript_mask(transcripts_image_path, ilastik_program_path,
                                                           pixel_model_path, object_model_path)

        transcripts_mask = transcripts_mask.T  # Transpose to match original transcripts dimensions

        on_tissue = []
        for fov in fovs.index:
            # Find mask coordinates of FOV
            mask_min_x = np.digitize(fovs.loc[fov, 'x_min'], mask_x_bins) - 1
            mask_max_x = np.digitize(fovs.loc[fov, 'x_max'], mask_x_bins)
            mask_min_y = np.digitize(fovs.loc[fov, 'y_min'], mask_y_bins) - 1
            mask_max_y = np.digitize(fovs.loc[fov, 'y_max'], mask_y_bins)

            # If 50% of FOV is on tissue, consider it on-tissue
            if np.sum(transcripts_mask[mask_min_x:mask_max_x, mask_min_y:mask_max_y]) >= \
                0.5 * (mask_max_x - mask_min_x) * (mask_max_y - mask_min_y) and \
                    fovs.loc[fov, 'height'] > 0 and fovs.loc[fov, 'width'] > 0:
                on_tissue.append(True)
            else:
                on_tissue.append(False)

        fovs['on_tissue'] = on_tissue

        if not np.any(on_tissue):
            raise RuntimeError('No on-tissue FOVs found (likely failure in Ilastik tissue mask generation or '
                               'failure in experiment transcript detection)')
        return fovs

    @staticmethod
    def detect_dropouts(transcripts: pd.DataFrame, fovs: pd.DataFrame, thresh: float = 0.15) -> pd.DataFrame:
        """
        Compares cardinal neighbors for each FOV to detect dropout

        FOVs are considered dropped if:
            - below delta threshold for number of transcripts for all 4 neighbors
            - OR below threshold for 3 neighbors and last neighbor is also a dropped FOV

        Parameters
        ----------
        transcripts : pd.DataFrame
            Transcripts table
        fovs : pd.DataFrame
            FOVs table dataframe
        threshold : float, optional
            Transcript count delta threshold for dropout detection

        Returns
        -------
        fovs : pd.DataFrame
            Adds columns for each gene/FOV pair for deltas, dropout status, and if neighbors average >100 transcripts
        """
        gene_counts = transcripts.groupby('gene').size()
        gene_counts = gene_counts.sort_values(ascending=False)

        # Find dropout for each gene
        for gene in gene_counts.index:
            # Count number of transcripts for each FOV for all trancsripts of the gene of interest
            gene_df = fovs.loc[:, gene].to_frame()  # FOVs dataframe already contains transcript counts per gene per FOV

            # First pass on 3 sides, so can detect dropout in 2 adjacent FOVs
            fov_deltas = []
            fov_dropout = []
            for fov in fovs.index:
                if fovs.loc[fov, 'on_tissue']:  # Only detect on-tissue dropout
                    neighbors = fovs.loc[fov, 'neighbors']

                    # Calculate ratio between number of transcripts in FOV and neighbors
                    # Take max with 1 to not divide by 0 on either side
                    # Zero on left will always be below delta and zero on right will divide by zero error
                    deltas = np.array([max(gene_df.loc[fov, gene], 1) /
                                       max(gene_df.loc[neighbor, gene], 1) for neighbor in neighbors])
                    fov_deltas.append(deltas)
                    # Only dropout if less than threshold on at least 3 sides
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        fov_dropout.append(True if np.count_nonzero(deltas < thresh) >= 3 else False)

                else:
                    fov_deltas.append(np.array([]))
                    fov_dropout.append(False)

            gene_df[f'deltas_{gene}'] = fov_deltas
            gene_df[f'dropout_{gene}'] = fov_dropout

            transcript_threshold = []  # Want to store if FOV was even considered for dropout (>100 avg among neighbors)

            # Second pass of 4 sides minus those which are adjacent to other dropout
            for fov in fovs.index:
                if fovs.loc[fov, 'on_tissue']:
                    neighbors = fovs.loc[fov, 'neighbors']
                    deltas = gene_df.loc[fov, f'deltas_{gene}']

                    # Calculate average transcripts of non-dropped neighbors
                    neighbor_avg_transcripts = []
                    for neighbor in neighbors:
                        if not gene_df.loc[neighbor, f'dropout_{gene}']:
                            neighbor_avg_transcripts.append(gene_df.loc[neighbor, gene])
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        above_transcript_threshold = np.mean(neighbor_avg_transcripts) >= 100
                        transcript_threshold.append(above_transcript_threshold)

                    if not above_transcript_threshold:  # If not >= 100 average neighboring transcripts, not considered
                        gene_df.loc[fov, f'dropout_{gene}'] = False
                    elif np.count_nonzero(deltas < thresh) == 4:  # If below delta threshold on 4 sides, dropped
                        gene_df.loc[fov, f'dropout_{gene}'] = True
                    # If less than threshold on 3 sides and one neighbor is dropped
                    elif np.count_nonzero(deltas < thresh) == 3 and np.any(gene_df.loc[neighbors,
                                                                                       f'dropout_{gene}']):
                        gene_df.loc[fov, f'dropout_{gene}'] = True
                    else:
                        gene_df.loc[fov, f'dropout_{gene}'] = False

                else:
                    fov_deltas.append(np.array([]))
                    fov_dropout.append(False)
                    transcript_threshold.append(False)

            gene_df[f'transcript_threshold_{gene}'] = transcript_threshold
            fovs = fovs.merge(gene_df[[f'transcript_threshold_{gene}', f'deltas_{gene}', f'dropout_{gene}']],
                              left_index=True, right_index=True, how='left', suffixes=('', '_remove'))
            # Remove duplicate columns
            fovs.drop([i for i in fovs.columns if 'remove' in i], axis=1, inplace=True)

        # Look over all FOVs, if no genes for an FOV had neighbors above transcript threshold, the FOV was never
        # considered for dropout, and can cause divide by zero errors downstream if not noted.
        # Roll them in with 'off-tissue' FOVs as a way of noting
        for fov in fovs.index:
            if fovs.loc[fov, 'on_tissue'] and np.count_nonzero(fovs.filter(regex='transcript_threshold').loc[fov]) == 0:
                fovs.loc[fov, 'on_tissue'] = False

        # Raise error if no on-tissue FOVs are found
        if not np.any(fovs['on_tissue']):
            raise RuntimeError('No on-tissue FOVs found (likely error in ilastik tissue mask generation'
                               ' or failure in experiment transcript detection)')

        return fovs

    @staticmethod
    def compare_codebook_fov_genes(fovs: pd.DataFrame, codebook: pd.DataFrame) -> pd.DataFrame:
        """
        Compares genes in FOVs and codebook dataframes and removes mismatched genes from consideration.

        Genes missing from codebook are removed from FOVs dataframe, since they cannot be false positive corrected

        Parameters
        ----------
        fovs : pd.DataFrame
            FOVs table
        codebook : pd.DataFrame
            Codebook table

        Returns
        -------
        fovs : pd.DataFrame
            Updated FOVs table including only genes in the codebook

        Raises
        ------
        UserWarning
            If there are genes in the FOVs table that are not present in the codebook, or vice versa.
        """
        # Warn user of potential problem if codebook genes are not the same as the transcripts.csv genes
        # Remove genes from FOVs dataframe
        missing_genes_bool = False
        missing_transcripts_genes = []
        missing_codebook_genes = []
        transcripts_genes = list(fovs.filter(regex='dropout').columns.str.replace('dropout_', ''))
        codebook_genes = list(codebook.index)

        for transcripts_gene in transcripts_genes:
            if transcripts_gene not in codebook_genes:
                missing_transcripts_genes.append(transcripts_gene)
                missing_genes_bool = True
                fovs = fovs.drop(columns=list(fovs.filter(regex=transcripts_gene)))  # Drop missing gene from FOVs
        for codebook_gene in codebook_genes:
            if codebook_gene not in transcripts_genes:
                missing_codebook_genes.append(codebook_gene)
                missing_genes_bool = True

        if missing_genes_bool:
            warnings.warn(f'WARNING: The codebook and transcripts.csv file contain differing genes.\n'
                          f'The transcripts.csv file is missing {",".join(missing_codebook_genes)}.\n'
                          f'The codebook is missing {",".join(missing_transcripts_genes)}.\n'
                          f'Genes missing from the codebook will be removed from dropout consideration as they'
                          f'cannot be properly false positive corrected.')

        return fovs

    @staticmethod
    def detect_false_positives(fovs: pd.DataFrame, codebook: pd.DataFrame) -> pd.DataFrame:
        """
        Detects false positive FOV dropouts by evaluating codebook rounds in which the genes dropped for a target FOV

        Parameters
        ----------
        fovs : pd.DataFrame
            FOVs table
        codebook : pd.DataFrame
            Experiment codebook table

        Returns
        ------
        fovs : pd.DataFrame
            Updated FOVs dataframe with false positive information
        """
        # Create dataframe to store information on imaging rounds (3 codebook bits to a round)
        bits = codebook.shape[1]
        round_df = codebook.T.groupby((np.arange(bits) // 3) + 1).sum()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=pd.errors.PerformanceWarning)
            fovs['false_positive'] = ['' for i in range(len(fovs))]

        # Loop over all dropped FOVs
        for fov in fovs[np.any(fovs.filter(regex='dropout'), axis=1)].index:
            dropped_genes = list(fovs.filter(regex='dropout').columns[np.where(fovs.filter(regex='dropout').loc[fov])].str.replace('dropout_', ''))
            # Find distribution of imaging rounds for all genes that experienced dropout in the FOV
            round_freqs = np.array(round_df.loc[dropped_genes].astype(bool).sum(axis=0) / len(dropped_genes))

            # If one of the rounds is present in 100% of the genes, know it is truly dropped
            # Must have more than `gene_thresh` genes to be considered, since can randomly choose N genes such that
            # they all share the same imaging round. Choose such that <10% probability of all sharing same round.
            if codebook.shape[0] > 300:
                gene_thresh = 5
            elif codebook.shape[0] < 300:
                gene_thresh = 7
            elif codebook.shape[0] == 300:
                gene_thresh = 6

            if np.max(round_freqs) == 1 and len(dropped_genes) >= gene_thresh:
                continue

            # If two high-scoring rounds, need to determine if random chance or due to FOV being dropped in 2 rounds
            # Could consider 3 high-scoring rounds, but should be very unlikely and brings distribution near uniform

            # Threshold determined by chance of one high-scoring round not having a bit in the other high-scoring round
            # Know there are 4 positive bits for each gene code, which is why there are 3 divisions
            round_freq_threshold = 0.5 + 0.5 * (1 - (((bits - 4) / (bits - 1)) *
                                                     ((bits - 5) / (bits - 2)) *
                                                     ((bits - 6) / (bits - 3))))

            # If both are over threshold and have >= 2*gene_thresh (double required for single), truly dropped
            if np.all(np.sort(round_freqs)[-2:] > round_freq_threshold) and len(dropped_genes) >= (2 * gene_thresh):
                continue
            # If last round has few bits and last round has a freq > .5, have to calculate differently
            # Still require 2 * gene_thresh genes
            elif bits % 3 and round_freqs[-1] > .5 and len(dropped_genes) >= (2 * gene_thresh):
                final_bits = bits % 3
                final_round_threshold = 0.5 + 0.5 * (1 - (((bits - final_bits - 1) / (bits - 1)) *
                                                          ((bits - final_bits - 2) / (bits - 2)) *
                                                          ((bits - final_bits - 3) / (bits - 3))))

                if np.max(round_freqs) > round_freq_threshold and np.sort(round_freqs)[-2] > final_round_threshold:
                    continue

            # If none of the above situations are true, consider the FOV to be a false positive and update status
            fovs.loc[fov, [f'dropout_{gene}' for gene in dropped_genes]] = False
            fovs.loc[fov, 'false_positive'] = ';'.join(dropped_genes)

        return fovs

    @staticmethod
    def save_fov_pkl(fovs: pd.DataFrame, output_dir: Union[str, Path]) -> None:
        """
        Save FOVs dataframe as .pkl file

        Parameters
        ----------
        fovs : pd.DataFrame
            FOVs dataframe
        output_dir : str or Path
            Path at which to save pkl file

        Returns
        -------
        None
        """
        try:
            os.makedirs(output_dir)
        except FileExistsError:
            pass

        fovs.to_pickle(Path(output_dir, 'fovs.pkl'))

    @staticmethod
    def save_fov_tsv(fovs: pd.DataFrame, output_dir: Union[str, Path]) -> None:
        """
        Save FOVs dataframe as .txt.gz file

        Parameters
        ----------
        fovs : pd.DataFrame
            FOVs dataframe
        output_dir : str or Path
            Path at which to save .txt.gz file

        Returns
        -------
        None
        """
        try:
            os.makedirs(output_dir)
        except FileExistsError:
            pass

        fovs.to_csv(Path(output_dir, 'fovs.txt.gz'), sep='\t')


class DropoutResult:
    def __init__(self, fovs: pd.DataFrame, transcripts: pd.DataFrame = None):
        """
        Initialize dropout result, using FOVs dataframe already run through dropout pipeline

        Parameters
        ----------
        fovs : pd.DataFrame
            FOVs table with FOV dropout information
        transcripts : pd.DataFrame, optional
            Transcripts table dataframe. Default is None.
        """
        # Read in FOVs table to dataframe
        if isinstance(fovs, pd.DataFrame):
            self.fovs = fovs
        elif isinstance(fovs, str):
            if fovs.endswith('.pkl'):
                self.fovs = pd.read_pickle(fovs)
            else:
                self.fovs = pd.read_table(fovs, sep="\t", header=0, index_col='fov')
        elif isinstance(fovs, Path):
            if fovs.suffix == '.pkl':
                self.fovs = pd.read_pickle(fovs)
            else:
                self.fovs = pd.read_table(str(fovs), sep="\t", header=0, index_col='fov')
        else:
            raise ValueError("Invalid type for FOV table")

        self.transcripts = transcripts

        # Get list of genes
        genes = []
        for col in self.fovs.columns:
            if col.startswith('delta'):
                genes.append(col.split('_')[-1])
        self.genes = genes

    def get_dropout_count(self):
        """
        Get total number of dropped FOVs
        """
        return np.count_nonzero(self.fovs.filter(regex='dropout'))

    def get_dropped_genes(self, fov: str = '', dic: bool = False):
        """
        Get a list of dropped genes.
        If FOV is specified, gets a list of dropped genes for the specified FOV.
        If `dic=True`, creates a dictionary of FOVs and dropped genes

        Parameters
        ----------
        fov : str, optional
            FOV to subset dropped gene list by. Default is ''.
        dic : bool, optional
            Whether to return dictionary of dropped FOVs and genes. Default is False.

        Returns
        -------
        list
            If `fov` is specified. List of dropped genes in that FOV.
        dict
            If `dic` is `True`. Dictionary of dropped genes per FOV
        list
            If all arguments are default. List of dropped genes.
        """
        if fov != '':
            return list(self.fovs.filter(regex='dropout').columns[np.where(self.fovs.filter(regex='dropout').loc[fov])].str.replace('dropout_', ''))
        elif dic:
            genes = np.array(self.fovs.filter(regex='dropout').columns.str.replace('dropout_', ''))
            arr = np.zeros((len(self.get_considered_fovs()), len(self.genes)), dtype=object)
            fov_idx, gene_idx = np.where(self.fovs.loc[self.get_considered_fovs()].filter(regex='dropout_'))
            arr[fov_idx, gene_idx] = genes[gene_idx]
            return {fov: list(arr[i][arr[i] != 0]) for i, fov in enumerate(self.get_considered_fovs())}
        else:
            return list(self.fovs.filter(regex='dropout').columns[np.any(self.fovs.filter(regex='dropout'),
                                                                         axis=0)].str.replace('dropout_', ''))

    def get_dropped_gene_counts(self, fov: str = '', dic: bool = False):
        """
        Gets number of dropped genes.
        If FOV is specified, gets number of dropped genes for specified FOV.
        If `dic=True`, creates dictionary of FOVs and dropped gene counts

        Parameters
        ----------
        fov : str, optional
            FOV for which to get gene counts for. Default is ''.
        dic : bool, optional
            Whether to create dictionary of FOVs and their dropped gene counts

        Returns
        -------
        int
            If `fov` is specified. Number of dropped genes for that FOV
        dict
            If `dic` is `True`. Dictionary of dropped gene counts per FOV
        int
            If all arguments are default. Number of dropped genes.
        """
        if fov != '':
            return len(self.fovs.filter(regex='dropout').columns[np.where(self.fovs.filter(regex='dropout').loc[fov])].str.replace('dropout_', ''))
        elif dic:
            genes = np.array(self.fovs.filter(regex='dropout').columns.str.replace('dropout_', ''))
            arr = np.zeros((len(self.get_considered_fovs()), len(self.genes)), dtype=object)
            fov_idx, gene_idx = np.where(self.fovs.loc[self.get_considered_fovs()].filter(regex='dropout_'))
            arr[fov_idx, gene_idx] = genes[gene_idx]
            return {fov: len(arr[i][arr[i] != 0]) for i, fov in enumerate(self.get_considered_fovs())}
        else:
            return len(self.fovs.filter(regex='dropout').columns[np.any(self.fovs.filter(regex='dropout'),
                                                                        axis=0)].str.replace('dropout_', ''))

    def get_dropped_fovs(self, gene: str = '', dic: bool = False):
        """
        Get a list of dropped FOV names.
        If `gene` is specified, returns list of dropped FOVs for that gene.
        If `dic=True`, returns dictionary of genes and dropped FOVs.

        Parameters
        ----------
        gene : str, optional
            Gene for which to get list of dropped FOVs. Default is ''.
        dic : bool, optional
            Whether to return dictionary of genes and their dropped FOVs. Default is False.

        Returns
        -------
        list
            If `gene` is specified. List of dropped FOVs for that gene
        dict
            If `dic` is True. Dictionary of dropped FOVs per gene.
        list
            If all arguments are default. List of dropped FOVs.
        """
        if gene != '':
            return list(self.fovs[self.fovs[f'dropout_{gene}']].index)
        elif dic:
            fovs = np.array(self.fovs.index)
            genes = np.array(self.fovs.filter(regex='dropout').columns.str.replace('dropout_', ''))
            arr = -np.ones((len(genes), len(fovs)), dtype=np.int64)
            fov_idx, gene_idx = np.where(self.fovs.filter(regex='dropout_'))
            arr[gene_idx, fov_idx] = fovs[fov_idx]
            return {gene: list(arr[i][arr[i] != -1]) for i, gene in enumerate(genes)}
        else:
            return list(self.fovs[self.fovs.filter(regex='dropout').sum(axis=1) > 0].index)

    def get_dropped_fov_counts(self, gene: str = '', dic: bool = False):
        """
        Gets number of unique dropped FOVs.
        If `gene` is specified, returns number of dropped FOVs for specified gene.
        If `dic=True`, returns dictionary of genes and number of dropped FOVs.

        Parameters
        ----------
        gene : str, optional
            Gene for which to get number of dropped FOVs. Default is ''.
        dic : bool, optional
            Whether to return dictionary of genes and number of dropped FOVs.

        Returns
        -------
        int
            If `gene` is specified. Number of dropped FOVs for that gene.
        dict
            If `dic` is True. Dictionary of number of dropped FOVs per gene.
        int
            If all arguments are default. Number of unique dropped FOVs.
        """
        if gene != '':
            return len(self.fovs[self.fovs[f'dropout_{gene}']].index)
        elif dic:
            fovs = np.array(self.fovs.index)
            genes = np.array(self.fovs.filter(regex='dropout').columns.str.replace('dropout_', ''))
            arr = -np.ones((len(genes), len(fovs)), dtype=np.int64)
            fov_idx, gene_idx = np.where(self.fovs.filter(regex='dropout_'))
            arr[gene_idx, fov_idx] = fovs[fov_idx]
            return {gene: len(arr[i][arr[i] != -1]) for i, gene in enumerate(genes)}
        else:
            return len(self.fovs[self.fovs.filter(regex='dropout').sum(axis=1) > 0].index)

    def get_considered_genes(self, fov: str = '', dic: bool = False):
        """
        Gets a list of all genes with at least 1 FOV considered for dropout.
        FOV is considered for dropout only if its 4 carinal neighbors average at least 100 transcripts.
        If `fov` is specified, returns list for the specified FOV.
        If `dic=True`, returns a dictionary of considered genes for each considered FOV.

        Parameters
        ----------
        fov : str, optional
            FOV for which to get list of considered genes. Default is ''.
        dic : bool, optional
            Whether to return dictionary of considered genes for each considered FOV. Default is False.

        Returns
        -------
        list
            If `fov` is specified. List of considered genes for that FOV.
        dict
            If `dic` is True. Dictionary of considered genes per considered FOV
        list
            If all arguments are default. List of genes with at least one considered FOV.
        """
        if fov != '':
            return list(self.fovs.filter(regex='transcript_threshold').columns[np.where(self.fovs.filter(regex='transcript_threshold').loc[fov])[0]].str.replace('transcript_threshold_', ''))
        elif dic:
            genes = np.array(self.fovs.filter(regex='transcript_threshold').columns.str.replace('transcript_threshold_', ''))
            arr = np.zeros((len(self.get_considered_fovs()), len(self.genes)), dtype=object)
            fov_idx, gene_idx = np.where(self.fovs.loc[self.get_considered_fovs()].filter(regex='transcript_threshold'))
            arr[fov_idx, gene_idx] = genes[gene_idx]
            return {fov: list(arr[i][arr[i] != 0]) for i, fov in enumerate(self.get_considered_fovs())}
        else:
            return list(self.fovs.filter(regex='transcript_threshold').columns[np.sum(self.fovs.filter(regex='transcript_threshold'), axis=0) >= 1].str.replace('transcript_threshold_', ''))

    def get_considered_gene_counts(self, fov: str = '', dic: bool = False):
        """
        Gets number of genes with at least 1 FOV considered for dropout.
        FOV is considered for dropout only if its 4 carinal neighbors average at least 100 transcripts.
        If `fov` is specified, returns number of considered genes for the specified FOV.
        If `dic=True`, returns a dictionary of considered gene counts for each considered FOV.
        """
        if fov != '':
            return len(self.fovs.filter(regex='transcript_threshold').columns[np.where(self.fovs.filter(regex='transcript_threshold').loc[fov])[0]].str.replace('transcript_threshold_', ''))
        elif dic:
            genes = np.array(self.fovs.filter(regex='transcript_threshold').columns.str.replace('transcript_threshold_', ''))
            arr = np.zeros((len(self.get_considered_fovs()), len(self.genes)), dtype=object)
            fov_idx, gene_idx = np.where(self.fovs.loc[self.get_considered_fovs()].filter(regex='transcript_threshold'))
            arr[fov_idx, gene_idx] = genes[gene_idx]
            return {fov: len(arr[i][arr[i] != 0]) for i, fov in enumerate(self.get_considered_fovs())}
        else:
            return len(self.fovs.filter(regex='transcript_threshold').columns[np.sum(self.fovs.filter(regex='transcript_threshold'), axis=0) >= 1].str.replace('transcript_threshold_', ''))

    def get_considered_fovs(self):
        """
        Gets a list of all on-tissue FOVs

        Returns
        -------
        list
            List of all FOVs with >50% area on tissue
        """
        return list(self.fovs[self.fovs['on_tissue']].index)

    def get_considered_fov_counts(self):
        """
        Gets number of on-tissue FOVs

        Returns
        -------
        int
            Number of FOVs with >50% area on tissue
        """
        return len(self.fovs[self.fovs['on_tissue']].index)

    def get_false_positive_fovs(self, gene: str = '', dic: bool = False):
        """
        Get list of all FOVs not considered dropped due to false positive correction.

        Parameters
        ----------
        gene : str, optional
            If specified, return false positive FOVs for that gene. Default is ''.
        dic : bool, optional
            If True, return dictionary of genes and their false positive FOVs. Default is False.

        Returns
        -------
        list
            If `gene` is specified. List of false-positive corrected FOVs
        dict
            If `dic` is True. Dictionary of genes and their false positive FOVs
        """
        if gene != '':
            return list(self.fovs.loc[self.fovs['false_positive'].str.contains(gene), 'false_positive'].index)
        elif dic:
            return {gene: self.get_false_positive_fovs(gene=gene) for gene in self.genes}
        else:
            return list(self.fovs.iloc[np.where(self.fovs['false_positive'] != '')].index)

    def get_false_positive_fov_counts(self, gene: str = '', dic: bool = False):
        """
        Get number of FOVs not considered dropped due to false positive correction.

        Parameters
        ----------
        gene : str, optional
            If specified, return false positive FOV counts for that gene. Default is ''.
        dic : bool, optional
            If True, return dictionary of genes and their false positive FOV counts

        Returns
        -------

        """

    def dropout_summary(self, return_summary: bool = False):
        """
        Prints a summary of FOV dropout for the experiment

        Parameters
        ----------
        return_summary : bool, optional
            Whether to return summary as a string. Default is False.

        Returns
        -------
        summary : str
            If `return_summary` is True. Summary of FOV dropout for experiment.
        """
        dropped_fov_counts = self.get_dropped_fov_counts()
        considered_fov_counts = self.get_considered_fov_counts()
        dropout_count = self.get_dropout_count()
        dropped_gene_counts = self.get_dropped_gene_counts()
        considered_gene_counts = self.get_considered_gene_counts()

        summary = f'{dropped_fov_counts} unique FOVs were dropped out of {considered_fov_counts} considered FOVs ({100 * dropped_fov_counts / max(considered_fov_counts, 1):.2f}%)\n' +\
            f'FOVs with dropout dropped out in {dropout_count / max(dropped_fov_counts, 1):.2f} genes on average out of possible total of {considered_gene_counts} ({100 * dropout_count / max(considered_gene_counts * dropped_fov_counts, 1):.2f}%)\n' +\
            f'{dropped_gene_counts} genes were affected by dropout out of {considered_gene_counts} possible ({100 * dropped_gene_counts / max(considered_gene_counts, 1):.2f}%)\n' +\
            f'Genes with dropout averaged {dropout_count / max(considered_gene_counts, 1):.2f} dropped FOVs out of {considered_fov_counts} possible FOVs ({100 * dropout_count / max(dropped_gene_counts * considered_fov_counts, 1):.2f}%)'

        print(summary)

        if return_summary:
            return summary

    def draw_genes_dropped_per_fov(self, ax: plt.Axes = None, out_path: (Union[str, Path]) = ''):
        """
        Draws heatmap of number of genes dropped per FOV

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis object on which to plot. Default is None.
        out_path : (str, Path), optional
            Path at which to save plot. Default is ''.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Modified axis object with plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))

        # Dictionary with keys as FOVs and values as dropped genes
        genes_dropped_per_fov = self.get_dropped_gene_counts(dic=True)

        # Plot properties
        height = np.max(self.fovs['height'])
        width = np.max(self.fovs['width'])
        plot_min = min((np.min(self.fovs['x_min']),
                        np.min(self.fovs['y_min'])))
        plot_max = max((np.max(self.fovs['x_max']),
                        np.max(self.fovs['y_max'])))
        plt.setp(ax, xlim=(plot_min, plot_max), ylim=(plot_min, plot_max))

        ax.set_title('Genes Dropped per FOV')
        norm = color_normalize(vmin=0, vmax=max(np.max(list(genes_dropped_per_fov.values())), 1))

        for fov, val in genes_dropped_per_fov.items():
            ax.add_patch(Rectangle((self.fovs.loc[fov, 'x_min'], self.fovs.loc[fov, 'y_min']),
                                   width, height,
                                   facecolor=cm.viridis(norm(val)),
                                   fill=True,
                                   lw=0))
        self._determine_ticks(fig, ax, norm, max(np.max(list(genes_dropped_per_fov.values())), 1))

        if out_path != '':
            fig.savefig(out_path, bbox_inches='tight', dpi=400,
                        facecolor="#FFFFFF", edgecolor="#FFFFFF", transparent=False)

        plt.show()
        plt.close()

    def _determine_ticks(self, fig: plt.Figure, ax: plt.Axes, norm: color_normalize, max_tick: float):
        """
        Helper function to determine how many ticks to add to colorbars for dropout map

        Parameters
        ----------
        fig : matplotlib.Figure
            Figure on which to draw colorbar
        ax : matplotlib.Axes
            Axis on which to draw colorbar
        norm : matplotlib.colors.Normalize
            Normalization used for colormap
        max_tick : float
            Highest tick mark to be plotted on colorbar

        Returns
        -------
        None
        """
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.viridis), ax=ax)

        if max_tick <= 0.1:
            step_size = 0.01
        elif max_tick <= 1:
            step_size = 0.1
        elif max_tick <= 10:
            step_size = 1
        elif max_tick < 100:
            step_size = 10
        elif max_tick < 200:
            step_size = 20
        else:
            step_size = 50

        ticks = np.concatenate((np.arange(0, max_tick, step_size), np.array([max_tick])))
        if len(ticks) == 1:
            cb.set_ticks(np.array([0, 1]))
            cb.set_ticklabels(np.array([0, 1]))
        else:
            cb.set_ticks(ticks)
            cb.set_ticklabels(np.around(ticks, 3))
