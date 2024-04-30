from typing import Union
from pathlib import Path
import pandas as pd

class Experiment:

    def __init__(self, transcripts: pd.DataFrame = None, transcripts_path: Union[str, Path] = None):
        """
        Initilize an Experiment instance from transcripts dataframe 
        """
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
