from unittest.mock import TestCase, patch
from merquaco.experiment import Experiment
import pandas as pd
import os
from pathlib import Path
import json

# set path to the data folder 
data_path = os.path.join(os.path.dirname(__file__), '..', 'data')

# read config
with open(Path(data_path, "test_config.json"), 'r') as file:
    test_config = json.load(file)

class ExperimentObjectTests(TestCase):

    def test_experiment_init():

        # Define transcripts input for each possible type
        transcripts_df = pd.read_csv(Path(data_path, "test_detected_transcripts.csv"))
        transcripts_str = str(Path(data_path, "test_detected_transcripts.csv"))
        transcripts_path = Path(data_path, "test_detected_transcripts.csv")

        # define ilastik paths input for each possible type
        ilastik_paths_dict = test_config
        ilastik_paths_str = str(Path(data_path, "test_config.json"))
        iastik_paths_path = Path(data_path, "test_config.json")

        # mock data_processing functions
        with patch('merquaco.experiment.data_processing.process_input') as mock_process_input, \
            patch('merquaco.experiment.data_processing.check_if_none', return_value=False) as mock_check_if_none, \
            patch('merquaco.experiment.pc.generate_transcripts_mask') as mock_generate_mask:
            
            # Configure the mock to return data as expected
            mock_process_input.side_effect = [ilastik_paths_dict, transcripts_df, transcripts_input]
            