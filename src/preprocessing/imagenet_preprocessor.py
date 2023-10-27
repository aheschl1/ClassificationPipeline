import sys

# Adds the source to path for imports and stuff
sys.path.append("/home/andrew.heschl/Documents/ClassificationPipeline")
sys.path.append("/home/andrewheschl/PycharmProjects/classification_pipeline")
from src.utils.constants import *
from src.utils.utils import get_case_name_from_number
from src.preprocessing.preprocess_entry import Preprocessor
import shutil
import click
import os
from PIL import Image
import glob
from overrides import override
import yaml
from tqdm import tqdm
import json
from multiprocessing.pool import Pool, ThreadPool
import pandas as pd
from typing import Dict

class ImagenetPreprocessor(Preprocessor):
    def __init__(self, dataset_id: str, folds: int, processes: int, normalize: bool, imagenet_data_root: str, **kwargs):
        """
        Imagenet preprocessor subclass. Transforms data from original format to this pipelines format.
        :param dataset_id: The id of the dataset to use.
        :param folds: How many folds to make.
        :param processes: How many processes to use.
        :param normalize: If we should normalize the data.
        """
        super().__init__(dataset_id, folds, processes, normalize)
        assert os.path.exists(imagenet_data_root), f"The data root {imagenet_data_root} DNE."
        self.imagenet_data_root = imagenet_data_root
        self.current_case = 0
        self.fold = {
            "train":[],
            "val":[]
        }

    @override
    def process(self) -> None:
        """
        Moves and reshapes the plant data
        :return: Nothing
        """
        self._build_output_folder()
        # train stuff
        train_root = f"{self.imagenet_data_root}/ILSVRC/Data/CLS-LOC/train"
        train_samples = glob.glob(f"{train_root}/*/**.JPEG", recursive=True)
        # val stuff
        val_root = f"{self.imagenet_data_root}/ILSVRC/Data/CLS-LOC/val"
        val_labels = pd.read_csv(f"{self.imagenet_data_root}/LOC_val_solution.csv")
        # train set
        with ThreadPool(self.processes) as pool:
            with tqdm(total=len(train_samples), desc="Processing train cases") as pbar:
                for _ in pool.imap_unordered(self._process_train_case, train_samples):
                    pbar.update()
        # prepare for val
        val_samples = glob.glob(f"{val_root}/*.JPEG")
        val_data = [(path, 
                     val_labels.loc['ImageId' == path.split('/')[-1].split('.')[0]]['PredictionString']) 
                     for path in val_samples]
        # val set
        with ThreadPool(self.processes) as pool:
            with tqdm(total=len(train_samples), desc="Processing val cases") as pbar:
                for _ in pool.imap_unordered(self._process_val_case, val_data):
                    pbar.update()
        super().process()

    def _process_train_case(self, sample_path: str):
        """
        Move train case at sample_path to proper location
        """
        label = sample_path.split('/')[-2]
        # build label folder if nececarry
        try:
            os.mkdir(f"{self.raw_root}/{label}")
        except Exception: ...
        # get case name

    def _process_val_case(self, data: tuple):
        """
        Process val case and extract label
        """
        sample_path, prediction_string = data
        label = self._get_label_from_prediction_string(prediction_string)
        # get case name
        case_id = self.current_case
        self.current_case += 1
        case_name = get_case_name_from_number(case_id)
        self.fold['val'].append(case_name)
        shutil.copy(sample_path, f"{self.raw_root}/{label}/{case_name}.JPEG")
    
    def _get_label_from_prediction_string(self, prediction_string: str):
        """
        Convert predictionString from validation labels to proper class.
        """
        chunks = prediction_string.split(' ')
        labels = set()
        for chunk in chunks:
            if 'n' in chunk:
                labels.add(chunk)
        assert len(labels) == 1, f"Error processing prediction string: {prediction_string}"
        return labels[0]

    @override
    def get_folds(self, k: int) -> Dict[int, Dict[str, list]]:
        return {'0':self.fold}

    @override
    def post_preprocessing(self):
        ...

    def _build_output_folder(self) -> None:
        """
        Prepares the output folder for usage.
        :return: Nothing
        """
        self.raw_root = f"{RAW_ROOT}/{self.dataset_name}"
        shutil.rmtree(self.raw_root)
        assert not os.path.exists(self.raw_root), ("The raw data root already exists. "
                                                   "Try a new dataset name, or delete the folder.")
        os.makedirs(self.raw_root)


@click.command()
@click.option('--csv', '-c', help="Path to csv with labels.", required=True)
@click.option('--data', '-d', help="Path to the data root.", required=True)
@click.option('--set_id', '-s', help="The dataset id to target.", default=3, type=int)
def main(csv: str, data: str, set_id: int):
    # processor = CardiacEchoViewPreprocessor(
    #     csv, data, set_id
    # )
    # processor.process()
    pass


if __name__ == "__main__":
    main()
