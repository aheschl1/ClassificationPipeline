import sys

# Adds the source to path for imports and stuff
sys.path.append("/home/andrew.heschl/Documents/ClassificationPipeline")
sys.path.append("/home/andrewheschl/PycharmProjects/classification_pipeline")
from src.utils.utils import get_case_name_from_number
from src.utils.constants import *
from src.preprocessing.preprocess_entry import Preprocessor
import shutil
import SimpleITK as sitk
import click
import pandas as pd
import os
from multiprocessing.pool import ThreadPool
import numpy as np
from PIL import Image
import uuid
import glob
from typing import Dict
from sklearn.model_selection import train_test_split
from overrides import override


class CardiacEchoViewPreprocessor(Preprocessor):
    def __init__(self, dataset_id: str, folds: int, processes: int, normalize: bool,
                 csv_path: str, data_root: str):
        """
        Cardiac view preprocessor subclass. Transforms data from original format to this pipelines format.
        Also converts data to 2d slices, and applies movement mask.
        :param dataset_id: The id of the dataset to use.
        :param folds: How many folds to make.
        :param processes: How many processes to use.
        :param normalize: If we should normalize the data.
        :param csv_path: The path to the cardiac labeling csv.
        :param data_root: The root of the cardiac data.
        """
        super().__init__(dataset_id, folds, processes, normalize)
        assert os.path.exists(csv_path), f"The csv path {csv_path} DNE."
        assert os.path.exists(data_root), f"The data root {data_root} DNE."
        self.data_info = pd.read_excel(csv_path)
        self.data_root = data_root
        self.case_grouping = []
        self.uuid_case_mapping = None
        self.target_shape = (800, 600)

    @override
    def process(self) -> None:
        """
        Starts the processing of the data on process pool, and calls super version.
        Starts by naming the dta randomly, and then renames files to proper format.
        :return: Nothing
        """
        self._build_output_folder()
        _, row_series = zip(*self.data_info.iterrows())
        data = [row for _, row in
                enumerate(row_series)]
        with ThreadPool(self.processes) as pool:
            pool.map(self._process_case, data)
        # rename cases to be correct
        cases = glob.glob(f"{DATA_ROOT}/raw/{self.dataset_name}/**/*.png", recursive=True)
        uuid_case_mapping = {}
        for current_case, path in enumerate(cases):
            case_name = get_case_name_from_number(current_case)
            old_id = path.split('/')[-1].split('.')[0]
            uuid_case_mapping[old_id] = case_name
            shutil.move(path, '/'.join(path.split('/')[0:-1] + [f"{case_name}.png"]))
        self.uuid_case_mapping = uuid_case_mapping
        super().process()

    @override
    def get_folds(self, k: int) -> Dict[int, Dict[str, list]]:
        """
        Gets random fold at 80/20 split. Returns in a map.
        :param k: How many folds for kfold cross validation.
        :return: Folds map
        """
        assert k == 1, "Echo preprocessor can only do one fold right now. The dev sucked too much"
        train_groups, test_groups = train_test_split(self.case_grouping)
        train_cases, val_cases = [], []
        for group in train_groups:
            for case in group:
                train_cases.append(self.uuid_case_mapping[case])
        for group in test_groups:
            for case in group:
                val_cases.append(self.uuid_case_mapping[case])
        return {
            0: {
                "train": train_cases,
                "val": val_cases
            }
        }

    @override
    def assert_all_images_same_shape(self) -> None:
        """
        Ensures that all datapoints have the same shape.
        Overwritten because we reshape in this subclass.
        :return:
        """
        ...

    def _build_output_folder(self) -> None:
        """
        Prepares the output folder for usage.
        :return: Nothing
        """
        raw_root = f"{DATA_ROOT}/raw/{self.dataset_name}"
        shutil.rmtree(raw_root)
        assert not os.path.exists(raw_root), ("The raw data root already exists. "
                                              "Try a new dataset name, or delete the folder.")
        os.makedirs(raw_root)

    def _process_case(self, row: pd.Series) -> None:
        """
        Processes a single dicom instance.
        :return: Nothing
        """
        if '-D' in row[LABEL]:
            row[LABEL] = row[LABEL].replace('-D', '')
        output_folder = f"{DATA_ROOT}/raw/{self.dataset_name}/{row[LABEL]}"
        # create the labels folder
        try:
            os.mkdir(output_folder)
        except FileExistsError:
            ...
        # get data
        data_path = f"{self.data_root}/{row[PATIENT_PATH]}/{row[FILE_NAME]}"
        if not os.path.exists(data_path):
            print(f"Skipping {row[PATIENT_PATH]}/{row[FILE_NAME]}. DNE")
            return
        print(f"Working on {row[PATIENT_PATH]}/{row[FILE_NAME]}")
        data = CardiacEchoViewPreprocessor._read_dicom(data_path)
        data = CardiacEchoViewPreprocessor._apply_movement_mask(data)
        case_group = []
        for im_slice in data:
            case_id = uuid.uuid4()
            case_group.append(str(case_id))
            im_slice = Image.fromarray(im_slice).resize(self.target_shape)
            im_slice.save(f"{output_folder}/{case_id}.png")
        self.case_grouping.append(case_group)
        print(f"Completed {row[PATIENT_PATH]}/{row[FILE_NAME]}!")

    @staticmethod
    def _apply_movement_mask(image_array: np.array, k: int = 100) -> np.array:
        """
        Zeroes image region without movement across zeroth axis.
        :param image_array: The image in question
        :param k: How many frames to look across.
        :return: The masked image.
        """
        slices, height, width, ch = image_array.shape

        mask = np.zeros((height, width, ch), dtype='uint8')
        steps = min(k, slices)
        for i in range(steps - 1):
            mask[image_array[i, :, :, :] != image_array[i + 1, :, :, :]] = 1

        output = image_array * mask

        return output

    @staticmethod
    def _read_dicom(path: str) -> np.array:
        """
        Reads image and returns array.
        :param path: The path to the file to read.
        :return: The array of the dicom image.
        """
        return sitk.GetArrayFromImage(sitk.ReadImage(path))


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
