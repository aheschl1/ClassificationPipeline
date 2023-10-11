import sys

# Adds the source to path for imports and stuff
sys.path.append("/home/andrew.heschl/Documents/ClassificationPipeline")
sys.path.append("/home/andrewheschl/PycharmProjects/classification_pipeline")
from src.utils.constants import *
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
from multiprocessing.pool import Pool


class NutrientPreprocessor(Preprocessor):
    def __init__(self, dataset_id: str, folds: int, processes: int, normalize: bool, nutrient_data_root: str, **kwargs):
        """
        Cardiac view preprocessor subclass. Transforms data from original format to this pipelines format.
        Also converts data to 2d slices, and applies movement mask.
        :param dataset_id: The id of the dataset to use.
        :param folds: How many folds to make.
        :param processes: How many processes to use.
        :param normalize: If we should normalize the data.
        """
        super().__init__(dataset_id, folds, processes, normalize)
        assert os.path.exists(nutrient_data_root), f"The data root {nutrient_data_root} DNE."
        self.nutrient_data_root = nutrient_data_root
        self.case_name_to_og_name = {}

    @override
    def process(self) -> None:
        """
        Moves and reshapes the plant data
        :return: Nothing
        """
        self._build_output_folder()
        roots = [f"{self.nutrient_data_root}/WR2021", f"{self.nutrient_data_root}/WW2020"]
        current_case = 0
        for root in roots:
            images = f"{root}/images"
            labels = f"{root}/labels_trainval.yml"

            with open(labels) as file:
                labels = yaml.safe_load(file)

            for image_name, label in tqdm(labels.items()):
                case_name = '0' * (5 - len(str(current_case))) + str(current_case)
                case_name = f"case_{case_name}.jpg"
                self.case_name_to_og_name[case_name.split('.')[0]] = image_name.split('.')[0]
                os.makedirs(f"{self.raw_root}/{label}", exist_ok=True)
                shutil.copy(f"{images}/{image_name}", f"{self.raw_root}/{label}/{case_name}")
                current_case += 1
        target_folder = f"{PREPROCESSED_ROOT}/{self.dataset_name}"
        with open(f'{target_folder}/plant_name_mapping.json', 'w+') as file:
            json.dump(self.case_name_to_og_name, file)
        self.reshape_images()
        super().process()

    def reshape_images(self):
        """
        Reshapes all images to 1024 1024
        :return:
        """
        path = f"{self.raw_root}/**/*.jpg"

        paths = glob.glob(path)
        with Pool(self.processes) as pool:
            with tqdm(total=len(paths), desc="Reshaping images") as pbar:
                for _ in pool.imap_unordered(NutrientPreprocessor._reshape_single, paths):
                    pbar.update()

    @staticmethod
    def _reshape_single(p):
        image = Image.open(p).resize((1024, 1024))
        image.save(p)

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
