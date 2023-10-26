import glob
import sys

from tqdm import tqdm

# Adds the source to path for imports and stuff
sys.path.append("/home/andrew.heschl/Documents/ClassificationPipeline")
sys.path.append("/home/andrewheschl/PycharmProjects/classification_pipeline")
sys.path.append("/home/student/andrew/Documents/ClassificationPipeline")
import os.path
from typing import Dict

import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.constants import *
from src.utils.utils import get_dataset_name_from_id, get_config_from_dataset, \
    write_json
from src.json_models.src.model_generator import ModelGenerator
from src.inference.utils import get_dataset_from_folder
from src.utils.utils import batch_collate_fn


class Inferer:
    def __init__(self,
                 dataset_id: str,
                 fold: int,
                 result_folder: str,
                 config_name: str,
                 data_path: str,
                 weights: str
                 ):
        """
        Inferer for pipeline.
        :param dataset_id: The dataset id used for training
        :param fold: The fold to run inference with
        :param result_folder: The folder with the trained weights and config.
        :param config_name: The name of the config to use
        :param data_path: The path to the data to run inference on.
        :param weights: The name of the weights to load.
        """
        self.dataset_name = get_dataset_name_from_id(dataset_id)
        self.config = get_config_from_dataset(self.dataset_name, config_name)
        self.fold = fold
        self.lookup_root = f"{RESULTS_ROOT}/{self.dataset_name}/fold_{fold}/{result_folder}"
        print("Searching for model in results folder...")
        self.model_path = glob.glob(f"{RESULTS_ROOT}/{self.dataset_name}/fold_{fold}/{result_folder}/*.json")[0]
        print(f"Found {self.model_path}")
        self.weights = weights
        self.data_path = data_path
        assert os.path.exists(self.lookup_root)
        assert torch.cuda.is_available(), "No gpu available."
        self.dataloader = DataLoader(
            dataset=get_dataset_from_folder(data_path, self.dataset_name, fold, self.config),
            batch_size=1,
            num_workers=self.config['processes'],
            collate_fn=batch_collate_fn
        )
        self.model = self._get_model()

    def _get_model(self) -> nn.Module:
        """
        Loads the model and weights.
        :return:
        """
        gen = ModelGenerator(json_path=self.model_path)
        model = gen.get_model().to(0)
        print('Model log args: ')
        print(gen.get_log_kwargs())
        map_location = {'cuda:0': f'cuda:0'}
        weights = torch.load(f"{self.lookup_root}/{self.weights}.pth", map_location=map_location)
        model.load_state_dict(weights)
        return model

    def _infer_entry(self) -> Dict[str, int]:
        results = {}
        for data, _, points in tqdm(self.dataloader, desc="Running inference"):
            data = data.to(0)
            predictions = self.model(data)
            predicted_class = torch.argmax(predictions[0]).detach().item()
            results[points[0].path] = predicted_class
        return results

    def infer(self) -> None:
        save_path = f"{self.data_path}/results.json"
        self.model.eval()
        with torch.no_grad():
            results = self._infer_entry()
        print(f"Completed inference!")
        print(f"Saving results to {save_path}.")
        write_json(results, save_path)


@click.command()
@click.option('-dataset_id', '-d', required=True)
@click.option('-fold', '-f', required=True, type=int)
@click.option('-result_folder', '-r', required=True)
@click.option('-model_path', '-m', required=True)
@click.option('-data_path', '-data', required=True)
@click.option('-config', '-c', default='config')
@click.option('-weights', '-w', default='best')
def main(dataset_id: str, fold: int, result_folder: str, data_path: str, config: str, weights: str) -> None:
    inferer = Inferer(dataset_id, fold, result_folder, config, data_path, weights)
    inferer.infer()


if __name__ == "__main__":
    main()
