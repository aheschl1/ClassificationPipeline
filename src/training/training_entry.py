import os.path
from typing import Tuple, Type
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.utils.constants import *
from src.utils.utils import get_dataset_name_from_id, read_json, get_dataloaders_from_fold, get_config_from_dataset
import click
import multiprocessing_logging  # for multiprocess logging https://github.com/jruere/multiprocessing-logging
import logging
import uuid
from torch.optim import SGD


class Trainer:
    def __init__(self, dataset_name: str, fold: int):
        self.dataset_name = dataset_name
        self.fold = fold
        self.output_dir = self._prepare_output_directory()
        self._assert_preprocess_ready_for_train()
        self.config = get_config_from_dataset(dataset_name)
        self.id_to_label = read_json(f"{PREPROCESSED_ROOT}/{dataset_name}/id_to_label.json")
        self.label_to_id = read_json(f"{PREPROCESSED_ROOT}/{dataset_name}/label_to_id.json")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seperator = "======================================================================="
        # Start on important stuff here
        self.train_dataloader, self.val_dataloader = self._get_dataloaders()
        self.model = self._get_model()
        self.loss = Trainer._get_loss()
        self.optim = self._get_optim()

    def _assert_preprocess_ready_for_train(self) -> None:
        preprocess_dir = f"{PREPROCESSED_ROOT}/{self.dataset_name}"
        assert os.path.exists(preprocess_dir), f"Preprocess root for dataset {self.dataset_name} does not exist. " \
                                               f"run src.preprocessing.preprocess_entry.py before training."
        assert os.path.exists(f"{preprocess_dir}/fold_{self.fold}"), f"The preprocessed data path for fold {self.fold}" \
                                                                     f" does not exist. womp womp"

    def _prepare_output_directory(self) -> str:
        session_id = str(uuid.uuid4())[0:5]
        output_dir = f"{RESULTS_ROOT}/{self.dataset_name}/fold_{self.fold}/{session_id}"
        os.makedirs(output_dir)
        logging.basicConfig(
            level=logging.DEBUG,
            filename=f"{output_dir}/logs.txt"
        )
        print(f"Sending logging and outputs to {output_dir}")
        return output_dir

    def _get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        This method is responsible for creating the augmentation and then fetching dataloaders.
        :return: Train and val dataloaders.
        """
        train_transforms = None
        val_transforms = None
        return get_dataloaders_from_fold(
            self.dataset_name,
            self.fold,
            train_transforms=train_transforms,
            val_transforms=val_transforms
        )

    def _train_single_epoch(self) -> float:
        running_loss = 0.
        total_items = 0
        for data, labels, points in self.train_dataloader:
            data = data.to(self.device)
            labels = labels.to(self.device)
            batch_size = data.shape[0]
            # do prediction and calculate loss
            predictions = self.model(data)
            loss = self.loss(predictions, labels)
            # update model
            loss.backward()
            self.optim.step()
            # gather data
            running_loss += loss.item()
            total_items += batch_size

        return running_loss/total_items

    def _eval_single_epoch(self) -> Tuple[float, float]:
        def count_correct(a: torch.Tensor, b: torch.Tensor) -> int:
            """
            Given two tensors, counts the agreement using a one-hot encoding.
            :param a: The first tensor
            :param b: The second tensor
            :return: Count of agreement at dim 1
            """
            assert a.shape == b.shape, f"Tensor a and b are different shapes."
            assert len(a.shape) == 2, f"Why is the prediction or gt shape of {a.shape}"
            results = torch.argmax(a, dim=1) == torch.argmax(b, dim=1)
            return results.sum().item()

        running_loss = 0.
        correct_count = 0.
        total_items = 0
        for data, labels, points in self.val_dataloader:
            data = data.to(self.device)
            labels = labels.to(self.device)
            batch_size = data.shape[0]
            # do prediction and calculate loss
            predictions = self.model(data)
            loss = self.loss(predictions, labels)
            running_loss += loss.item()
            correct_count += count_correct(predictions, labels)
            total_items += batch_size

        return running_loss/total_items, correct_count/total_items

    def train(self) -> None:
        epochs = self.config['epochs']
        for epoch in range(epochs):
            log(self.seperator)
            log(f"Epoch {epoch+1}/{epochs} starting.")
            self.model.train()
            mean_train_loss = self._train_single_epoch()
            self.model.eval()
            with torch.no_grad():
                mean_val_loss, val_accuracy = self._eval_single_epoch()
            log(f"Train loss: {mean_train_loss}")
            log(f"Val loss: {mean_val_loss}")
            log(f"Val accuracy: {val_accuracy}")

    def _get_optim(self) -> torch.optim:
        return SGD(
            self.model.parameters(),
            lr=self.config['lr']
        )

    def _get_model(self) -> nn.Module:
        return nn.Identity().to(self.device)

    @staticmethod
    def _get_loss() -> nn.Module:
        return nn.CrossEntropyLoss()


def log(*messages):
    for message in messages:
        logging.info(f"{message} ")


@click.command()
@click.option('--fold', '-f', help='Which fold to train.', type=int)
@click.option('--dataset_id', '-d', help='The dataset id to train.', type=str)
def main(fold: int, dataset_id: str) -> None:
    multiprocessing_logging.install_mp_handler()
    dataset_name = get_dataset_name_from_id(dataset_id)
    trainer = Trainer(dataset_name, fold)
    trainer.train()


if __name__ == "__main__":
    main()
