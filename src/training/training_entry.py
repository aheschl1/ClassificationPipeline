import logging
import os.path
import time
import uuid
from typing import Tuple

import click
import multiprocessing_logging  # for multiprocess logging https://github.com/jruere/multiprocessing-logging
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from src.utils.constants import *
from src.utils.utils import get_dataset_name_from_id, read_json, get_dataloaders_from_fold, get_config_from_dataset
from src.json_models.src.model_generator import ModelGenerator
from src.json_models.src.modules import ModuleStateController


class Trainer:
    def __init__(self, dataset_name: str, fold: int, save_latest: bool, model_path: str):
        self.dataset_name = dataset_name
        self.fold = fold
        self.save_latest = save_latest
        self.output_dir = self._prepare_output_directory()
        self._assert_preprocess_ready_for_train()
        self.config = get_config_from_dataset(dataset_name)
        log(self.config)
        self.id_to_label = read_json(f"{PREPROCESSED_ROOT}/{dataset_name}/id_to_label.json")
        self.label_to_id = read_json(f"{PREPROCESSED_ROOT}/{dataset_name}/label_to_id.json")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seperator = "======================================================================="
        # Start on important stuff here
        self.train_dataloader, self.val_dataloader = self._get_dataloaders()
        self.model = self._get_model(model_path)
        self.loss = Trainer._get_loss()
        self.optim = self._get_optim()

    def _assert_preprocess_ready_for_train(self) -> None:
        preprocess_dir = f"{PREPROCESSED_ROOT}/{self.dataset_name}"
        assert os.path.exists(preprocess_dir), f"Preprocess root for dataset {self.dataset_name} does not exist. " \
                                               f"run src.preprocessing.preprocess_entry.py before training."
        assert os.path.exists(f"{preprocess_dir}/fold_{self.fold}"), \
            f"The preprocessed data path for fold {self.fold} does not exist. womp womp"

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

        return running_loss / total_items

    def _eval_single_epoch(self) -> Tuple[float, float]:
        def count_correct(a: torch.Tensor, b: torch.Tensor) -> int:
            """
            Given two tensors, counts the agreement using a one-hot encoding.
            :param a: The first tensor
            :param b: The second tensor
            :return: Count of agreement at dim 1
            """
            assert a.shape == b.shape, f"Tensor a and b are different shapes. Got {a.shape} and {b.shape}"
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

        return running_loss / total_items, correct_count / total_items

    def train(self) -> None:
        epochs = self.config['epochs']
        start_time = time.time()
        best_val_loss = 9090909.  # Arbitrary large number
        # last values to show change
        last_train_loss = 0
        last_val_loss = 0
        last_val_accuracy = 0
        for epoch in range(epochs):
            log(self.seperator)
            log(f"Epoch {epoch + 1}/{epochs} starting.")
            self.model.train()
            mean_train_loss = self._train_single_epoch()
            self.model.eval()
            with torch.no_grad():
                mean_val_loss, val_accuracy = self._eval_single_epoch()
            if self.save_latest:
                self.save_model_weights('latest')  # saving model every epoch
            log(f"Train loss: {mean_train_loss} =change= {mean_train_loss - last_train_loss}")
            log(f"Val loss: {mean_val_loss} =change= {mean_val_loss - last_val_loss}")
            log(f"Val accuracy: {val_accuracy} =change= {val_accuracy - last_val_accuracy}")
            # update 'last' values
            last_train_loss = mean_train_loss
            last_val_loss = mean_val_loss
            last_val_accuracy = val_accuracy
            # If best model, save!
            if mean_val_loss < best_val_loss:
                log('Nice, that\'s a new best loss. Saving the weights!')
                best_val_loss = mean_val_loss
                self.save_model_weights('best')

        # Now training is completed, print some stuff
        self.save_model_weights('final')  # save the final weights
        end_time = time.time()
        seconds_taken = end_time - start_time
        log(self.seperator)
        log(f"Finished training {epochs} epochs.")
        log(f"{seconds_taken} seconds")
        log(f"{seconds_taken / 60} minutes")
        log(f"{(seconds_taken / 3600)} hours")
        log(f"{(seconds_taken / 86400)} days")

    def save_model_weights(self, save_name: str) -> None:
        path = f"{self.output_dir}/{save_name}.pth"
        torch.save(self.model.state_dict(), path)

    def _get_optim(self) -> torch.optim:
        log(f"Optim being used is SGD")
        return SGD(
            self.model.parameters(),
            lr=self.config['lr']
        )

    def _get_model(self, path: str) -> nn.Module:
        gen = ModelGenerator(json_path=path)
        model = gen.get_model().to(self.device)
        log('Model log args: ')
        log(gen.get_log_kwargs())
        return model

    @staticmethod
    def _get_loss() -> nn.Module:
        log("Loss being used is nn.CrossEntropyLoss()")
        return nn.CrossEntropyLoss()


def log(*messages):
    print(*messages)
    for message in messages:
        logging.info(f"{message} ")


@click.command()
@click.option('-fold', '-f', help='Which fold to train.', type=int)
@click.option('-dataset_id', '-d', help='The dataset id to train.', type=str)
@click.option('-model', '-m', help='Path to model json definition.', type=str)
@click.option('--save_latest', '--sl', help='Should weights be saved every epoch', type=bool, is_flag=True)
@click.option('-state', '-s', help='Whether to trigger 2d or 3d model architecture. Only works with some modules.',
              type=str, default=ModuleStateController.TWO_D)
def main(fold: int, dataset_id: str, model: str, save_latest: bool, state: str) -> None:
    multiprocessing_logging.install_mp_handler()
    assert os.path.exists(model), "The model path you specified doesn't exist."
    assert state in ['2d', '3d'], f"Specified state {state} does not exist. Use 2d or 3d"
    print(f"Module state being set to {state}.")
    # This sets the behavior of some modules in json models utils.
    ModuleStateController.set_state(state)
    dataset_name = get_dataset_name_from_id(dataset_id)
    trainer = Trainer(dataset_name, fold, save_latest, model)
    trainer.train()


if __name__ == "__main__":
    main()
