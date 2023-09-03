import logging
import os.path
import time
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
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import matplotlib.pyplot as plt
import datetime


class LogHelper:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.losses_train, self.losses_val, self.accuracies_val = [], [], []

    def epoch_end(self, train_loss: float, val_loss: float, val_accuracy: float):
        self.losses_train.append(train_loss)
        self.losses_val.append(val_loss)
        self.accuracies_val.append(val_accuracy)

    def save_figs(self) -> None:
        num_epochs = len(self.accuracies_val)
        # accuracy
        plt.plot([i for i in range(num_epochs)], self.accuracies_val)
        plt.title('Validation Accuracy VS Epoch')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig(f"{self.output_dir}/graph_accuracies.png")
        plt.close()
        # train loss
        plt.plot([i for i in range(num_epochs)], self.losses_train)
        plt.title('Train Loss VS Epoch')
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.savefig(f"{self.output_dir}/graph_train_loss.png")
        plt.close()
        # val loss
        plt.plot([i for i in range(num_epochs)], self.losses_val)
        plt.title('Validation Loss VS Epoch')
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.savefig(f"{self.output_dir}/graph_val_loss.png")
        plt.close()
        # both losses
        plt.plot([i for i in range(num_epochs)], self.losses_val, label="Val Loss")
        plt.plot([i for i in range(num_epochs)], self.losses_train, label="Train Loss")
        plt.title('Losses VS Epoch')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{self.output_dir}/graph_both_loss.png")
        plt.close()


class Trainer:
    def __init__(self,
                 dataset_name: str,
                 fold: int,
                 save_latest: bool,
                 model_path: str,
                 gpu_id: int,
                 unique_folder_name: str,
                 checkpoint_name: str = None):
        """
        Trainer class for training and checkpointing of networks.
        :param dataset_name: The name of the dataset to use.
        :param fold: The fold in the dataset to use.
        :param save_latest: If we should save a checkpoint each epoch
        :param model_path: The path to the json that defines the architecture.
        :param gpu_id: The gpu for this process to use.
        :param checkpoint_name: None if we should train from scratch, otherwise the model weights that should be used.
        """
        assert torch.cuda.is_available(), "This pipeline only supports GPU training. No GPU was detected, womp womp."
        self.dataset_name = dataset_name
        self.fold = fold
        self.save_latest = save_latest
        self.device = gpu_id
        self.output_dir = self._prepare_output_directory(unique_folder_name)
        self.log_helper = LogHelper(self.output_dir)
        self._assert_preprocess_ready_for_train()
        self.config = get_config_from_dataset(dataset_name)
        if gpu_id == 0:
            log("Config:", self.config)
        self.id_to_label = read_json(f"{PREPROCESSED_ROOT}/{dataset_name}/id_to_label.json")
        self.label_to_id = read_json(f"{PREPROCESSED_ROOT}/{dataset_name}/label_to_id.json")
        self.seperator = "======================================================================="
        # Start on important stuff here
        self.train_dataloader, self.val_dataloader = self._get_dataloaders()
        self.model = self._get_model(model_path)
        self.model = DDP(self.model, device_ids=[gpu_id])
        if checkpoint_name is not None:
            self._load_checkpoint(checkpoint_name)
        self.loss = self._get_loss()
        self.optim = self._get_optim()
        log(f"Trainer finished initialized on rank {gpu_id}.")
        dist.barrier()

    def _assert_preprocess_ready_for_train(self) -> None:
        """
        Ensures that the preprocess folder exists for the current dataset,
        and that the fold specified has been processed.
        :return: None
        """
        preprocess_dir = f"{PREPROCESSED_ROOT}/{self.dataset_name}"
        assert os.path.exists(preprocess_dir), f"Preprocess root for dataset {self.dataset_name} does not exist. " \
                                               f"run src.preprocessing.preprocess_entry.py before training."
        assert os.path.exists(f"{preprocess_dir}/fold_{self.fold}"), \
            f"The preprocessed data path for fold {self.fold} does not exist. womp womp"

    def _prepare_output_directory(self, session_id: str) -> str:
        """
        Prepares the output directory, and sets up logging to it.
        :return: str which is the output directory.
        """
        output_dir = f"{RESULTS_ROOT}/{self.dataset_name}/fold_{self.fold}/{session_id}"
        os.makedirs(output_dir)
        logging.basicConfig(
            level=logging.INFO,
            filename=f"{output_dir}/logs.txt"
        )
        if self.device == 0:
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
            val_transforms=val_transforms,
            sampler=DistributedSampler
        )

    def _train_single_epoch(self) -> float:
        """
        The training of each epoch is done here.
        :return: The mean loss of the epoch.
        """
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
        """
        Runs evaluation for a single epoch.
        :return: The mean loss and mean accuracy respectively.
        """

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
        """
        Starts the training process.
        :return: None
        """
        epochs = self.config['epochs']
        start_time = time.time()
        best_val_loss = 9090909.  # Arbitrary large number
        # last values to show change
        last_train_loss = 0
        last_val_loss = 0
        last_val_accuracy = 0
        for epoch in range(epochs):
            if self.device == 0:
                log(self.seperator)
                log(f"Epoch {epoch + 1}/{epochs} starting.")
            self.model.train()
            mean_train_loss = self._train_single_epoch()
            self.model.eval()
            with torch.no_grad():
                mean_val_loss, val_accuracy = self._eval_single_epoch()
            if self.save_latest:
                self.save_model_weights('latest')  # saving model every epoch
            if self.device == 0:
                self.log_helper.epoch_end(mean_train_loss, mean_val_loss, val_accuracy)
                log(f"Train loss: {mean_train_loss} --change-- {mean_train_loss - last_train_loss}")
                log(f"Val loss: {mean_val_loss} --change-- {mean_val_loss - last_val_loss}")
                log(f"Val accuracy: {val_accuracy} --change-- {val_accuracy - last_val_accuracy}")
                self.log_helper.save_figs()
            # update 'last' values
            last_train_loss = mean_train_loss
            last_val_loss = mean_val_loss
            last_val_accuracy = val_accuracy
            # If best model, save!
            if mean_val_loss < best_val_loss:
                if self.device == 0:
                    log('Nice, that\'s a new best loss. Saving the weights!')
                best_val_loss = mean_val_loss
                self.save_model_weights('best')

        # Now training is completed, print some stuff
        dist.barrier()
        self.save_model_weights('final')  # save the final weights
        end_time = time.time()
        seconds_taken = end_time - start_time
        if self.device == 0:
            log(self.seperator)
            self.log_helper.save_figs()
            log(f"Finished training {epochs} epochs.")
            log(f"{seconds_taken} seconds")
            log(f"{seconds_taken / 60} minutes")
            log(f"{(seconds_taken / 3600)} hours")
            log(f"{(seconds_taken / 86400)} days")

    def save_model_weights(self, save_name: str) -> None:
        """
        Save the weights of the model, only if the current device is 0.
        :param save_name: The name of the checkpoint to save.
        :return: None
        """
        if self.device == 0:
            path = f"{self.output_dir}/{save_name}.pth"
            torch.save(self.model.state_dict(), path)

    def _get_optim(self) -> torch.optim:
        """
        Instantiates and returns the optimizer.
        :return: Optimizer object.
        """
        if self.device == 0:
            log(f"Optim being used is SGD")
        return SGD(
            self.model.parameters(),
            lr=self.config['lr'],
            momentum=self.config.get('momentum', 0.9),
            weight_decay=self.config.get('weight_decay', 0)
        )

    def _get_model(self, path: str) -> nn.Module:
        """
        Given the path to the network, generates the model, and verifies its integrity with onnx.
        Additionally, the model will be saved with onnx for visualization at https://netron.app
        :param path: The path to the json architecture definition.
        :return: The pytorch network module.
        """

        def onnx_export() -> None:
            """
            Generates an onnx file with network topology data for visualization.
            :return: None
            """
            log(f"Generating onnx model for visualization and to verify model sanity...\n")
            dummy_input = torch.randn(1, *self.data_shape, device=torch.device(self.device))
            input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
            output_names = ["output1"]
            file = f"{self.output_dir}/model_topology.onnx"
            torch.onnx.export(model, dummy_input, file,
                              verbose=False,
                              input_names=input_names,
                              output_names=output_names)
            log(f"Saved onnx model to {file}. Architecture works!")
            log(f"Go to https://netron.app/ to view the architecture.")
            log(self.seperator)

        gen = ModelGenerator(json_path=path)
        model = gen.get_model().to(self.device)
        if self.device == 0:
            log('Model log args: ')
            log(gen.get_log_kwargs())
        if self.device == 0:
            onnx_export()
        return model

    def _load_checkpoint(self, weights_name) -> None:
        """
        Loads network checkpoint onto the DDP model.
        :param weights_name: The name of the weights to load in the form of *result folder*/*weight name*.pth
        :return: None
        """
        log(f"Starting to load weights on process {self.device}...")
        assert len(weights_name.split('/')) == 2, \
            ("To load weights provide a path string in the format of *result folder*/*weight name*. "
             "For example: h4f56/final.pth")
        result_folder = weights_name.split('/')[0]
        weights_name = weights_name.split('/')[-1]
        weights_path = f"{RESULTS_ROOT}/{self.dataset_name}/fold_{self.fold}/{result_folder}/{weights_name}"
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f'The file {weights_path} does not exist. Check the weights argument.')
        map_location = {'cuda:0': f'cuda:{self.device}'}
        weights = torch.load(weights_path, map_location=map_location)
        self.model.load_state_dict(weights)
        log(f"Successfully loaded weights on rank {self.device}.")

    def _get_loss(self) -> nn.Module:
        """
        Build the criterion object.
        :return: The loss function to be used.
        """
        if self.device == 0:
            log("Loss being used is nn.CrossEntropyLoss()")
        return nn.CrossEntropyLoss()

    @property
    def data_shape(self) -> Tuple[int, int, int]:
        """
        Property which is the data shape we are training on.
        :return: Shape of data.
        """
        return self.train_dataloader.dataset.datapoints[0].get_data().shape


def log(*messages):
    """
    Prints to screen and logs a message.
    :param messages: The messages to display and log.
    :return: None
    """
    print(*messages)
    for message in messages:
        logging.info(f"{message} ")


def setup_ddp(rank: int, world_size: int) -> None:
    """
    Prepares the ddp on a specific process.
    :param rank: The device we are initializing.
    :param world_size: The total number of devices.
    :return: None
    """
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "12345"
    init_process_group(backend='nccl', rank=rank, world_size=world_size)


def ddp_training(rank, world_size: int, dataset_id: int,
                 fold: int, save_latest: bool, model: str, session_id: str, load_weights: str) -> None:
    """
    Launches training on a single process using pytorch ddp.
    :param session_id: Session id to be used for folder name on output.
    :param rank: The rank we are starting.
    :param world_size: The total number of devices
    :param dataset_id: The dataset to train on
    :param fold: The fold to train
    :param save_latest: If the latest checkpoint should be saved per epoch
    :param model: The path to the model json definition
    :param load_weights: The weights to load, or None
    :return: Nothing
    """
    setup_ddp(rank, world_size)
    dataset_name = get_dataset_name_from_id(dataset_id)
    trainer = Trainer(dataset_name, fold, save_latest, model, rank, session_id, load_weights)
    trainer.train()
    destroy_process_group()


@click.command()
@click.option('-fold', '-f', help='Which fold to train.', type=int)
@click.option('-dataset_id', '-d', help='The dataset id to train.', type=str)
@click.option('-model', '-m', help='Path to model json definition.', type=str)
@click.option('--save_latest', '--sl', help='Should weights be saved every epoch', type=bool, is_flag=True)
@click.option('-state', '-s',
              help='Whether to trigger 2d or 3d model architecture. Only works with some modules.',
              type=str, default=ModuleStateController.TWO_D)
@click.option('--gpus', '-g', help='How many gpus for ddp', type=int, default=1)
@click.option('--load_weights', '-l', help='Weights to continue training with', type=str, default=None)
def main(fold: int, dataset_id: str, model: str, save_latest: bool, state: str, gpus: int, load_weights: str) -> None:
    """
    Initializes training on multiple processes, and initializes logger.
    :param fold:
    :param dataset_id:
    :param model:
    :param save_latest:
    :param state:
    :param gpus:
    :param load_weights:
    :return:
    """
    multiprocessing_logging.install_mp_handler()
    assert os.path.exists(model), "The model path you specified doesn't exist."
    assert state in ['2d', '3d'], f"Specified state {state} does not exist. Use 2d or 3d"
    print(f"Module state being set to {state}.")
    # This sets the behavior of some modules in json models utils.
    ModuleStateController.set_state(state)
    session_id = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%f')

    mp.spawn(
        ddp_training,
        args=(gpus, dataset_id, fold, save_latest, model, session_id, load_weights),
        nprocs=gpus
    )


if __name__ == "__main__":
    main()
