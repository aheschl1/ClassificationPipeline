from src.utils.utils import write_json, get_dataset_name_from_id, check_raw_exists, get_raw_datapoints, \
    get_dataloaders_from_fold
from src.preprocessing.utils import maybe_make_preprocessed, get_labels_from_raw
from src.utils.constants import *
from src.preprocessing.splitting import Splitter
import click
from typing import Dict, Union, Tuple, Type
from tqdm import tqdm
import time
from PIL import ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Preprocessor:
    def __init__(self, dataset_id: str, folds: int, processes: int, normalize: bool, **kwargs):
        """
        :param folds: How many folds to generate.
        :param processes: How many processes should be used.
        :param normalize: Should normalized data be saved.
        :param dataset_id: The id of the dataset.

        This is the main driver for preprocessing.
        """
        self.dataset_name = get_dataset_name_from_id(dataset_id)
        self.processes = processes
        self.normalize = normalize
        self.labels = None
        self.datapoints = None
        self.folds = folds
        assert check_raw_exists(self.dataset_name), \
            f"It appears that you haven't created the 'raw/{self.dataset_name}' folder. Womp womp"
        maybe_make_preprocessed(self.dataset_name, query_overwrite=True)
        # We export the config building to a new method
        self.build_config()

    def process(self) -> None:
        # Here we will find what labels are present in the dataset. We will also map them to int labels, and save the
        # mappings.
        self.labels = get_labels_from_raw(self.dataset_name)
        assert len(self.labels) > 1, "We only found one label folder, maybe the folder structure is wrong."
        label_to_id_mapping, id_to_label_mapping = self.map_labels_to_id(return_inverse=True)
        write_json(label_to_id_mapping, f"{PREPROCESSED_ROOT}/{self.dataset_name}/label_to_id.json")
        write_json(id_to_label_mapping, f"{PREPROCESSED_ROOT}/{self.dataset_name}/id_to_label.json")
        self.datapoints = get_raw_datapoints(self.dataset_name, label_to_id_mapping)
        self.verify_dataset_integrity()
        # Label stuff done, start with fetching data. We will also save a case to label mapping.
        case_to_label_mapping = self.get_case_to_label_mapping()
        write_json(case_to_label_mapping, f"{PREPROCESSED_ROOT}/{self.dataset_name}/case_label_mapping.json")
        splits_map = self.get_folds(self.folds)
        write_json(splits_map, f"{PREPROCESSED_ROOT}/{self.dataset_name}/folds.json")
        # We now have the folds: time to preprocess the data
        for fold_id, _ in splits_map.items():
            self.process_fold(fold_id)
        self.post_preprocessing()

    def post_preprocessing(self):...

    def build_config(self) -> None:
        """
        Creates the config.json file that should contain training hyperparameters. Hardcode default values here.
        :return: None
        """
        config = {
            'batch_size': 64,
            'processes': self.processes,
            'lr': 0.01,
            'epochs': 100,
            'momentum': 0,
            'weight_decay': 0,
            'target_size': [224, 224]
        }
        write_json(config, f"{PREPROCESSED_ROOT}/{self.dataset_name}/config.json")

    def verify_dataset_integrity(self) -> None:
        """
        Ensures that all datapoints have the same shape.
        :return:
        """
        names = set()
        for point in tqdm(self.datapoints, desc="Verifying dataset integrity"):
            names_before = len(names)
            names.add(point.case_name)
            assert names_before < len(names), f"The name {point.case_name} is in the dataset at least 2 times :("
    
    def get_folds(self, k: int) -> Dict[int, Dict[str, list]]:
        """
        Gets random fold at 80/20 split. Returns in a map.
        :param k: How many folds for kfold cross validation.
        :return: Folds map
        """
        splitter = Splitter(self.datapoints, k)
        return splitter.get_split_map()

    def map_labels_to_id(self, return_inverse: bool = False) -> Union[
        Tuple[Dict[str, int], Dict[int, str]], Dict[str, int]]:
        """
        :param return_inverse: If true returns id:name as well as name:id mapping.
        :return: Dict that maps label name to id.
        """
        mapping = {}
        inverse = {}

        for i, label in enumerate(self.labels):
            mapping[label] = i
            inverse[i] = label

        if return_inverse:
            return mapping, inverse
        return mapping

    def get_case_to_label_mapping(self) -> Dict[str, int]:
        """
        Given a list of datapoints, we create a mapping of label name to label id.
        :return:
        """
        mapping = {}
        for point in self.datapoints:
            mapping[point.case_name] = point.label
        return mapping

    def process_fold(self, fold: int) -> None:
        """
        Preprocesses a fold. This method indirectly triggers saving of metadata if necessary,
        writes data to proper folder, and will perform any other future preprocessing.
        :param fold: The fold that we are currently preprocessing.
        :return: Nothing.
        """
        print(f"Now starting with fold {fold}...")
        time.sleep(1)
        train_loader, val_loader = get_dataloaders_from_fold(self.dataset_name, fold,
                                                             preprocessed_data=False,
                                                             batch_size=1,
                                                             shuffle=False,
                                                             store_metadata=True,
                                                             preload=False
                                                             )
        # prep dirs
        os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}")
        os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/train")
        os.mkdir(f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/val")
        # start saving preprocessed stuff
        train_normalize_loader = train_loader.dataset[0][1].normalizer(train_loader, active=self.normalize)
        val_normalize_loader = val_loader.dataset[0][1] \
            .normalizer(val_loader, active=self.normalize, calculate_early=False)
        val_normalize_loader.sync(train_normalize_loader)
        if train_normalize_loader.mean is not None:
            mean_json = {
                "mean": train_normalize_loader.mean.tolist(),
                "std": train_normalize_loader.std.tolist()
            }
            write_json(mean_json, f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/mean_std.json")

        for _set in ['train', 'val']:
            for data, labels, points in \
                    tqdm(train_normalize_loader if _set == 'train' else
                         val_normalize_loader, desc=f"Preprocessing {_set} set"):
                point = points[0]
                writer = point.reader_writer
                data = data[0].float()  # Take 0 cause batched
                if data.shape[-1] == 3 and len(data.shape) == 3:
                    # move channel first
                    data = np.transpose(data, (2, 0, 1))
                writer.write(
                    data,
                    f"{PREPROCESSED_ROOT}/{self.dataset_name}/fold_{fold}/{_set}/{point.case_name}."
                    f"{point.extension if point.extension == 'nii.gz' else 'npy'}"
                )


def get_preprocessor_from_name(name: str) -> Type[Preprocessor]:
    from src.preprocessing.echo_preprocessor import CardiacEchoViewPreprocessor
    from src.preprocessing.nutrient_preprocessor import NutrientPreprocessor
    from src.preprocessing.imagenet_preprocessor import ImagenetPreprocessor
    return {
        ECHO: CardiacEchoViewPreprocessor,
        BASE: Preprocessor,
        NUTRIENT: NutrientPreprocessor,
        IMAGENET: ImagenetPreprocessor
    }[name]


@click.command()
@click.option('-folds', '-f', help='How many folds should be generated.', type=int)
@click.option('-processes', '-p', help='How many processes can be used.', type=int, default=8)
@click.option('--normalize', '--n', help='Should we compute and save normalized data.', type=bool, is_flag=True)
@click.option('-dataset_id', '-d', help='The dataset id to work on.', type=str)
@click.option('-preprocessor', help="Identifier of the preprocessor you want to use", default='base')  # echo
@click.option('-cardiac_data_root', help="The data root for cardiac data.", required=False)  # echo
@click.option('-cardiac_csv_path', help="The path to label csv for cardiac data.", required=False)  # echo
@click.option('-nutrient_data_root', help="The data root for nutrient data.", required=False)  # nutrient
@click.option('-imagenet_data_root', help="The data root for imagenet data from "+
              "https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data .", required=False)  # imagenet
def main(folds: int, processes: int, normalize: bool, dataset_id: str, preprocessor: str,
         cardiac_data_root: str, cardiac_csv_path: str, nutrient_data_root:str, imagenet_data_root: str):
    assert preprocessor in [BASE, ECHO, NUTRIENT, IMAGENET], f"Only {BASE}, {IMAGENET}, {ECHO} and {NUTRIENT} are supported preprocessors."
    assert not preprocessor == ECHO or (cardiac_data_root is not None and cardiac_csv_path is not None), \
        "You specified echo preprocessor which requires cardiac_data_root and cardiac_csv_path arguments."
    assert not preprocessor == NUTRIENT or (nutrient_data_root is not None), \
        "You specified nutrient preprocessor which requires nutrient_data_root argument."
    assert not preprocessor == IMAGENET or (imagenet_data_root is not None), \
        "You specified imagenet preprocessor which requires imagenet_data_root argument."
    preprocessor = get_preprocessor_from_name(preprocessor)
    preprocessor = preprocessor(
        dataset_id=dataset_id,
        normalize=normalize,
        folds=folds,
        processes=processes,
        data_root=cardiac_data_root,
        csv_path=cardiac_csv_path,
        nutrient_data_root=nutrient_data_root,
        imagenet_data_root=imagenet_data_root
    )
    preprocessor.process()
    print("Preprocessing completed!")


if __name__ == "__main__":
    main()
