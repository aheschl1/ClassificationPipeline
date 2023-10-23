import sys, os
sys.path.append("/home/andrew.heschl/Documents/ClassificationPipeline")
sys.path.append("/home/andrewheschl/PycharmProjects/classification_pipeline")
sys.path.append("/home/student/andrew/Documents/ClassificationPipeline")
import click
from src.utils.constants import *
from src.utils.utils import get_dataset_name_from_id
import glob

@click.command()
@click.option('-fold', '-f', help='Which fold to train.', type=int, required=True)
@click.option('-dataset_id', '-d', help='The dataset id to train.', type=str, required=True)
def main(fold, dataset_id):
    result_root = f"{RESULTS_ROOT}/{get_dataset_name_from_id(dataset_id)}/fold_{fold}"
    trial_folders = [folder for folder in glob.glob(f"{result_root}/*") if os.path.isdir(folder)]
    print(trial_folders)

if __name__ == "__main__":
    main()