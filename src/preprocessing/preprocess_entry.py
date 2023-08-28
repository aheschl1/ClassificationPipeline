from src.utils.utils import write_json
from src.utils.constants import *
from src.preprocessing.splitting import Splitter
import click


class Preprocessing:
    """
    This object should define all preprocessing steps
    """

    def __init__(self):
        pass


@click.command()
@click.option('--folds', '-f', help='How many folds should be generated.', type=int)
@click.option('--processes', '-p', help='How many processes can be used.', type=int, default=8)
@click.option('--normalize', '--n', help='Should we compute and save normalized data.', type=bool,
              default=True, is_flag=True)
@click.option('--dataset_id', '-d', help='The dataset id to work on.', type=str)
def main(folds: int, processes: int, normalize: bool, dataset_id: str):
    # This method should now create a Preprocessing object, which should have a bunch of static methods.
    # This method will now sequentially run these commands based on arguments.
    pass


if __name__ == "__main__":
    main()
