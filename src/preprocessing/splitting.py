from typing import List, Dict

from sklearn.model_selection import train_test_split

from src.dataloading.datapoint import Datapoint


class Splitter:
    def __init__(self, data: List[Datapoint], folds: int) -> None:
        assert folds > 0, 'Folds must be > 0.'
        self.data = data
        self.folds = folds

    def get_split_map(self) -> Dict[int, Dict[str, List[str]]]:
        results = {}
        for i in range(self.folds):
            xtrain, xtest = train_test_split(self.data, random_state=i)
            results[i] = {
                'train': [x.case_name for x in xtrain],
                'val': [x.case_name for x in xtest]
            }
        return results


if __name__ == "__main__":
    dummy_data = [Datapoint('faafas', 2, case_name='fds') for _ in range(10)]
    splitter = Splitter(dummy_data, 3)
    print(splitter.get_split_map())
