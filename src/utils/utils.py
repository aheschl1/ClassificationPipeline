import json
from typing import Dict
import os


def write_json(data: Dict, path: str, create_folder: bool = False) -> None:
    if not os.path.exists('/'.join(path.split('/')[0:-1])):
        assert create_folder, 'Path does not exist, and you did not indicate create_folder.'
        os.makedirs(path)

    with open(path, 'w') as file:
        json.dump(data, file)
