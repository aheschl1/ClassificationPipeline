import os

class LogAnalyzer:
    def __init__(self, log_path: str):
        assert os.path.exists(log_path), "Log path does not exist"
        with open(log_path) as log_file:
            parse(log_file)