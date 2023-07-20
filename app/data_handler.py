import logging
import os
import pandas as pd
class DataHandler:
    def __init__(self, log_file):
        """
        Initialize the DataHandler object.

        Args:
            log_file (str): The name of the log file.
        """
        if not os.path.exists(log_file):
            raise FileNotFoundError(f"Log file {log_file} not found.")
        self.log_file = log_file
    def parse_logs(self):
        """
        Parse the log file and return a list of parsed logs.

        Returns:
            list: A list of dictionaries representing the parsed logs.
        """
        try:
            with open(self.log_file, 'r') as f:
                logs = [{'timestamp': parts[0], 'model': parts[1], 'message': ': '.join(parts[2:])} 
                        for line in f 
                        if len((parts := line.strip().split(': '))) >= 3]
        except FileNotFoundError:
            logging.error('Log file not found.')
            return None
        except (FileNotFoundError, IOError) as e:
            logging.exception('Error reading log file: %s', e)
            return None

        return logs

    def parse_logs_to_dataframe(self):
        """
        Parse the log file and return a pandas DataFrame of the parsed logs.

        Returns:
            pandas.DataFrame: A DataFrame representing the parsed logs.
        """
        logs = self.parse_logs()
        if logs is not None:
            df = pd.DataFrame(logs)
            return df
        else:
            return None

    def get_logs_dataframe(self):
        df = pd.DataFrame(self.parse_logs())
        return df