import pandas as pd

class DataHandler:
    def __init__(self, log_file='model_interactions.log'):
        self.log_file = log_file

    def parse_logs(self):
        # Open the log file and read the lines
        with open(self.log_file, 'r') as f:
            lines = f.readlines()

        # Parse the lines into a list of dictionaries
        data = []
        for line in lines:
            parts = line.strip().split(': ')
            data.append({
                'timestamp': parts[0],
                'model': parts[1],
                'message': ': '.join(parts[2:])
            })

        # Convert the list of dictionaries into a DataFrame
        df = pd.DataFrame(data)

        return df
