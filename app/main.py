import json
import logging
import os
from datetime import datetime
import csv
def main(input_model, central_model, output_model, evaluation_model_1, evaluation_model_2):
    user_id = input("Enter your user ID: ")
    while True:
        user_input = input("Enter your message: ")
        if not user_input:
            continue
        # Log the user input with the user ID
        with open('user.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([user_id, user_input])
        processed_input = input_model.process(user_input)
        try:
            response = central_model.generate_response(processed_input)
        except ValueError as e:
            logging.error(f'Error generating response: {e}')
            response = 'Error generating response'
        evaluation_1 = evaluation_model_1.evaluate(processed_input, response)
        logging.info({'event': 'EvaluationModel1', 'timestamp': datetime.now(), 'evaluation': evaluation_1})
        processed_response = output_model.process(response)
        evaluation_2 = evaluation_model_2.evaluate(response, processed_response)
        logging.info({'event': 'EvaluationModel2', 'timestamp': datetime.now(), 'evaluation': evaluation_2})
        print(processed_response)

if __name__ == "__main__":
    try:
        with open('logging_config.json', 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f'Error reading logging_config.json file: {e}')
        config = {}
    logging.basicConfig(filename=config.get('filename', 'log.txt'), level=config.get('level', 'INFO'))

    # Create a CSV file to store user data if it doesn't exist
    if not os.path.exists('user.csv'):
        with open('user.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['user_id', 'user_input'])

    input_model = None
    central_model = None
    output_model = None
    evaluation_model_1 = None
    evaluation_model_2 = None

    if input_model is None or central_model is None or output_model is None or evaluation_model_1 is None or evaluation_model_2 is None:
        raise ValueError('One or more models are None')

    main(input_model, central_model, output_model, evaluation_model_1, evaluation_model_2)
    central_model = os.environ.get('CENTRAL_MODEL')
    output_model = os.environ.get('OUTPUT_MODEL')
    evaluation_model_1 = os.environ.get('EVALUATION_MODEL_1')
    evaluation_model_2 = os.environ.get('EVALUATION_MODEL_2')

    if input_model is None or central_model is None or output_model is None or evaluation_model_1 is None or evaluation_model_2 is None:
        raise ValueError('One or more models are None')

    main(input_model, central_model, output_model, evaluation_model_1, evaluation_model_2)
    if input_model is None or central_model is None or output_model is None or evaluation_model_1 is None or evaluation_model_2 is None:
        raise ValueError('One or more models are None')

    main(input_model, central_model, output_model, evaluation_model_1, evaluation_model_2)

    if input_model is None or central_model is None or output_model is None or evaluation_model_1 is None or evaluation_model_2 is None:
        raise ValueError('One or more models are None')

    main(input_model, central_model, output_model, evaluation_model_1, evaluation_model_2)
