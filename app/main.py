import logging
from input_model.input_model import InputModel
from central_model.central_model import CentralModel
from output_model.output_model import OutputModel
from evaluation_model_1 import EvaluationModel1
from evaluation_model_2 import EvaluationModel2

logging.basicConfig(filename='model_interactions.log', level=logging.INFO)

def main():
    input_model = InputModel()
    central_model = CentralModel()
    output_model = OutputModel()
    evaluation_model_1 = EvaluationModel1()
    evaluation_model_2 = EvaluationModel2()

    while True:
        user_input = input("Enter your message: ")
        processed_input = input_model.process(user_input)
        response = central_model.generate_response(processed_input)
        evaluation_1 = evaluation_model_1.evaluate(processed_input, response)
        logging.info(f'EvaluationModel1: {evaluation_1}')
        processed_response = output_model.process(response)
        evaluation_2 = evaluation_model_2.evaluate(response, processed_response)
        logging.info(f'EvaluationModel2: {evaluation_2}')
        print(processed_response)

if __name__ == "__main__":
    main()
