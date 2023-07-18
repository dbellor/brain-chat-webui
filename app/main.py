from input_model.input_model import InputModel
from central_model.central_model import CentralModel
from output_model.output_model import OutputModel

def main():
    input_model = InputModel()
    central_model = CentralModel()
    output_model = OutputModel()

    while True:
        user_input = input("Enter your message: ")
        processed_input = input_model.process(user_input)
        response = central_model.generate_response(processed_input)
        processed_response = output_model.process(response)
        print(processed_response)

if __name__ == "__main__":
    main()
