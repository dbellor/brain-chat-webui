import torch
import os
import logging
import streamlit as st
from input_model.input_model import InputModel
from central_model.central_model import CentralModel
from output_model.output_model import OutputModel
from app.fine_tuner import FineTuner

class ImprovedCode:
    def __init__(self):
        self.input_model = InputModel()
        self.central_model = CentralModel()
        self.output_model = OutputModel()
        self.fine_tuner = FineTuner()
        self.logger = logging.getLogger(__name__)

    def generate_response(self, input):
        # Convert the input to tensor for the model
        input_tensor = torch.tensor([ord(c) for c in input], dtype=torch.long)
        # Generate a response from the model
        try:
            response, _ = self.central_model.model(input_tensor)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                raise ValueError('Out of memory error')
            else:
                raise e
        # Convert the response tensor to string
        response_str = self.central_model.tokenizer.decode(response[0], skip_special_tokens=True)
        return response_str

    def process_message(self, prompt):
        processed_input = self.input_model.process(prompt)
        try:
            response = self.generate_response(processed_input)
        except Exception as e:
            response = 'Error: ' + str(e)
        processed_response = self.output_model.process(response)
        return processed_response

    def main(self):
        if not os.path.exists('messages.txt'):
            with open('messages.txt', 'w') as f:
                f.write('How can I help you?\n')
        with open('messages.txt', 'r') as f:
            messages = f.readlines()
            messages = [msg.strip() for msg in messages]
        for msg in messages:
            self.logger.info(msg)

        if prompt := st.text_input("Enter your message"):
            with open('messages.txt', 'a') as f:
                f.write(prompt + '\n')
            processed_response = self.process_message(prompt)
            with open('messages.txt', 'a') as f:
                f.write(processed_response + '\n')
            self.logger.info(processed_response)

        if st.button("Train Evaluation Models"):
            self.fine_tuner.train()

if __name__ == "__main__":
    main_obj = ImprovedCode()
    main_obj.main()
    if prompt := st.text_input("Enter your message", key="input1"):
        st.session_state.messages.append(prompt)
        processed_response = main_obj.process_message(prompt)
        st.session_state.messages.append(processed_response)
        main_obj.logger.info(processed_response)

    if st.button("Train Evaluation Models", key="train2"):
        main_obj.fine_tuner.train()