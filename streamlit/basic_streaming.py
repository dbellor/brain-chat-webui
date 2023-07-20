import logging
import streamlit as st
from input_model.input_model import InputModel
from central_model.central_model import CentralModel
from output_model.output_model import OutputModel
from app.fine_tuner import FineTuner
def main():
    input_model = InputModel()
    central_model = CentralModel()
    output_model = OutputModel()
    fine_tuner = FineTuner()

    if "messages" not in st.session_state:
        st.session_state["messages"] = ["How can I help you?"]

    logger = logging.getLogger(__name__)

    for msg in st.session_state.messages:
        logger.info(msg)

    if prompt := st.text_input("Enter your message"):
        st.session_state.messages.append(prompt)
        processed_input = input_model.process(prompt)
        try:
            response = central_model.generate_response(processed_input)
        except Exception as e:
            response = 'Error: ' + str(e)
        processed_response = output_model.process(response)
        st.session_state.messages.append(processed_response)
        logger.info(processed_response)

    if st.button("Train Evaluation Models"):
        fine_tuner.train()

if __name__ == "__main__":
    main()