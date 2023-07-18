from input_model.input_model import InputModel
from central_model.central_model import CentralModel
from output_model.output_model import OutputModel
import streamlit as st

input_model = InputModel()
central_model = CentralModel()
output_model = OutputModel()

if "messages" not in st.session_state:
    st.session_state["messages"] = ["How can I help you?"]

for msg in st.session_state.messages:
    st.text(msg)

if prompt := st.text_input("Enter your message"):
    st.session_state.messages.append(prompt)
    processed_input = input_model.process(prompt)
    response = central_model.generate_response(processed_input)
    processed_response = output_model.process(response)
    st.session_state.messages.append(processed_response)

