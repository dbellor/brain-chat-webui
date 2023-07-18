## BrainChat App

This is a chat app that uses GPT models to generate responses to user input.

## How It Works

The BrainChat app is designed around the principles of human brain function. It consists of three key components inspired by different aspects of brain processing:

Perception: The "Perception" section corresponds to input processing, similar to how the brain receives and interprets sensory information. In the BrainChat app, the Perception component involves the input model that processes user input, including text, images, or other forms of data.

Cognition: The "Cognition" section represents the central processing and generation of responses, mirroring the brain's cognitive abilities. In the BrainChat app, the Cognition component includes the central model, which leverages GPT models to generate appropriate responses based on the processed input.

Expression: The "Expression" section focuses on presenting and communicating the output in a meaningful way, just as the brain expresses thoughts and actions. In the BrainChat app, the Expression component involves the output model that processes the generated responses and presents them to the user in a clear and understandable format.

## Directory Structure
```
chat_app
│   README.md - This file.
│   requirements.txt - Lists the Python packages that the project depends on.
│
└───app
│   │   main.py - The main Python script that runs the chat app.
│   │
│   └───models
│       │   pretrained_model.pth - Contains the pretrained weights for the GPT model.
│   
└───streamlit
    │   basic_streaming.py - Contains the Streamlit app that interfaces with the GPT model.
│
└───input_model
    │   input_model.py - Contains the code for the input model, which processes user input.
│
└───central_model
    │   central_model.py - Contains the code for the central model, which generates responses.
│
└───output_model
    │   output_model.py - Contains the code for the output model, which processes the responses.
```
## Installation

1. Clone this repository.
2. Install the required Python packages: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run streamlit/basic_streaming.py`

## Usage

Enter your message in the text input field and press Enter. The app will display a response from the GPT model.