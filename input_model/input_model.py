from app.model import GPT, GPTConfig
import torch

class InputModel:
    def __init__(self):
        mconf = GPTConfig(vocab_size=256, block_size=128, n_layer=8, n_head=8, n_embd=512)
        self.model = GPT(mconf)

    def process(self, input):
        # Convert the input to tensor for the model
        input_tensor = torch.tensor([ord(c) for c in input], dtype=torch.long)
        # Generate a response from the model
        response, _ = self.model(input_tensor)
        # Convert the response tensor to string
        response_str = "".join(chr(i) for i in response.tolist())
        return response_str