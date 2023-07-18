from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split

class FineTuner:
    def __init__(self, model_name='gpt2', device='cuda'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.device = device

    def train(self, df, epochs=1, learning_rate=1e-4):
        # Convert the DataFrame into a Dataset
        dataset = DataFrameDataset(df, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # Set up the optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Train the model
        self.model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs, labels=labels)
                loss = outputs.loss

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

class DataFrameDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = self.tokenizer.encode(row['message'], return_tensors='pt')
        labels = inputs.clone()
        return inputs, labels

# Load the data
df = pd.read_csv('model_interactions.csv')

# Split the data into a training set and a test set
train_df, test_df = train_test_split(df, test_size=0.2)

# Initialize the fine tuners
fine_tuner1 = FineTuner()
fine_tuner2 = FineTuner()

# Train the evaluation models
fine_tuner1.train(train_df)
fine_tuner2.train(train_df)

# Save the trained models
fine_tuner1.model.save_pretrained('evaluation_model_1')
fine_tuner2.model.save_pretrained('evaluation_model_2')
