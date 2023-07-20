import logging
import os
import sys
import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
class FineTuner:
    def __init__(self, model_name='gpt2', device='cuda'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.device = device

    def train(self, df, epochs=1, learning_rate=1e-4):
        # Convert the DataFrame into a Dataset
        dataset = DataFrameDataset(df, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Set up the optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        num_training_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

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
                scheduler.step()
                optimizer.zero_grad()

    def evaluate(self):
        # Set the model to evaluation mode
        self.model.eval()

        # Save the trained models
        with torch.no_grad():
            self.model.save_pretrained('evaluation_model_1')
            self.model.save_pretrained('evaluation_model_2')

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

# Check if the file exists
if not os.path.exists('model_interactions.csv'):
    # Create a default file with necessary columns
    with open('model_interactions.csv', 'w') as f:
        f.write('timestamp,model,message\n')

# Load the data
try:
    df = pd.read_csv('model_interactions.csv')
except Exception as e:
    logging.error(f'An error occurred while loading the data: {e}')
    raise e

# Split the data into a training set and a test set
train_df, test_df = train_test_split(df, test_size=0.2)

# Initialize the fine tuners
fine_tuner1 = FineTuner()
fine_tuner2 = FineTuner()

# Train the evaluation models
fine_tuner1.train(train_df)
fine_tuner2.train(train_df)

# Evaluate and save the trained models
fine_tuner1.evaluate()
fine_tuner2.evaluate()

# Split the data into a training set and a test set
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize the fine tuners
fine_tuner1 = FineTuner()
fine_tuner2 = FineTuner()

# Train the evaluation models
fine_tuner1.train(train_df)
fine_tuner2.train(train_df)

# Evaluate and save the trained models
fine_tuner1.evaluate()
fine_tuner2.evaluate()
# Initialize the fine tuners
fine_tuner1 = FineTuner()
fine_tuner2 = FineTuner()

# Train the evaluation models
fine_tuner1.train(train_df)
fine_tuner2.train(train_df)

# Evaluate and save the trained models
fine_tuner1.evaluate()
fine_tuner2.evaluate()        # For example, you might want to add some default data to it
with open('model_interactions.csv', 'w') as f:
    f.write('timestamp,model,message\n')

# Load the data
try:
    df = load_data('model_interactions.csv')
except Exception as e:
    logging.error(f'An error occurred while loading the data: {e}')
    sys.exit(1)


# Split the data into a training set and a test set
train_df, test_df = train_test_split(df, test_size=0.2)

# Initialize the fine tuners
fine_tuner1 = FineTuner()
fine_tuner2 = FineTuner()

# Train the evaluation models
fine_tuner1.train(train_df)
fine_tuner2.train(train_df)

# Evaluate and save the trained models
try:
    fine_tuner1.evaluate()
except Exception as e:
    logging.error(f'An error occurred while evaluating fine_tuner1: {e}')
try:
    fine_tuner2.evaluate()
except Exception as e:
    logging.error(f'An error occurred while evaluating fine_tuner2: {e}')