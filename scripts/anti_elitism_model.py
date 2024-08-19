
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from sklearn.metrics import classification_report
import pandas as pd

import sys
import os

# Define the missing paths to load the train and test data from PopBERT
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../PopBERT'))
sys.path.insert(0, src_path)

from src.bert.dataset import PBertDataset
from common_methods import BaseMVLabelStrategy

class AntiElitismTrainer:
    def __init__(self, train_data, model_name="deepset/bert-large", batch_size=8, epochs=3, lr=2e-5):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.MODEL_NAME = model_name
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.lr = lr

        # Set the training data
        self.train_data = train_data

        # Set the test data
        test = PBertDataset.from_disk(
            path=self.test_csv_path,
            label_strategy=BaseMVLabelStrategy(),
            exclude_coders=[],
        )
        test_data = test.df_labels[["id", "text", "vote"]]
        test_data["elite"] = test_data["vote"]

        test_data = test_data[["id", "text", "elite"]]
        


        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = BertForSequenceClassification.from_pretrained(self.MODEL_NAME, num_labels=2).to(self.DEVICE)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, correct_bias=False)
        self.num_training_steps = self.EPOCHS * (len(self.train_data) // self.BATCH_SIZE)
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps,
        )

    def tokenize_data(self, data):
        texts = data["text"].tolist()
        labels = data["elite"].tolist()

        # Tokenize and apply padding
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

        # Convert labels to tensor
        labels = torch.tensor(labels)

        return encodings, labels

    def train_model(self):
        self.model.train()
        for epoch in range(self.EPOCHS):
            for i in range(0, len(self.train_data), self.BATCH_SIZE):
                batch_data = self.train_data.iloc[i:i + self.BATCH_SIZE]
                encodings, labels = self.tokenize_data(batch_data)
                input_ids = encodings['input_ids'].to(self.DEVICE)
                attention_mask = encodings['attention_mask'].to(self.DEVICE)
                labels = labels.to(self.DEVICE)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = torch.nn.CrossEntropyLoss()(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

            print(f'Epoch {epoch + 1} completed. Loss: {loss.item()}')

    def evaluate_model(self):
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for i in range(0, len(self.test_data), self.BATCH_SIZE):
                batch_data = self.test_data.iloc[i:i + self.BATCH_SIZE]
                encodings, labels = self.tokenize_data(batch_data)
                input_ids = encodings['input_ids'].to(self.DEVICE)
                attention_mask = encodings['attention_mask'].to(self.DEVICE)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)

                y_true.extend(labels.tolist())
                y_pred.extend(preds.cpu().tolist())

        print(
            classification_report(
                y_true,
                y_pred,
                target_names=["not anti-elitism", "anti-elitism"],
                zero_division=0,
            )
        )

    def save_model(self, output_dir="results"):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir + "anti_elitism_model")
        self.tokenizer.save_pretrained(output_dir + "anti_elitism_tokenizer")
