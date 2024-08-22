
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    def __init__(self, train_data, model_name="deepset/gbert-base", batch_size=8, epochs=3, lr=0.000009, weight_decay=0.01):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.MODEL_NAME = model_name
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.lr = lr
        self.weight_decay = weight_decay

        # Set the training data
        self.train_data = train_data

        # Set the test data
        test = PBertDataset.from_disk(
            path=os.path.join(src_path, "data/labeled_data/test.csv.zip"),
            label_strategy=BaseMVLabelStrategy(),
            exclude_coders=[],
        )
        test_data = test.df_labels[["id", "text", "vote"]]
        test_data["elite"] = test_data["vote"]

        self.test_data = test_data[["id", "text", "elite"]]
        


        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = BertForSequenceClassification.from_pretrained(self.MODEL_NAME, num_labels=2).to(self.DEVICE)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, correct_bias=False)
        self.num_training_steps = self.EPOCHS * (len(self.train_data) // self.BATCH_SIZE)
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=20,
            eta_min=self.lr / 10,
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

    def evaluate_model(self, approach):
        self.model.eval()
        y_true = []
        y_pred = []

        # Extract all texts from the test data
        all_texts = self.test_data["text"].tolist()

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

        # Create a DataFrame to store the texts, actual labels, and predicted labels
        results_df = pd.DataFrame({
            "predicted_label": y_pred,
            "actual_label": y_true,
            "text": all_texts
        })

        # Save the DataFrame to a CSV file or print it
        results_df.to_csv("test_results" + approach + ".csv", index=False)
        print(results_df)

    def save_model(self, output_dir="results"):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir + "/anti_elitism_model")
        self.tokenizer.save_pretrained(output_dir + "/anti_elitism_tokenizer")
