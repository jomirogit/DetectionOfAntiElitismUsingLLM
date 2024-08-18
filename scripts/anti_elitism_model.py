import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from sklearn.metrics import classification_report

import sys
import os

# Define the missing paths to load the train and test data from PopBERT
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../PopBERT'))
sys.path.insert(0, src_path)

from src.bert.dataset import PBertDataset
from common_methods import BaseMVLabelStrategy

class AntiElitismTrainer:
    def __init__(self, train_set=None, train_csv_path=None, model_name="bert-base-german-cased", batch_size=8, epochs=3, lr=2e-5):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.MODEL_NAME = model_name
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.lr = lr
        self.test_csv_path = os.path.abspath(os.path.join(src_path, 'data/labeled_data/test.csv.zip'))

        # Load train data
        if train_set is not None:
            self.train = train_set
        elif train_csv_path is not None:
            self.train = PBertDataset.from_disk(
                path=train_csv_path,
                label_strategy=BaseMVLabelStrategy(),
                exclude_coders=[],
            )
        else:
            raise ValueError("Either 'train_set' or 'train_csv_path' must be provided.")

        # Load test data
        self.test = PBertDataset.from_disk(
            path=self.test_csv_path,
            label_strategy=BaseMVLabelStrategy(),
            exclude_coders=[],
        )

        # Initialize tokenizer and data loaders
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)
        self.collate_fn = self.train.create_collate_fn(self.tokenizer)
        self.train_loader = DataLoader(self.train, batch_size=self.BATCH_SIZE, shuffle=True, collate_fn=self.collate_fn)
        self.test_loader = DataLoader(self.test, batch_size=64, shuffle=False, collate_fn=self.collate_fn)

        # Initialize the model
        self.model = BertForSequenceClassification.from_pretrained(self.MODEL_NAME, num_labels=2).to(self.DEVICE)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, correct_bias=False)
        self.num_training_steps = self.EPOCHS * len(self.train_loader)
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps,
        )

    def train_model(self):
        for epoch in range(self.EPOCHS):
            self.model.train()
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                input_ids = batch['encodings']['input_ids'].to(self.DEVICE)
                attention_mask = batch['encodings']['attention_mask'].to(self.DEVICE)
                labels = batch['labels'].to(self.DEVICE).long()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = torch.nn.CrossEntropyLoss()(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

            print(f'Epoch {epoch + 1} completed.')

    def evaluate_model(self):
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.inference_mode():
            for batch in self.test_loader:
                input_ids = batch['encodings']['input_ids'].to(self.DEVICE)
                attention_mask = batch['encodings']['attention_mask'].to(self.DEVICE)
                labels = batch['labels'].to(self.DEVICE)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

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
        self.model.save_pretrained(output_dir + "/anti_elitism_model")
        self.tokenizer.save_pretrained(output_dir + "/anti_elitism_tokenizer")