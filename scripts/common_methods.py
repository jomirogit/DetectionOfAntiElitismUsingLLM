import torch
import numpy as np

import sys
import os

# Define the missing paths to load the train data from Erhard et al.,2023
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../PopBERT'))
sys.path.insert(0, src_path)

from src.bert.dataset import PBertDataset
from src.bert.dataset.strategies import LabelStrategy

# Implement the label strategy applying the majority vote
class BaseMVLabelStrategy(LabelStrategy):
    labels = ["elite"]

    @staticmethod
    def vote(votes):
        votes = np.array(votes)
        label_dim = votes[:, 0]
        count_ones = np.count_nonzero(label_dim == 1)

        if count_ones > len(label_dim) / 2:
            return 1  # majority vote for anti-elitism
        else:
            return 0  # majority vote against anti-elitism
	
    @staticmethod
    def create_label(row):
        label = [0]
        if row["elite"]:
            label[0] = 1
        return tuple(label)

# Load the train data from Erhard et al.,2023
def load_training_data():
    train = PBertDataset.from_disk(
        path=os.path.join(src_path, "data/labeled_data/train.csv.zip"),
        label_strategy=BaseMVLabelStrategy(),
        exclude_coders=[],
    )
    train_data = train.df_labels[["id", "text", "vote"]]
    train_data["elite"] = train_data["vote"]

    train_data = train_data[["id", "text", "elite"]]
    true_label_data = train_data[train_data['elite'] == 1]

    return train_data

# Load the training data and return both the train_data as DataFrame and the true_label_data
def load_training_true_data():
    train = PBertDataset.from_disk(
        path=os.path.join(src_path, "data/labeled_data/train.csv.zip"),
        label_strategy=BaseMVLabelStrategy(),
        exclude_coders=[]
    )
    train_data = train.df_labels[["id", "text", "vote"]]
    train_data["elite"] = train_data["vote"]

    train_data = train_data[["id", "text", "elite"]]
    true_label_data = train_data[train_data['elite'] == 1]

    return train_data, true_label_data


# Extract the generated assistant text
def extract_assistant_text(response):
    text = response['generated_text']
    
    start = text.find('assistant\n')
    if start != -1:
        return text[start + len('assistant\n'):]
    return None



