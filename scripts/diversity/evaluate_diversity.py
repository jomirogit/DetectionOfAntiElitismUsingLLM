#!/usr/bin/env python
# coding: utf-8

import sys
import os

src_path_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../PopBERT'))
sys.path.insert(0, src_path_1)
src_path_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path_2)

from src.bert.dataset import PBertDataset

from anti_elitism_model import BaseMVLabelStrategy
from chamfer_remote_evaluation import SimilarityScore
from common_methods import load_training_data

import pandas as pd



if __name__ == "__main__":
    
    print("Baseline: Diversity of training data set without additional training examples")

    train_data, true_labeled_data = load_training_data()
    train_data.to_csv("train.csv", index=False)

    analyzer = SimilarityScore("train.csv")
    analyzer.analyze()