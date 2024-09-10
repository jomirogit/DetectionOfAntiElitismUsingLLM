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
    train_data = load_training_data()
    train_data.to_csv("train_baseline.csv", index=False)

    analyzer_baseline = SimilarityScore("train_baseline.csv")
    analyzer_baseline.analyze()

    print("\n" + "="*50 + "\n")

    print("Few Shot Data")
    analyzer_few_shot = SimilarityScore("../generated_data/csv_training_data/expanded_train_data_few_shot.csv")
    analyzer_few_shot.analyze()

    print("\n" + "="*50 + "\n")
    
    print("Similarity Scores for Multi-Step Prompting Data:")
    analyzer_cot = SimilarityScore("../generated_data/csv_training_data/expanded_train_data_msp.csv")
    analyzer_cot.analyze()
    
    print("\n" + "="*50 + "\n")
    
    print("Similarity Scores for Role Playing: Basic Data:")
    analyzer_role_basic = SimilarityScore("../generated_data/csv_training_data/expanded_train_data_role_basic.csv")
    analyzer_role_basic.analyze()
    
    print("\n" + "="*50 + "\n")
    
    print("Similarity Scores for Role Playing: Diverse Roles Data:")
    analyzer_role_diverse = SimilarityScore("../generated_data/csv_training_data/expanded_train_data_role_diverse.csv")
    analyzer_role_diverse.analyze()
    
    print("\n" + "="*50 + "\n")
    
    print("Similarity Scores for Topic Approach")
    analyzer_role_diverse = SimilarityScore("../generated_data/csv_training_data/expanded_train_data_role_topic.csv")
    analyzer_role_diverse.analyze()