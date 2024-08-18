#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist

import os

class SimilarityScore:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.train_data = self.load_data()
        self.text_embeddings = self.vectorize_text()

    def load_data(self):
        
        df = pd.read_csv(self.csv_path)
                
        return df

    def vectorize_text(self):
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(self.train_data['text']).toarray()

    def calculate_remote_clique_score(self):
        pairwise_distances = cdist(self.text_embeddings, self.text_embeddings, metric='euclidean')
        mean_distances = np.mean(pairwise_distances, axis=1)
        return np.mean(mean_distances)

    def calculate_chamfer_distance_score(self):
        pairwise_distances = cdist(self.text_embeddings, self.text_embeddings, metric='euclidean')
        np.fill_diagonal(pairwise_distances, np.inf)
        min_distances = np.min(pairwise_distances, axis=1)
        return np.mean(min_distances)

    def analyze(self):
        remote_clique_score = self.calculate_remote_clique_score()
        chamfer_distance_score = self.calculate_chamfer_distance_score()
        
        print(f"Remote Clique Score: {remote_clique_score:.4f}")
        print(f"Chamfer Distance Score: {chamfer_distance_score:.4f}")

