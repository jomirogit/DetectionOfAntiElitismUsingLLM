#!/usr/bin/env python
# coding: utf-8

import pandas as pd

import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Set the paths to enable importing the implemented classes and methods
import sys
import os

src_path_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../PopBERT'))
sys.path.insert(0, src_path_1)
src_path_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path_2)

from src.bert.dataset import PBertDataset

from anti_elitism_model import BaseMVLabelStrategy
from common_methods import load_training_true_data
from common_methods import extract_assistant_text

# Initialize the model
def initialize_generator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = pipeline(model="LeoLM/leo-hessianai-13b-chat", device=device, torch_dtype=torch.float16, trust_remote_code=False)
    return generator

# Initialize SBERT
def initialize_sbert():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model

# Proof the similarity score based on SBERT
def is_text_diverse(new_text, few_shot_texts, sbert_model, threshold=0.8):
    
    new_embedding = sbert_model.encode(new_text, convert_to_tensor=True)
    few_shot_embeddings = sbert_model.encode(few_shot_texts, convert_to_tensor=True)
    
    cosine_scores = util.pytorch_cos_sim(new_embedding, few_shot_embeddings)
    
    # If the maximum cosine score is above the threshold, the text is not diverse enough
    if torch.max(cosine_scores) > threshold:
        return False
    return True

# generate the texts
def generate_texts(generator, true_label_data, train_data, sbert_model, threshold=0.8, n_texts=3000):
    generated_texts_df = pd.DataFrame(columns=["text", "id", "elite"])
    
    prompt_format = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    for i in range(n_texts):
        # Provide four training instances as few-shot examples
        few_shot_examples = true_label_data.sample(n=4)
        formatted_examples = "\n".join(f"text: {example['text']}" for _, example in few_shot_examples.iterrows())
        few_shot_texts = few_shot_examples['text'].tolist()
        
        prompt = f"""
        Hier sind 4 Beispiele fuer Anti-Elitismus:
        {formatted_examples}\n
        Taetige genau eine neue anti-elitistische Aussage zu einem politischen Thema.
        """
        
        generated_response = generator(prompt_format.format(prompt=prompt), do_sample=True, top_p=0.95, max_length=8192)
        assistant_text = extract_assistant_text(generated_response[0])
        
        if assistant_text and is_text_diverse(assistant_text, few_shot_texts, sbert_model, threshold):
            new_data = pd.DataFrame({
                "text": [assistant_text],
                "id": [train_data["id"].max() + 1 + i],
                "elite": [1] 
            })
            generated_texts_df = pd.concat([generated_texts_df, new_data], ignore_index=True)
    
    return generated_texts_df


def save_generated_texts(generated_texts_df):
    generated_texts_df.to_csv("expanded_train_data_few_shot2.csv", index=False)
    print("3000 texts successfully generated.")

def main():
    
    train_data, true_label_data = load_training_true_data()
    generator = initialize_generator()
    sbert_model = initialize_sbert()
    generated_texts_df = generate_texts(generator, true_label_data, train_data, sbert_model)
    save_generated_texts(generated_texts_df)

if __name__ == "__main__":
    main()
