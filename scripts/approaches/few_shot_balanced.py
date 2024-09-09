#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import sys
import os

src_path_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../PopBERT'))
sys.path.insert(0, src_path_1)
src_path_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path_2)

from src.bert.dataset import PBertDataset
from anti_elitism_model import BaseMVLabelStrategy
from common_methods import load_training_true_data
from common_methods import load_training_false_data
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

# Check similarity score based on SBERT
def is_text_diverse(new_text, few_shot_texts, sbert_model, threshold=0.8):
    new_embedding = sbert_model.encode(new_text, convert_to_tensor=True)
    few_shot_embeddings = sbert_model.encode(few_shot_texts, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(new_embedding, few_shot_embeddings)
    
    if torch.max(cosine_scores) > threshold:
        return False
    return True

# Generate texts
def generate_texts_true(generator, true_label_data, train_data, sbert_model, threshold=0.8, n_texts=450):
    generated_texts_df = pd.DataFrame(columns=["text", "id", "elite"])
    prompt_format = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    few_shot_log_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated_data/qualitative_analysis/few_shot_log.txt'))

    i = 0
    while len(generated_texts_df) < n_texts:
        few_shot_examples = true_label_data.sample(n=4)
        formatted_examples = "\n".join(f"text: {example['text']}" for _, example in few_shot_examples.iterrows())
        few_shot_texts = few_shot_examples['text'].tolist()
        
        prompt = f"""
        Hier sind 4 Beispiele fuer Anti-Elitismus:
        {formatted_examples}\n
        Taetige genau eine neue anti-elitistische Aussage zu einem politischen Thema.
        """
        
        generated_response = generator(prompt_format.format(prompt=prompt), do_sample=True, top_p=0.95, max_length=3000)
        assistant_text = extract_assistant_text(generated_response[0])

        with open(few_shot_log_file, "a", encoding="utf-8") as log:
            os.makedirs(os.path.dirname(log.name), exist_ok=True)
            log.write("Few-Shot Inputs:\n")
            for text in few_shot_texts:
                log.write(f"- {text}\n")
            log.write(f"\nGenerated Text: {assistant_text}\n")

            if assistant_text and is_text_diverse(assistant_text, few_shot_texts, sbert_model, threshold):
                new_data = pd.DataFrame({
                    "text": [assistant_text],
                    "id": [train_data["id"].max() + 1 + i],
                    "elite": [1] 
                })
                generated_texts_df = pd.concat([generated_texts_df, new_data], ignore_index=True)
                log.write("Result: ACCEPTED\n")
                i = i + 1
            else:
                log.write("Result: REJECTED (too similar)\n")
            log.write("\n" + "="*50 + "\n\n")
            
    return generated_texts_df

def generate_texts_false(generator, false_label_data, train_data, sbert_model, threshold=0.8, n_texts=2550):
    generated_texts_df = pd.DataFrame(columns=["text", "id", "elite"])
    prompt_format = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    few_shot_log_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated_data/qualitative_analysis/few_shot_log.txt'))

    i = 0
    while len(generated_texts_df) < n_texts:
        few_shot_examples = false_label_data.sample(n=4)
        formatted_examples = "\n".join(f"text: {example['text']}" for _, example in few_shot_examples.iterrows())
        few_shot_texts = few_shot_examples['text'].tolist()
        
        prompt = f"""
        Hier sind 4 Beispiele fuer politische Aussagen, die keine anti-elitische Aussagen sind:
        {formatted_examples}\n
        Taetige genau eine neue politische Aussage zu einem Thema, die keine anti-elitistische Aussage ist.
        """
        
        generated_response = generator(prompt_format.format(prompt=prompt), do_sample=True, top_p=0.95, max_length=3000)
        assistant_text = extract_assistant_text(generated_response[0])

        with open(few_shot_log_file, "a", encoding="utf-8") as log:
            os.makedirs(os.path.dirname(log.name), exist_ok=True)
            log.write("Few-Shot Inputs:\n")
            for text in few_shot_texts:
                log.write(f"- {text}\n")
            log.write(f"\nGenerated Text: {assistant_text}\n")

            if assistant_text and is_text_diverse(assistant_text, few_shot_texts, sbert_model, threshold):
                new_data = pd.DataFrame({
                    "text": [assistant_text],
                    "id": [train_data["id"].max() + 1 + i],
                    "elite": [0]  
                })
                generated_texts_df = pd.concat([generated_texts_df, new_data], ignore_index=True)
                log.write("Result: ACCEPTED\n")
                i = i + 1
            else:
                log.write("Result: REJECTED (too similar)\n")
            log.write("\n" + "="*50 + "\n\n")
            
    return generated_texts_df

def save_generated_texts(generated_texts_df):
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated_data/csv_training_data'))
    output_csv_file = os.path.join(output_dir, "expanded_train_data_few_shot.csv")
    generated_texts_df.to_csv(output_csv_file, index=False)

def main():
    train_data, true_label_data = load_training_true_data()
    false_label_data = load_training_false_data()
    generator = initialize_generator()
    sbert_model = initialize_sbert()
    
    generated_texts_df_true = generate_texts_true(generator, true_label_data, train_data, sbert_model)
    generated_texts_df_false = generate_texts_false(generator, false_label_data, train_data, sbert_model)
    generated_texts_df = pd.concat([generated_texts_df_true, generated_texts_df_false], ignore_index=True)
    
    with open(os.path.join(os.path.dirname(__file__), '../generated_data/qualitative_analysis/generated_texts_few_shot.txt'), "w", encoding="utf-8") as f:
        os.makedirs(os.path.dirname(f.name), exist_ok=True)
        for _, row in generated_texts_df.iterrows():
            f.write(row["text"] + "\n\n")
            f.write("\n" + "="*50 + "\n\n")
   
    expanded_train_data = pd.concat([train_data, generated_texts_df], ignore_index=True)
    save_generated_texts(expanded_train_data)

if __name__ == "__main__":
    main()
