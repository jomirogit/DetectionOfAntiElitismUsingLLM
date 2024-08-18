#!/usr/bin/env python
# coding: utf-8

from transformers import pipeline
import torch

import pandas as pd

import sys
import os

src_path_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../PopBERT'))
sys.path.insert(0, src_path_1)
src_path_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path_2)

from src.bert.dataset import PBertDataset

from anti_elitism_model import BaseMVLabelStrategy
from anti_elitism_model import load_training_data
from anti_elitism_model import extract_assistant_text()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Get the train data provided by Erhard et al.,2023
train_data, true_label_data = load_training_data()


# system message
system_message = ""

# Providing no system message, only relating to the topics as input
prompt_format = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

# list of the different topics
topics = [
    "Umweltschutz und Klimapolitik",
    "Gesundheitswesen und Krankenversicherung",
    "Wohnungsmarkt und sozialer Wohnungsbau",
    "Arbeitsmarkt und Arbeitsrecht",
    "Bildungspolitik und Schulsystem",
    "Migration und Integrationspolitik",
    "Medienlandschaft und Pressefreiheit",
    "Digitalisierung und Datenschutz",
    "Rentenpolitik und Altersversorgung",
    "Handelspolitik und internationale Abkommen",
    "Sozialpolitik und Sozialhilfe",
    "Steuerpolitik und Steuerreformen",
    "Infrastrukturprojekte und Verkehrsplanung",
    "Aussenpolitik und internationale Beziehungen",
    "Justizsystem und Rechtsreformen",
    "Gleichstellungspolitik und Genderfragen",
    "Energiepolitik und erneuerbare Energien",
    "Landwirtschaft und Ernährungssicherheit",
    "Veteranenangelegenheiten und Militaerpolitik",
    "Kulturpolitik und kulturelle Foerderung"
]

# Initialization of the generator
generator = pipeline(model="LeoLM/leo-hessianai-13b-chat", device="cuda", torch_dtype=torch.float16, trust_remote_code=False)

generated_texts_df = pd.DataFrame(columns=["text", "id", "elite"])
generated_texts = []


# Durchlaufen der Themenliste und Generieren der entsprechenden Aussagen
# Loop to generate texts
for _ in range(1000):
    for topic in topics:
        
        # Create a prompt for text generation
        prompt = f"Taetige eine anti-elitistische Aussage zu {topic}."
        
        # Generate a response
        response = generator(
            prompt_format.format(system_message=system_message, prompt=prompt),
            do_sample=True, top_p=0.95, max_length=2000
        )
        
        # Extract the generated text
        assistant_text = extract_assistant_text(response[0])

        if assistant_text:
            generated_texts.append(assistant_text)

            # Create a DataFrame with the generated text
            new_data = pd.DataFrame({
                "text": [assistant_text],
                "id": [train_data["id"].max() + 1 + len(generated_texts)],
                "elite": [1]  # Assuming all generated texts are labeled as anti-elitism
            })
            
            # Append the new text to the main DataFrame
            generated_texts_df = pd.concat([generated_texts_df, new_data], ignore_index=True)

# Append the generated DataFrame to the existing DataFrame
expanded_train_data = pd.concat([train_data, generated_texts_df], ignore_index=True)

# Save the expanded DataFrame to a new CSV file for training a new model
expanded_train_data.to_csv("expanded_train_data_role_topic.csv", index=False)


# Save the generated texts to a file for qualitative analysis
with open("generated_texts_role_topic.txt", "w", encoding="utf-8") as f:
    for text in generated_texts:
        f.write(text + "\n\n")
