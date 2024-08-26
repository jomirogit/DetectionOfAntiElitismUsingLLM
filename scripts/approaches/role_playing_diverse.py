#!/usr/bin/env python
# coding: utf-8

import pandas as pd

from transformers import pipeline
import torch

# Set the paths to enable importing the needed methods and classes
import sys
import os

src_path_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../PopBERT'))
sys.path.insert(0, src_path_1)
src_path_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path_2)

from src.bert.dataset import PBertDataset

from anti_elitism_model import BaseMVLabelStrategy
from common_methods import load_training_data
from common_methods import extract_assistant_text


# Define the roles and positions used for role playing
roles_positions = [
    {"role": "Kanzlerkandidat", "position": "konservativ"},
    {"role": "Finanzminister", "position": "konservativ"},
    {"role": "Gesundheitsminister", "position": "konservativ"},
    {"role": "Umweltminister", "position": "gruen"},
    {"role": "Aussenminister", "position": "liberal"},
    {"role": "Innenminister", "position": "liberal"},
    {"role": "Justizminister", "position": "sozialdemokratisch"},
    {"role": "Verteidigungsminister", "position": "sozialdemokratisch"},
    {"role": "Wirtschaftsminister", "position": "sozialdemokratisch"},
    {"role": "Bildungsminister", "position": "zentristisch"},
    {"role": "Arbeitsminister", "position": "zentristisch"},
    {"role": "Familienminister", "position": "zentristisch"},
    {"role": "Sozialminister", "position": "zentristisch"},
    {"role": "Energieminister", "position": "zentristisch"},
    {"role": "Landwirtschaftsminister", "position": "zentristisch"},
    {"role": "Parteivorsitzender", "position": "sozialdemokratisch"},
    {"role": "Generalsekretaer", "position": "konservativ"},
    {"role": "Praesident des Bundesverbandes", "position": "liberal"},
    {"role": "Bundespraesident", "position": "neutral"},
    {"role": "Verfassungsrichter", "position": "neutral"}
]
train_data = load_training_data()

generator = pipeline(model="LeoLM/leo-hessianai-13b-chat", device="cuda", torch_dtype=torch.float16, trust_remote_code=False)

generated_texts = []
generated_texts_df = pd.DataFrame(columns=["text", "id", "elite"])

for _ in range(150):
    # Generate anti-elitism statements for each role and position
    for role_position in roles_positions:
        role = role_position['role']
        position = role_position['position']

        # Generate a specific system message based on the role and the position
        system_message = f"""system
        Du bist ein {role} mit einer {position} politischen Einstellung und sprichst im deutschen Bundestag.
        """
        
        # Define the 'generation prompt'
        prompt = "Taetige eine anti-elitistische Aussage"
        
        # Define the prompt format
        prompt_format = "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Generate response
        response = generator(prompt_format.format(prompt=prompt, system_message=system_message), do_sample=True, top_p=0.95, max_length=2000)
        assistant_text = extract_assistant_text(response[0])
        
        if assistant_text:
            generated_texts.append(assistant_text)
            # Append each generated text to the DataFrame
            new_data = pd.DataFrame({
                "text": [assistant_text],
                "id": [train_data["id"].max() + 1 + len(generated_texts)], 
                "elite": [1]  # assuming all generated texts are labeled as anti-elitism
            })
            generated_texts_df = pd.concat([generated_texts_df, new_data], ignore_index=True)


# Save the generated texts to a file for qualitative analysis
with open(os.path.join(os.path.dirname(__file__), '../generated_data/qualitative_analysis/generated_texts_role_diverse.txt'), "w", encoding="utf-8") as f:
    os.makedirs(os.path.dirname(f.name), exist_ok=True)
    for text in generated_texts:
        f.write(text + "\n\n")
        f.write("\n" + "="*50 + "\n\n")


# Append the generated DataFrame to the existing DataFrame
expanded_train_data = pd.concat([train_data, generated_texts_df], ignore_index=True)

# Save the expanded DataFrame to a new CSV file for training a new model
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated_data/csv_training_data'))    
output_csv_file = os.path.join(output_dir, "expanded_train_data_role_diverse.csv")
expanded_train_data.to_csv(output_csv_file, index=False)


