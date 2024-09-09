#!/usr/bin/env python
# coding: utf-8

from transformers import pipeline
import torch

import pandas as pd

# Set the paths to enaable importing the implemented classes and methods
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_data, true_label_data = load_training_true_data()

# System message to provide an basic role
system_message = """<|im_start|>system
Du bist Politiker und sprichst im deutschen Bundestag.<|im_end|>

"""

prompt_format = "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

generator = pipeline(model="LeoLM/leo-hessianai-13b-chat", device="cuda", torch_dtype=torch.float16, trust_remote_code=False)


generated_texts_df = pd.DataFrame(columns=["text", "id", "elite"])
generated_texts = []


# Generate 3000 different anti-elitist statements
for i in range(3000):

    prompt = "Taetige eine anti-elitistische Aussage."

    response = generator(prompt_format.format(system_message=system_message, prompt=prompt), do_sample=True, top_p=0.95, max_length=3000)
    assistant_text = extract_assistant_text(response[0])
    if assistant_text:
        generated_texts.append(assistant_text)

        # Append each generated text to the DataFrame
        new_data = pd.DataFrame({
            "text": [assistant_text],
            "id": [train_data["id"].max() + 1 + i],  # increment ID to ensure uniqueness
            "elite": [1]  # assuming all generated texts are labeled as anti-elitism
        })
        generated_texts_df = pd.concat([generated_texts_df, new_data], ignore_index=True)


# Save the generated texts to a file for qualitative analysis
with open(os.path.join(os.path.dirname(__file__), '../generated_data/qualitative_analysis/generated_texts_role_basic.txt'), "w", encoding="utf-8") as f:
    os.makedirs(os.path.dirname(f.name), exist_ok=True)
    for text in generated_texts:
        f.write(text + "\n\n")
        f.write("\n" + "="*50 + "\n\n")


# Append the generated DataFrame to the existing DataFrame
expanded_train_data = pd.concat([train_data, generated_texts_df], ignore_index=True)

# Save the expanded DataFrame to a new CSV file for training a new model
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated_data/csv_training_data'))
output_csv_file = os.path.join(output_dir, "expanded_train_data_role_basic.csv")
expanded_train_data.to_csv(output_csv_file, index=False)


