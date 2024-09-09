
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
from common_methods import load_training_data
from common_methods import extract_assistant_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the training data from Erhard et al., 2023
train_data = load_training_data()

# Initialization of prompt format and the system message
prompt_format = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

# Initialization of the model
generator = pipeline(model="LeoLM/leo-hessianai-13b-chat", device="cuda", torch_dtype=torch.float16, trust_remote_code=False)

# Optional: Generate context through a definition of anti-elitism as context
definition = ""

def generate_multi_step_prompting_statements(definition):
    # Step 1: Generate contexts/settings
    prompt1 = "Generiere ein Thema, das Anti-Elitismus aufgreift."
    response1 = generator(prompt_format.format(prompt=prompt1), do_sample=True, top_p=0.95, max_length=3000)
    contexts = extract_assistant_text(response1[0])
    
    # Step 2: Generate instance seeds
    prompt2 = f"Generiere zu folgendem Thema einen Satz, der Anti-Elitismus beinhaltet:\n{contexts}"
    response2 = generator(prompt_format.format(prompt=prompt2), do_sample=True, top_p=0.95, max_length=3000)
    instance_seeds = extract_assistant_text(response2[0])
       
    # Step 3: Generate Data instances on the basis of the instance seeds
    prompt3 = f"Verwende den folgenden Satz, um genau eine anti-elitistische Aussage zu generieren:\n{instance_seeds}"
    response3 = generator(prompt_format.format(prompt=prompt3), do_sample=True, top_p=0.95, max_length=3000)
    data_instances = extract_assistant_text(response3[0])
        
    # Step 4: Self-Correction
    prompt4 = f"Ueberpruefe die folgenden anti-elitistischen Aussagen und korrigiere eventuelle falsch etikettierte Instanzen:\n{data_instances}"
    response4 = generator(prompt_format.format(prompt=prompt4), do_sample=True, top_p=0.95, max_length=3000)
    corrected_data_instances = extract_assistant_text(response4[0])
    
    return contexts, instance_seeds, data_instances, corrected_data_instances


# Initialize an empty DataFrame to store corrected data instances
corrected_data_df = pd.DataFrame(columns=["text", "id", "elite"])

# Save the generated texts to a file for qualitative analysis
with open(os.path.join(os.path.dirname(__file__), '../generated_data/qualitative_analysis/generated_texts_cot.txt'), "w", encoding="utf-8") as file:
    os.makedirs(os.path.dirname(file.name), exist_ok=True)
    for i in range(500):
        # Perform the steps
        contexts, instance_seeds, data_instances, corrected_data_instances = generate_multi_step_prompting_statements(definition)

        # Write the results to a text file for qualitative analysis
        file.write(f"Iteration {i + 1}:\n")
        file.write("Generierte Kontexte/Settings:\n")
        file.write(contexts + "\n\n")
        
        file.write("Generierte Instanz-Samen:\n")
        file.write(instance_seeds + "\n\n")
        
        file.write("Generierte Dateninstanzen:\n")
        file.write(data_instances + "\n\n")
        
        file.write("Korrigierte Dateninstanzen:\n")
        file.write(corrected_data_instances + "\n\n")
        
        # Add corrected data instances to the DataFrame
        if corrected_data_instances:
            for line in corrected_data_instances.split("\n"):
                if line.strip():
                    corrected_data_df = pd.concat([corrected_data_df, pd.DataFrame({
                        "text": [line],
                        "id": [train_data["id"].max() + 1 + i], 
                        "elite": [1]
                    })], ignore_index=True)


# Append the corrected DataFrame to the existing DataFrame
expanded_train_data = pd.concat([train_data, corrected_data_df], ignore_index=True)

# Save the expanded DataFrame to a new CSV file for training a new model
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated_data/csv_training_data'))
output_csv_file = os.path.join(output_dir, "expanded_train_data_msp.csv")
expanded_train_data.to_csv(output_csv_file, index=False)
