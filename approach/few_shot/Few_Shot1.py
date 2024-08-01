#!/usr/bin/env python
# coding: utf-8

from src.bert.dataset import PBertDataset
from src.bert.dataset.strategies import MLMin1PopIdeol  
from pathlib import Path
import src
import pandas as pd

from transformers import pipeline
import torch

# import the few shot data from the training set
EXCLUDE_CODERS = []
train = PBertDataset.from_disk(
    path=src.PATH / "data/labeled_data/train.csv.zip",
    label_strategy=MLMin1PopIdeol(),
    exclude_coders=EXCLUDE_CODERS,
)
train.apply_label_strategy

train_data = train.df_labels
train_data = train_data[["id", "text", "vote"]]

train_data["elite"] = train_data["vote"].apply(lambda x: x[0])
train_data = train_data[["id", "text", "elite"]]

# filter for training data labeled with True for anit-elitism
true_label_data = train_data[train_data['elite'] == 1]

prompt_format = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
generator = pipeline(model="LeoLM/leo-hessianai-13b-chat", device="cuda", torch_dtype=torch.float16, trust_remote_code=False)

generated_texts = []

# extract the assistant text 
def extract_assistant_text(response):
    text = response['generated_text']
    # search for the answer of assistant
    start = text.find('assistant\n')
    if start != -1:
        # extract the text after 'assistant\n'
        return text[start + len('assistant\n'):]
    return None

# generate 1000 answers providing the generated data
for i in range(1000):
    # select four random examples of anti-elitism
    few_shot_examples = true_label_data.sample(n=4)
    
   # Format the examples
    formatted_examples = ""
    for _, example in few_shot_examples.iterrows():
        formatted_examples += f"text: {example['text']}"
    
    # define the "generation prompt" for new examples of anti-elitism
    prompt = f"""
    Hier sind 4 Beispiele fuer Anti-Elitismus:
    {formatted_examples}\n
    Taetige genau eine anti-elitistische Aussagen zu einem politischen Thema.
    """
    generated_response = generator(prompt_format.format(prompt=prompt), do_sample=True, top_p=0.95, max_length=8192)
    assistant_text = extract_assistant_text(generated_response[0])
    if assistant_text:
        generated_texts.append(assistant_text)

# save the generated texts to a file
with open("generated_texts.txt", "w", encoding="utf-8") as f:
    for text in generated_texts:
        f.write(text + "\n\n")

print("successfully generated 1000 examples of anti-elitism statements.")
