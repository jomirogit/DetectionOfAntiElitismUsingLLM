#!/usr/bin/env python
# coding: utf-8

# In[21]:


#!/usr/bin/env python
# coding: utf-8

from transformers import pipeline
import torch
import pandas as pd


# Load the entire dataset
dataset = pd.read_csv("cleared_data_majority_vote.csv")

# Filter for True labels
true_label_data = dataset[dataset['majority_vote'] == True]

# taking examples out of the test data set only
# test_data = pd.read_csv("test_data.csv")
# test_texts = test_data['text'].tolist()
# few_shot_data = true_label_data[~true_label_data['text'].isin(test_texts)]

# Select four examples
few_shot_examples = true_label_data.sample(n=4, random_state=1)

# Format the examples
formatted_examples = ""
for _, example in few_shot_examples.iterrows():
    formatted_examples += f"text: {example['text']}\nanti_elitism: {example['majority_vote']}\n\n"

# Define the 'generation prompt' for new examples of anti-elitism
prompt = f"""
Hier sind 4 Beispiele für Anti-Elitismus:
{formatted_examples}\n
Taetige 20 anti-elitistische Aussagen zu verschiedenen politischen Themen.
"""

# Initialize the model
generator = pipeline(model="LeoLM/leo-hessianai-13b-chat", device="cuda", torch_dtype=torch.float16, trust_remote_code=False)

# Generate the anti-elitism statements
generated_responses = generator(prompt, do_sample=True, top_p=0.95, max_length=2000)

# Print the generated statements
for response in generated_responses:
    print(response['generated_text'])


# In[ ]:




