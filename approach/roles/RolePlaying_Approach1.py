#!/usr/bin/env python
# coding: utf-8

from transformers import pipeline
import torch

# Initialisierung des Generators

# system message including a "role prompt" and a "audience prompt"
system_message = """<|im_start|>system
Du bist Politiker und sprichst im deutschen Bundestag.<|im_end|>

"""

# declaring the prompt format consisting of the system prompt, the user prompt and the generated answer by the assistant
prompt_format = "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
generator = pipeline(model="LeoLM/leo-hessianai-13b-chat", device="cuda", torch_dtype=torch.float16, trust_remote_code=False)


# only extract the assistant text serving as generated data
def extract_assistant_text(response):
    text = response['generated_text']
    # Suche nach dem Start der Assistant-Antwort
    start = text.find('assistant\n')
    if start != -1:
        # Extrahiere den Text nach 'assistant\n'
        return text[start + len('assistant\n'):]
    return None

# generate 1000 different political statements based on the system message and a basic "generation prompt"
for i in range(1000):
    prompt = "Taetige eine anti-elitistische Aussage."
    # usage of the model to generate the instances
    response = generator(prompt_format.format(system_message=system_message, prompt=prompt), do_sample=True, top_p=0.95, max_length=200)
    final_text = extract_assistant_text(response[0])
    
    # print the final generated data
    print(final_text)
    print("=" * 50) 
