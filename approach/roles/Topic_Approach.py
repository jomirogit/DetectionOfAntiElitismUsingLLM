#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

from transformers import pipeline
import torch

system_message = "Du bist Politiker und sprichst im deutschen Bundestag."

prompt_format = """
system
{system_message}
user
{prompt}
assistant
"""

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
    "Außenpolitik und internationale Beziehungen",
    "Justizsystem und Rechtsreformen",
    "Gleichstellungspolitik und Genderfragen",
    "Energiepolitik und erneuerbare Energien",
    "Landwirtschaft und Ernährungssicherheit",
    "Veteranenangelegenheiten und Militaerpolitik",
    "Kulturpolitik und kulturelle Foerderung"
]

generator = pipeline(model="LeoLM/leo-hessianai-13b-chat", device="cuda", torch_dtype=torch.float16, trust_remote_code=False)

for topic in topics:
    prompt = f"Taetige eine anti-elitistische Aussagen zu {topic}."
    print(generator(prompt_format.format(system_message=system_message, prompt=prompt), do_sample=True, top_p=0.95, max_length=20000))


