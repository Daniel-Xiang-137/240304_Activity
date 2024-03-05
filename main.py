#Create summarization, ner, text generation, and classification pipelines using hugging face transformers.

import os
from transformers import pipeline

os.system("cls")

"""
with open(file="pg345.txt", mode="r", encoding="utf8") as f:
    text = " ".join(f.readlines())
print(text)
print()
"""

text = "Richard of York gave battle in vain."
print(text)
print()

#Summarization
sum_pipe = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)
print(sum_pipe(text))
print()

#NER
ner_pipe = pipeline(
    "ner",
    model="dslim/bert-base-NER"
)
print(ner_pipe(text))
print()

#Text Generation
tg_pipe = pipeline(
    model="openai-community/gpt2"
)
print(tg_pipe("the quick brown fox", do_sample=False))
print()

#Classification
classification_pipe = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
)
print(classification_pipe(text))
print()