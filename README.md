---
license: mit
language:
- en
library_name: transformers
pipeline_tag: text2text-generation
datasets:
- bookcorpus
- openwebtext
- CommonCrawl
- c4
- vblagoje/lfqa
- vblagoje/lfqa_support_docs
tags:
- abstractive-question-answering
- generative-pre-trained-transformer
metrics:
- accuracy
---

# Model Description

ARISE stands for Advanced Response and Information Synthesis Engine. It is an open source Bidirectional Auto Regressive Language Model similar to BART pre-trained on multiple datasets which has already beem mentioned in the model tags, this LM has shown SOTA performance in terms of generating Anstractive Answers. This LM is capable of performing Abstractive Question Answering task, it even outperforms GPT-3 like Large Language Models in terms of providing Abstractive Answers.

# Training Limitations

The model was already been trained on a downstream task which is Abstractive Question-Answering. On fine-tuning this model on any dataset will completly overwrite its abilities, so i would suggest you to fine tune BART or T5 like models.

# Must Know About this LM

ARISE needs context inorder to generate an Abstractive Answer to a Query.

# Use Case

The Model is already been trained on a downstream task as mentioned before, So Out Of The Box this model can be used for Abstractive or Generative Question-Answering Task. Below is The code for use case----

```python

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_checkpoints = "ragnarOk321/GPT-Q-160M"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoints)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoints)

question = "Why does water heated to room temperature feel colder than the air around it?" 

context = [
"when the skin is completely wet. The body continuously loses water by...",

"at greater pressures. There is an ambiguity, however, as to the meaning of the terms 'heating' and 'cooling'...",

"are not in a relation of thermal equilibrium, heat will flow from the hotter to the colder, by whatever pathway...",

"air condition and moving along a line of constant enthalpy toward a state of higher humidity. A simple example ...",   

"Thermal contact conductance In physics, thermal contact conductance is the study of heat conduction between solid ..."
]
            
context = "<P>" + "<P>".join([line for line in context])
prompt = f"question: {question} context:{context} "

input= tokenizer(
                  prompt,
                  truncation=True,
                  padding=True, 
                  return_tensors='pt' 
) 

response = model.generate(
                  input_ids=input['input_ids'],
                  attention_mask=input['attention_mask'], 
                  min_length=64, 
                  max_length=256, 
                  do_sample=False, 
                  early_stopping=True, 
                  num_beams=8, 
                  temperature=1.0, 
                  top_k=None, 
                  top_p=None, 
                  eos_token_id=tokenizer.eos_token_id, 
                  no_repeat_ngram_size=3, 
                  num_return_sequences =1
) 

decoded_response = tokenizer.batch_decode(
                  response, 
                  skip_special_tokens=True, 
                  clean_up_tokenization_spaces=True
) 

print(f"Answer: {decoded_response} ")
```
# Libraries 

The necessary dependencies must be installed before deployment of the model:-
```python
pip install transformers
pip install torch
```

# GitHub 

# Email 

Contact me on this email if necessary-
cl7c.rn04@assemblyofangels.org
