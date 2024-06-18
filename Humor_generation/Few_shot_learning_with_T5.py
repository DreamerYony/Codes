# 1. Install libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

# 2. Prepare Model
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# optimizer
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in t5_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in t5_model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-4, eps=1e-8)

# dataset preparation
true_false_adjective_tuples = [
    ('The Shawshank Redemption', 'The sawshank Redemption'),
    ('The Dark Knight', 'The darke Knight'),
    ('Fight Club', 'fright Club'),
    ('Pulp Fiction', 'Pulp friction'),
    ('Forrest Gump', 'Forrest gumpp'),
    ('The Lord of the Rings: The Fellowship of the Ring', "The Lord of the ring's: The Fellowship of the Ring"),
    ('The Lord of the Rings: The Return of the King', "The Lord of the ring's: The Return of the King"),
    ('The Godfather', 'The grandfather'),
    ('Game of Thrones', 'Aim of Thrones'),
    ('The Dark Knight Rises', 'The darke Knight Rises'),
    ('The Lord of the Rings: The Two Towers', "The Lord of the ring's: The Two Towers"),
    ('Gladiator', 'generator'),
    ('Batman Begins', 'bethann Begins'),
    ('Breaking Bad', 'baking Bad'),
    ('Star Wars: Episode IV - A New Hope', 'spahr Wars: Episode IV - A New Hope'),
    ('The Silence of the Lambs', "The Silence of the lamb's")
]

# 3. Train Loop
t5_model.train()

epochs = 10

for epoch in range(epochs):
    print("epoch ", epoch)
    for input, output in true_false_adjective_tuples:
        input_sent = "sound_change: " + input + " </s>"
        ouput_sent = output + " </s>"

        tokenized_inp = tokenizer.encode_plus(input_sent, max_length=96, pad_to_max_length=True, return_tensors="pt")
        tokenized_output = tokenizer.encode_plus(ouput_sent, max_length=96, pad_to_max_length=True, return_tensors="pt")

        input_ids = tokenized_inp["input_ids"]
        attention_mask = tokenized_inp["attention_mask"]

        lm_labels = tokenized_output["input_ids"]
        decoder_attention_mask = tokenized_output["attention_mask"]

        # the forward function automatically creates the correct decoder_input_ids
        output = t5_model(input_ids=input_ids, labels=lm_labels, decoder_attention_mask=decoder_attention_mask, attention_mask=attention_mask)
        loss = output.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 4. Test model
sentence = [
    'Avengers: Infinity War', 'As Good as It Gets', 'Blue Velvet', 'In the Heart of the Sea', 'Peaky Blinders', 
    'Ghost Rider', 'Die Hard 2', 'The Girl Next Door', 'Men in Black II', 'Enemy of the State'
]

def get_result(sentence):
    test_sent = f"Sound change: {sentence} </s>"
    test_tokenized = tokenizer.encode_plus(test_sent, return_tensors="pt")
    test_input_ids = test_tokenized["input_ids"]
    test_attention_mask = test_tokenized["attention_mask"]

    t5_model.eval()
    beam_outputs = t5_model.generate(
        input_ids=test_input_ids, attention_mask=test_attention_mask,
        max_length=64,
        early_stopping=True,
        num_beams=10,
        num_return_sequences=5,
        no_repeat_ngram_size=2
    )
    sent_list = []
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        sent_list.append(sent)
    return sent_list

results = []
for s in sentence:
    results.append(get_result(s))
print(results)

# test_sent = "generate humor: Avengers: Infinity War </s>"
test_tokenized = tokenizer.encode_plus(test_sent, return_tensors="pt")

test_input_ids = test_tokenized["input_ids"]
test_attention_mask = test_tokenized["attention_mask"]

t5_model.eval()
beam_outputs = t5_model.generate(
    input_ids=test_input_ids, attention_mask=test_attention_mask,
    max_length=64,
    early_stopping=True,
    num_beams=10,
    num_return_sequences=5,
    no_repeat_ngram_size=2
)

for beam_output in beam_outputs:
    sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(sent)