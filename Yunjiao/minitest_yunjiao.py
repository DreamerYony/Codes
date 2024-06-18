from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-large")
model = AutoModelForMaskedLM.from_pretrained("ethanyt/guwenbert-large")

# Input sequence with masked token
sequence = f"筑基仿崔巍，鞭石轻险{tokenizer.mask_token}。"

# Tokenize input sequence
input = tokenizer.encode(sequence, return_tensors="pt")
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

# Get logits for masked token
token_logits = model(input).logits
mask_token_logits = token_logits[0, mask_token_index, :]

# Get top 10 tokens for the masked position
top_10_tokens = torch.topk(mask_token_logits, 10, dim=1).indices[0].tolist()

# Print the sequence with each top token replacing the masked token
for token in top_10_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))

# Collect the top tokens into a list
topk_list = [tokenizer.decode([token]) for token in top_10_tokens]

# Print the list of top tokens
print(topk_list)