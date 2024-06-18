from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-large")
model = AutoModelForMaskedLM.from_pretrained("ethanyt/guwenbert-large")

sequence = f"筑基仿崔巍，鞭石轻险{tokenizer.mask_token}。"

input = tokenizer.encode(sequence, return_tensors="pt")
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

token_logits = model(input).logits
mask_token_logits = token_logits[0, mask_token_index, :]

top_10_tokens = torch.topk(mask_token_logits, 10, dim=1).indices[0].tolist()

for token in top_10_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))

topk_list = [tokenizer.decode([token]) for token in top_10_tokens]

print(topk_list)
