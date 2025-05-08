# LAB8
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 🔹 Load BART model with attentions enabled
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name, output_attentions=True)

# 🔹 Input text
text = """Albert Einstein was a theoretical physicist who revolutionized science with his theory of relativity.
He also made significant contributions to quantum mechanics.
 He received the Nobel Prize in 1921 for the photoelectric effect."""

# 🔹 Tokenize
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
input_ids = inputs["input_ids"]

# 🔹 Generate summary
output = model.generate(
    input_ids,
    max_length=40,
    min_length=10,
    num_beams=4,
    early_stopping=True,
    output_attentions=True,
    return_dict_in_generate=True
)

summary_ids = output.sequences[0]
summary_tokens = tokenizer.convert_ids_to_tokens(summary_ids)
input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# 🔹 Get cross-attention from first decoder layer, averaged across heads
cross_attn = output.cross_attentions[0]  # List of tuples (layers)
avg_attn = cross_attn[0].mean(dim=0).squeeze().detach().numpy()  # (target_len, source_len)

# 🔹 Truncate tokens to match shapes
target_tokens = summary_tokens[:avg_attn.shape[0]]
source_tokens = input_tokens[:avg_attn.shape[1]]

# 🔹 Plot attention heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(avg_attn, xticklabels=source_tokens, yticklabels=target_tokens, cmap="YlGnBu")
plt.xlabel("Input Tokens")
plt.ylabel("Generated Summary Tokens")
plt.title("Cross-Attention Heatmap (BART)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
