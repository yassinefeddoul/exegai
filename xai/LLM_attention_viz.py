import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model
model_name = "C:/Users/yassi/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, device_map="auto", torch_dtype=torch.float16)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Read LIME output
with open("lime_output.json") as f:
    data = json.load(f)

explanation_items = "\n".join(
    f"- {item['feature']} = {item['value']} (contribution: {item['contribution']:.3f})"
    for item in data["explanation"]
)

risk_status = "at risk of stroke" if data["prediction"] else "not at risk of stroke"

prompt = f"""You are a medical AI assistant. A patient was classified as {risk_status}.
Below are the contributing factors from an explainable AI model (LIME):

{explanation_items}

Write a clear, human-friendly explanation summarizing *why* the patient was classified this way, highlighting the most influential factors."""

# Tokenize with return_tensors
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Forward pass with attentions
with torch.no_grad():
    outputs = model(**inputs)

# Get attention from layer 10, head 0
attn = outputs.attentions[10][0, 0]

# Compute total attention received by each token
attn_scores = attn.sum(dim=0).cpu().numpy()

# Get top-N attended tokens
top_n = 30
top_indices = attn_scores.argsort()[-top_n:]
top_indices.sort()
top_tokens = [tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][i].item()) for i in top_indices]
top_matrix = attn[top_indices][:, top_indices].cpu().numpy()

# Plot
plt.figure(figsize=(12, 10))
sns.heatmap(top_matrix, xticklabels=top_tokens, yticklabels=top_tokens, cmap='viridis', square=True)
plt.title("LLaMA 3.1 Attention (Layer 0, Head 0) - Top 30 Tokens")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("attention_heatmap_top30.png")
plt.show()

# Generate response (optional)
response = generator(prompt, max_new_tokens=200)[0]["generated_text"]
print("LLM Response:\n", response)
