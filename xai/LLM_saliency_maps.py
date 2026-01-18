import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model
model_name = "C:/Users/yassi/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, device_map="cuda", torch_dtype=torch.float16)
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

input_ids = inputs["input_ids"]
input_embeds = model.get_input_embeddings()(input_ids)
input_embeds.retain_grad()

# Forward pass
model.zero_grad()
outputs = model(inputs_embeds=input_embeds)
logits = outputs.logits
target_logit = logits[0, -1].max()  # Use the most probable next token logit

# Backward pass to get gradients
target_logit.backward()
grads = input_embeds.grad[0].norm(dim=-1).detach().cpu().numpy()

# Convert token ids to words
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
token_labels = [token.replace('▁', ' ') for token in tokens]

# Plot saliency map
plt.figure(figsize=(12, 2))
sns.barplot(x=token_labels, y=grads, palette="viridis")
plt.xticks(rotation=90)
plt.title("Saliency Map (Token Importance via Gradient Norm)")
plt.tight_layout()
plt.savefig("saliency_map.png")