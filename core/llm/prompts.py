import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load model
model_name = "C:/Users/yassi/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             torch_dtype=torch.float16)
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

response = generator(prompt, max_new_tokens=200)[0]["generated_text"]
print("LLM Response:\n", response)