import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os
import torch.nn.functional as F

# This must be the FIRST Streamlit command
st.set_page_config(page_title="Explainable LLM", layout="wide")

# Load the Hugging Face token from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Model & tokenizer setup
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_PATH = "/mnt/llama"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, cache_dir=MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Use float32 on CPU
        device_map="cpu",
        token=HF_TOKEN,
        cache_dir=MODEL_PATH
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Saliency computation
def compute_saliency(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Get embeddings and enable gradients
    inputs_embeds = model.get_input_embeddings()(input_ids)
    inputs_embeds.retain_grad()
    inputs_embeds.requires_grad_()

    # Forward pass
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=inputs["attention_mask"])
    logits = outputs.logits
    loss = logits[:, -1, :].sum()  # Aggregate some scalar output
    loss.backward()

    # Compute saliency as norm of gradients
    saliency = inputs_embeds.grad.norm(dim=-1).squeeze()
    return input_ids.squeeze(), saliency

# Visualization
def render_saliency(input_ids, saliency_scores):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    saliency_scores = saliency_scores / saliency_scores.max()
    html = ""
    for token, score in zip(tokens, saliency_scores):
        intensity = int(255 * (1 - score.item()))
        color = f"rgb(255,{intensity},{intensity})"
        html += f'<span style="background-color: {color}; padding:2px; margin:2px; border-radius:4px;">{token}</span>'
    return html

# Streamlit UI
st.title("🧠 Explainable LLM Chatbot")

user_input = st.text_area("Enter your question:", height=150)

generate_button = st.button("Generate Response")
explain = st.checkbox("Explain this response (saliency map)")

if generate_button and user_input:
    with st.spinner("Generating response..."):
        inputs = tokenizer(user_input, return_tensors="pt")
        input_ids = inputs["input_ids"]
        output = model.generate(input_ids=input_ids, max_new_tokens=100)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        st.subheader("Response:")
        st.markdown(response)

        if explain:
            with st.spinner("Computing saliency map..."):
                input_ids, saliency = compute_saliency(user_input)
                html = render_saliency(input_ids, saliency)
                st.subheader("Explanation (Saliency Map):")
                st.markdown(html, unsafe_allow_html=True)
