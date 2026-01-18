import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# Load model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Replace with your local path if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float32
)
model.eval()

st.title("🧠 Explainable AI Chatbot")

# User input
user_input = st.text_area("Ask me anything:", height=100)
show_explanation = st.checkbox("Show explanation")

def compute_saliency(prompt):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Get input embeddings
    inputs_embeds = model.get_input_embeddings()(input_ids)
    inputs_embeds.retain_grad()

    # Forward pass using embeddings
    outputs = model(inputs_embeds=inputs_embeds)
    logits = outputs.logits

    # Use the most confident output token as proxy loss
    loss = logits[0, -1].max()
    loss.backward()

    # Compute token-wise saliency from gradient norm
    saliency = inputs_embeds.grad.norm(dim=-1).squeeze()
    return input_ids.squeeze(), saliency

def highlight_tokens(input_ids, saliency):
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)  # normalize
    html_output = ""
    for token, score in zip(tokens, saliency):
        color = f"rgba(255, 0, 0, {score.item():.2f})"
        token = token.replace("Ġ", " ")  # for Falcon-style tokenizers
        html_output += f'<span style="background-color: {color}; padding: 2px; margin:1px; border-radius: 4px;">{token}</span>'
    return html_output

if st.button("Generate"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating response..."):
            inputs = tokenizer(user_input, return_tensors="pt")
            output_ids = model.generate(**inputs, max_new_tokens=100)
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Display response
        st.markdown("**Chatbot Response:**")
        st.markdown(response)

        # If explanation is requested
        if show_explanation:
            with st.spinner("Computing explanation..."):
                try:
                    input_ids, saliency = compute_saliency(user_input)
                    saliency_html = highlight_tokens(input_ids, saliency)
                    st.markdown("**Token Contribution (Saliency Map):**", unsafe_allow_html=True)
                    st.markdown(saliency_html, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error computing explanation: {e}")