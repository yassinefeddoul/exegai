# ExeGAI  
**Explainable Generative Artificial Intelligence**

ExeGAI is an end-to-end **Explainable AI (XAI)** application that combines **classical machine-learning explainability** (e.g. LIME / SHAP) with **Large Language Models (LLMs)** to generate **human-interpretable explanations** of model predictions through an interactive **Streamlit** interface.

The project explores how **statistical explanations** can be enriched by **generative reasoning**, bridging the gap between model-level interpretability and user-level understanding.

---

## Key Features

- **Hybrid XAI pipeline**
  - Classical ML predictor (e.g., Random Forest)
  - Local explainability methods (LIME / SHAP)
  - LLM-based explanation synthesis

- **LLM-augmented explanations**
  - Converts raw XAI outputs into structured, natural-language insights
  - Supports instruction-tuned LLaMA models

- **Interactive Streamlit UI**
  - User input → prediction → explanation in a single workflow
  - Designed for both technical and non-technical users

- **Research-oriented architecture**
  - Modular, extensible, and reproducible
  - Suitable for experimentation and academic dissemination

---

## High-Level Architecture

User Input
↓
ML Predictor
↓
XAI Module (LIME / SHAP)
↓
LLM Reasoning Layer
↓
Natural-Language Explanation
↓
Streamlit Interface

## Repository Structure

```bash
exegai/
├── app/ # Streamlit application
├── core/ # Core logic (models, XAI, LLMs)
├── data/ # Sample or documented datasets
├── configs/ # Configuration files
├── scripts/ # Utilities and local execution
├── docs/ # Technical documentation
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yassinefeddoul/exegai.git
cd exegai
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Create a .env file

```bash
HF_TOKEN=your_huggingface_token
MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
```

## Running the Application

```bash
streamlit run app/streamlit_app.py
```
