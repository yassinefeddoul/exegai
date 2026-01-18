import json
import pandas as pd
from core.models.utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from interpret.blackbox import LimeTabular
import boto3

# Load dataset
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
X_train, X_test, y_train, y_test = data_loader.get_data_split()
X_train, y_train = data_loader.oversample(X_train, y_train)
X_train = X_train.apply(pd.to_numeric, errors='coerce')
bool_cols = X_train.select_dtypes(include=['bool']).columns
X_train[bool_cols] = X_train[bool_cols].astype(int)

# Initialize Bedrock client for us-east-1
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def generate_llama_response(prompt):
    """Generate response using Bedrock's Llama 3 8B."""
    response = bedrock.invoke_model(
        modelId="meta.llama3-8b-instruct-v1:0",  # Updated model ID
        body=json.dumps({
            "prompt": prompt,
            "max_gen_len": 512,
            "temperature": 0.7
        })
    )
    return json.loads(response["body"].read())["generation"]

def prepare_user_input(user_data, reference_df):
    df = pd.DataFrame([user_data])
    for col in reference_df.columns:
        if col not in df.columns:
            df[col] = 0
    df = df[reference_df.columns]
    df = df.apply(pd.to_numeric, errors='coerce')
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df

def model_fn(model_dir):
    """Load the model (placeholder for SageMaker)."""
    return {}

def predict_fn(input_data, model_dict):
    """Run LIME + LLM inference."""
    # Step 1: Run LIME
    rf = RandomForestClassifier().fit(X_train, y_train)
    user_df = prepare_user_input(input_data, X_train)
    lime = LimeTabular(model=rf, data=X_train, random_state=1)
    lime_local = lime.explain_local(user_df, [rf.predict(user_df)[0]], name='LIME')
    data = lime_local.data(0)

    explanation = [
        {"feature": f, "value": float(v), "contribution": float(s)}
        for f, v, s in zip(data["names"], data["values"], data["scores"])
    ]

    # Step 2: Generate LLM explanation
    explanation_items = "\n".join(
        f"- {item['feature']} = {item['value']} (contribution: {item['contribution']:.3f})"
        for item in explanation
    )

    risk_status = "at risk of stroke" if rf.predict(user_df)[0] else "not at risk of stroke"

    prompt = f"""You are a medical AI assistant. A patient was classified as {risk_status}.
    Below are the contributing factors from an explainable AI model (LIME):

    {explanation_items}

    Write a clear, human-friendly explanation summarizing *why* the patient was classified this way, highlighting the most influential factors.
    Do not include any code snippets or technical details in your response. Focus only on the explanation."""

    response = generate_llama_response(prompt)  # Use Bedrock's Llama 3.1 8B

    return {
        # "prediction": int(rf.predict(user_df)[0]),
        # "prediction_score": float(lime_local.data(0)["perf"]["predicted_score"]),
        # "explanation": explanation,
        "llm_explanation": response
    }

if __name__ == "__main__":
    print("Enter patient data:")

    sample_input = {
        "age": int(input("Age: ")),
        "hypertension": int(input("Hypertension (0 = No, 1 = Yes): ")),
        "heart_disease": int(input("Heart disease (0 = No, 1 = Yes): ")),
        "avg_glucose_level": float(input("Average glucose level: ")),
        "bmi": float(input("BMI: ")),
        "gender": input("Gender (Male/Female): "),
        "ever_married": input("Ever married (Yes/No): "),
        "work_type": input("Work type (Private/Self-employed/Govt_job/etc.): "),
        "Residence_type": input("Residence type (Urban/Rural): "),
        "smoking_status": input("Smoking status (never smoked/formerly smoked/smokes/Unknown): ")
    }

    result = predict_fn(sample_input, model_fn(None))
    print("\n Explanation Result:")
    print(json.dumps(result, indent=2))