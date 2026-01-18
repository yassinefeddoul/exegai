import json
import subprocess

user_data = {
    "age": 39,
    "hypertension": 0,
    "heart_disease": 0,
    "avg_glucose_level": 55.12,
    "bmi": 30.5,
    "gender": 1,
    "ever_married": 1,
    "work_type": 2,
    "Residence_type": 1,
    "smoking_status": 2
}

# Save user input
with open("user_input.json", "w") as f:
    json.dump(user_data, f)

# Step 1: Generate LIME explanation
subprocess.run(["python", "02_lime.py"])

# Step 2: Generate LLM explanation
subprocess.run(["python", "LLM.py"])