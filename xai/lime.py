# %% Imports
from core.models.utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from interpret.blackbox import LimeTabular
from interpret import show
import pandas as pd
import json

def prepare_user_input(user_data, reference_df):
    df = pd.DataFrame([user_data])
    
    # Add missing columns
    for col in reference_df.columns:
        if col not in df.columns:
            df[col] = 0  # or np.nan if you'd rather impute later

    # Drop any extra columns not used in training
    df = df[reference_df.columns]
    
    # Coerce types
    df = df.apply(pd.to_numeric, errors='coerce')
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df

# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
# Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)
X_train = X_train.apply(pd.to_numeric, errors='coerce')
bool_cols = X_train.select_dtypes(include=['bool']).columns
X_train[bool_cols] = X_train[bool_cols].astype(int)
print(X_train.shape)
print(X_test.shape)

# %% Fit blackbox model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %% Apply lime
# Initilize Lime for Tabular data
lime = LimeTabular(model=rf, 
                   data=X_train, 
                   random_state=1)
# Get local explanations
lime_local = lime.explain_local(X_test[-1:], 
                                y_test[-1:], 
                                name='LIME')

# Load user input
with open("user_input.json") as f:
    user_data = json.load(f)

user_df = prepare_user_input(user_data, X_train)

# %% Predict for user input
user_pred = rf.predict(user_df)[0]
user_proba = rf.predict_proba(user_df)[0]

# %% Generate LIME Explanation
lime = LimeTabular(model=rf, data=X_train, random_state=1)
lime_local = lime.explain_local(user_df, [user_pred], name='LIME')
data = lime_local.data(0)

explanation = [
    {"feature": f, "value": float(v), "contribution": float(s)}
    for f, v, s in zip(data["names"], data["values"], data["scores"])
]

output = {
    "prediction": int(user_pred),
    "prediction_score": float(data["perf"]["predicted_score"]),
    "explanation": explanation
}

with open("lime_output.json", "w") as f:
    json.dump(output, f)


# %%
# show(lime_local)