# %% Imports
from core.models.utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
# Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)
print(X_train.shape)
print(X_test.shape)

# %% Fit blackbox model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")


# %% Create diverse counterfactual explanations
# pip install dice-ml
import dice_ml
# Dataset
dataframe = data_loader.data
dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
bool_cols = dataframe.select_dtypes(include=['bool']).columns
dataframe[bool_cols] = dataframe[bool_cols].astype(int)
data_dice = dice_ml.Data(dataframe=dataframe, 
                         # For perturbation strategy
                         continuous_features=['age', 
                                              'avg_glucose_level',
                                              'bmi'], 
                         outcome_name='stroke')
# Model
rf_dice = dice_ml.Model(model=rf, 
                        # There exist backends for tf, torch, ...
                        backend="sklearn")
explainer = dice_ml.Dice(data_dice, 
                         rf_dice, 
                         # Random sampling, genetic algorithm, kd-tree,...
                         method="random")

# %% Create explanation
# Generate CF based on the blackbox model
input_datapoint = X_test[0:1]
input_datapoint = input_datapoint.apply(pd.to_numeric, errors='coerce')
bool_cols = input_datapoint.select_dtypes(include=['bool']).columns
input_datapoint[bool_cols] = input_datapoint[bool_cols].astype(int)
print(input_datapoint)
print("Unique values in training data:", input_datapoint['gender_Female'].unique())
cf = explainer.generate_counterfactuals(input_datapoint, 
                                  total_CFs=3, 
                                  desired_class="opposite")
# Visualize it
cf.visualize_as_dataframe(show_only_changes=True)


# %% Create feasible (conditional) Counterfactuals
features_to_vary=['avg_glucose_level',
                  'bmi',
                  'smoking_status_smokes']
permitted_range={'avg_glucose_level':[50,250],
                'bmi':[18, 35]}
# Now generating explanations using the new feature weights
cf = explainer.generate_counterfactuals(input_datapoint, 
                                  total_CFs=3, 
                                  desired_class="opposite",
                                  permitted_range=permitted_range,
                                  features_to_vary=features_to_vary)
# Visualize it
cf.visualize_as_dataframe(show_only_changes=True)

