import numpy as np
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd


file_path = './heart.csv'
data = pd.read_csv(file_path)


# Define fuzzy membership functions for continuous features
def fuzzify_age(age):
    young = fuzz.trimf(age, [20, 20, 35])
    middle_aged = fuzz.trimf(age, [30, 45, 60])
    old = fuzz.trimf(age, [50, 75, 100])
    return young, middle_aged, old

def fuzzify_chol(chol):
    low = fuzz.trimf(chol, [100, 150, 200])
    medium = fuzz.trimf(chol, [180, 230, 280])
    high = fuzz.trimf(chol, [240, 300, 400])
    return low, medium, high

def fuzzify_trestbps(trestbps):
    normal = fuzz.trimf(trestbps, [80, 120, 140])
    high = fuzz.trimf(trestbps, [120, 140, 160])
    very_high = fuzz.trimf(trestbps, [140, 180, 200])
    return normal, high, very_high

def fuzzify_thalach(thalach):
    low = fuzz.trimf(thalach, [70, 100, 130])
    medium = fuzz.trimf(thalach, [120, 150, 170])
    high = fuzz.trimf(thalach, [160, 190, 220])
    return low, medium, high

# Fuzzify the continuous features
age = np.array(data['age'])
chol = np.array(data['chol'])
trestbps = np.array(data['trestbps'])
thalach = np.array(data['thalach'])

fuzzy_age = fuzzify_age(age)
fuzzy_chol = fuzzify_chol(chol)
fuzzy_trestbps = fuzzify_trestbps(trestbps)
fuzzy_thalach = fuzzify_thalach(thalach)

# Create a new dataset with fuzzified features (taking one fuzzy set per variable for simplicity)
fuzzy_X = np.column_stack((
    fuzz.interp_membership(age, fuzzy_age[1], age),       # middle-aged fuzzified values
    fuzz.interp_membership(chol, fuzzy_chol[1], chol),    # medium cholesterol fuzzified values
    fuzz.interp_membership(trestbps, fuzzy_trestbps[1], trestbps),  # high BP fuzzified values
    fuzz.interp_membership(thalach, fuzzy_thalach[1], thalach)      # medium heart rate fuzzified values
))

# Add categorical and binary features directly
fuzzy_X = np.column_stack((fuzzy_X, data[['sex', 'cp', 'fbs', 'restecg', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]))

# Target variable (heart disease presence)
y = data['target']

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(fuzzy_X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation metrics
classification_rep = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(classification_rep, roc_auc)