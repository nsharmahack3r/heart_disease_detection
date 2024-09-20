import numpy as np
import skfuzzy as fuzz
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Load the dataset
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

# Define the TensorFlow (Keras) model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# Predict on the test data
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluation metrics
classification_rep = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(classification_rep)
print(f"ROC AUC Score: {roc_auc:.4f}")
#
# converter  = tf.lite.TFLiteConverter.from_keras_model(model)
# tfmodel = converter.convert()

print(X_test[0])
# open("heart.tflite", "wb").write(tfmodel)
