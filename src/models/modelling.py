import pandas as pd
import numpy as np 
import pickle 
import yaml
from sklearn.ensemble import RandomForestClassifier

with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

n_estimators = params['model_training']['n_estimators']
max_depth = params['model_training']['max_depth']

# Load the training data with Bag-of-Words features
train_data = pd.read_csv("data/interim/train_tfidf.csv")

# Separate features and target variable
x_train = train_data.drop(columns=['label']).values  # Features for training
y_train = train_data['label'].values                # Target labels

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

# Train the model on the training data
model.fit(x_train, y_train)

# Save the trained model to disk using pickle
pickle.dump(model, open("models/random_forest_model.pkl", "wb"))