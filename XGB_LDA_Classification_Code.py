# XGB Classifer Stomach trained on in vivo data
# Instructions for use: 
# 1) Update the "csv_path" to desired data and keep csv data in the folder opened
# 2) Make sure that both LDA and XGB pickle files are in the folder that is opened
# OUTPUT: CSV file with column of Predictions including probability of predictions
# Predictions: 0=Normal, 1=Tumour


# %%
## Prediction Script which includes the LDA preprocessing step
import pandas as pd
import numpy as np
import pickle
import os

# Step 1: Load the Trained Model
model_path = 'stomach_in_vivo_xgb_lda1_none.pkl'  # Update with your actual path
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Step 2: Load the LDA Transformation
lda_path = 'lda_transform_stomach.pkl'
try:
    with open(lda_path, 'rb') as lda_file:
        lda = pickle.load(lda_file)
    print("LDA transformation loaded successfully!")
except Exception as e:
    print(f"Error loading LDA: {e}")
    exit()

# Step 3: Load the CSV File
csv_path = 'new_20240415_Stomach_Tumour.csv'  # Update with your CSV file path
try:
    data = pd.read_csv(csv_path)
    print(f"CSV file loaded successfully! Shape: {data.shape}")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

# Step 4: Prepare Data for Prediction
try:
    # Convert to NumPy array
    feature_array = data.to_numpy()

    # Apply LDA transformation
    feature_array_reduced = lda.transform(feature_array)
    print(f"Feature array after LDA: {feature_array_reduced.shape}")
except Exception as e:
    print(f"Error during feature preparation: {e}")
    exit()

# Step 5: Classify Each Row
try:
    predictions = model.predict(feature_array_reduced)
    prediction_probs = model.predict_proba(feature_array_reduced)[:, 1]
    print("Classification completed!")
except Exception as e:
    print(f"Error during classification: {e}")
    exit()

# Step 6: Save Predictions to a CSV File
output_path = 'predictions.csv'
try:
    # Combine original data with predictions and probabilities
    output_df = data.copy()
    output_df['Prediction'] = predictions
    output_df['Probability'] = prediction_probs

    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}!")
except Exception as e:
    print(f"Error saving predictions: {e}")

# %%
