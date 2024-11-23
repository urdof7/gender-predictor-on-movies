"""
Author: Brayden Miller
Date: November 22, 2024

Project: Gender Prediction Using Machine Learning of Movie Dataset

Description:
This script serves as a baseline gender prediction model using the 
`gender_guesser` library. It reads a CSV file containing first names 
and true genders, predicts gender for each first name, and evaluates 
the predictions using accuracy and classification metrics. The script 
also calculates the percentage of 'unknown' predictions and provides 
an overall classification report, excluding 'unknown' cases.

Acknowledgments:
This script was developed by Brayden Miller with assistance from ChatGPT.
"""

import pandas as pd
import gender_guesser.detector as gender
import os
from sklearn.metrics import accuracy_score, classification_report

def baseline_predictor(input_csv):
    """
    Reads the input CSV, predicts gender for each first name using the baseline predictor,
    and outputs evaluation metrics without modifying the CSV file.

    Parameters:
    - input_csv: Path to the input CSV file.
    """
    # Load the CSV file
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Filter to records with known true genders
    df = df[df['true_gender'].isin(['male', 'female'])].copy()

    # Drop rows with missing first names
    df = df.dropna(subset=['first_name'])

    # Initialize the gender detector
    detector = gender.Detector(case_sensitive=False)

    # Function to map gender_guesser output to 'male', 'female', or 'unknown'
    def map_predicted_gender(name):
        g = detector.get_gender(name)
        if g in ['male', 'mostly_male']:
            return 'male'
        elif g in ['female', 'mostly_female']:
            return 'female'
        else:
            return 'unknown'

    # Predict gender for each first name
    df['predicted_gender'] = df['first_name'].apply(map_predicted_gender)

    # Extract true and predicted genders
    y_true = df['true_gender']
    y_pred = df['predicted_gender']

    # Calculate overall accuracy including 'unknown' predictions
    overall_accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy (including 'unknown' predictions): {overall_accuracy:.2%}")

    # Exclude 'unknown' predictions for accuracy calculation
    mask_known_predictions = y_pred != 'unknown'
    y_true_known = y_true[mask_known_predictions]
    y_pred_known = y_pred[mask_known_predictions]

    if y_true_known.empty:
        print("No valid predictions to evaluate after excluding 'unknown'.")
    else:
        # Accuracy excluding 'unknown' predictions
        accuracy = accuracy_score(y_true_known, y_pred_known)
        print(f"Accuracy (excluding 'unknown' predictions): {accuracy:.2%}")

        # Classification report
        print("Baseline Predictor Classification Report:")
        print(classification_report(y_true_known, y_pred_known, labels=['male', 'female']))
        

    # Calculate the percentage of 'unknown' predictions
    unknown_count = (y_pred == 'unknown').sum()
    total_predictions = len(y_pred)
    unknown_percentage = (unknown_count / total_predictions) * 100
    print(f"Percentage of 'unknown' predictions: {unknown_percentage:.2f}%")

    # Overall message about evaluation scope
    print(f"Evaluation based on {len(y_true)} records with known true gender.")

if __name__ == "__main__":
    # Set paths relative to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, '../data/first_names_and_gender.csv')

    # Run the baseline predictor
    baseline_predictor(input_csv)
