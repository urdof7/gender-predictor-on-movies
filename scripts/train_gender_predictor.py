"""
Author: Brayden Miller
Date: November 22, 2024

Project: Gender Prediction Using Machine Learning of Movie Dataset

Description:
This script trains a custom gender prediction model using first names and evaluates 
its performance against a baseline predictor. The custom predictor uses character-level 
TF-IDF vectorization and a Multinomial Naive Bayes classifier, while the baseline predictor 
utilizes the `gender_guesser` library. The script compares the accuracy and classification 
performance of both models and provides detailed evaluation metrics.

Acknowledgments:
This script was developed by Brayden Miller with assistance from ChatGPT.
"""

import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import gender_guesser.detector as gender

def train_and_evaluate(input_csv):
    """
    Trains a gender prediction model and evaluates its performance.
    Compares the custom model with the baseline predictor.
    """
    # Load the data
    df = pd.read_csv(input_csv)

    # Filter to records with known true genders
    df = df[df['true_gender'].isin(['male', 'female'])].copy()

    # Drop rows with missing first names
    df = df.dropna(subset=['first_name'])

    # Reset index after filtering
    df = df.reset_index(drop=True)

    # Prepare features and labels
    X_names = df['first_name']
    y = df['true_gender']

    # Split the data (80% training, 20% testing)
    X_train_names, X_test_names, y_train, y_test = train_test_split(
        X_names, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature extraction using character-level TF-IDF
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    X_train = vectorizer.fit_transform(X_train_names)
    X_test = vectorizer.transform(X_test_names)

    # Train a Multinomial Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Predict on the test set using the custom predictor
    y_pred_custom = clf.predict(X_test)

    # Evaluate the custom predictor
    custom_accuracy = accuracy_score(y_test, y_pred_custom)
    # Print only the classification report
    print("Custom Predictor Classification Report:")
    print(classification_report(y_test, y_pred_custom, labels=['male', 'female']))

    # Evaluate baseline predictor on the same test set
    baseline_overall_accuracy, baseline_accuracy_excl_unknown, y_pred_baseline_known, y_true_baseline_known = evaluate_baseline(X_test_names, y_test)

    # Print the Baseline Predictor Classification Report
    print("Baseline Predictor Classification Report:")
    print(classification_report(y_true_baseline_known, y_pred_baseline_known, labels=['male', 'female']))

    # Compare accuracies
    print("\nAccuracy Comparison of Predictors:")
    print(f"Baseline Predictor Overall Accuracy (including 'unknown'): {baseline_overall_accuracy:.2%}")
    print(f"Baseline Predictor Accuracy (excluding 'unknown'): {baseline_accuracy_excl_unknown:.2%}")
    print(f"Custom Predictor Accuracy: {custom_accuracy:.2%}")

def evaluate_baseline(X_test_names, y_test):
    """
    Predicts gender using the baseline predictor and evaluates performance.
    Returns both overall accuracy and accuracy excluding 'unknown' predictions,
    along with predictions and true labels for known predictions.
    """
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

    # Predict gender using the baseline predictor
    y_pred = X_test_names.apply(map_predicted_gender)

    # Extract true and predicted genders
    y_true = y_test

    # Calculate overall accuracy including 'unknown' predictions
    overall_accuracy = accuracy_score(y_true, y_pred)

    # Exclude 'unknown' predictions for accuracy calculation
    mask_known_predictions = y_pred != 'unknown'
    y_true_known = y_true[mask_known_predictions]
    y_pred_known = y_pred[mask_known_predictions]

    if y_true_known.empty:
        accuracy_excl_unknown = None
    else:
        # Accuracy excluding 'unknown' predictions
        accuracy_excl_unknown = accuracy_score(y_true_known, y_pred_known)

    return overall_accuracy, accuracy_excl_unknown, y_pred_known, y_true_known

if __name__ == "__main__":
    # Set path to the CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, '../data/first_names_and_gender.csv')

    # Run the training and evaluation
    train_and_evaluate(input_csv)