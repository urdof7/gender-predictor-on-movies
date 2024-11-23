"""
Author: Brayden Miller
Date: November 22, 2024

Project: Gender Prediction Using Machine Learning of Movie Dataset

Description:
This script extracts first names and gender information from a SQLite database 
containing movie-related data. It processes the `person` table to retrieve and 
map gender information, extracts first names from full names, and saves the 
processed data into a CSV file. The script ensures data integrity by handling 
missing values, mapping gender to standardized labels (`male`, `female`), and 
removing duplicates.

Acknowledgments:
This script was developed by Brayden Miller with assistance from ChatGPT.
"""

import sqlite3
import pandas as pd
import os

def extract_first_names(db_path, output_csv):
    """
    Connects to the SQLite database, extracts full names and genders of cast members,
    obtains their first names, and saves the result to a CSV file.

    Parameters:
    - db_path: Relative path to the SQLite database file.
    - output_csv: Relative path to the output CSV file.
    """
    # Check if the database file exists
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
        return

    # Connect to the SQLite database
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return

    # Modify the SQL query to include 'gender'
    query = """
    SELECT person_id, name, gender FROM person
    """

    try:
        # Read data into a pandas DataFrame
        df_person = pd.read_sql_query(query, conn)
        print(f"Retrieved {len(df_person)} records from 'person' table.")
    except Exception as e:
        print(f"Error querying database: {e}")
        conn.close()
        return

    # Close the database connection
    conn.close()

    # Check if 'name' and 'gender' columns exist
    if 'name' not in df_person.columns or 'gender' not in df_person.columns:
        print("Required columns 'name' or 'gender' not found in 'person' table.")
        return

    # Extract first names
    df_person['first_name'] = df_person['name'].apply(extract_first_name)

    # Map gender values to 'male' and 'female' (if necessary)
    df_person['true_gender'] = df_person['gender'].apply(map_gender)

    # Remove duplicates if any
    df_person = df_person.drop_duplicates(subset=['person_id'])

    # Save to CSV
    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_person.to_csv(output_csv, index=False)
        print(f"First names and true genders extracted and saved to data/first_names_and_gender.csv")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def extract_first_name(full_name):
    """
    Extracts the first name from a full name string.

    Parameters:
    - full_name: The full name string.

    Returns:
    - The first name.
    """
    # Split the name by spaces
    if pd.isnull(full_name) or not isinstance(full_name, str):
        return ''
    parts = full_name.strip().split()
    if len(parts) == 0:
        return ''
    else:
        return parts[0]

def map_gender(gender_value):
    """
    Maps the gender value from the database to 'male' or 'female'.

    Parameters:
    - gender_value: The gender value from the database.

    Returns:
    - 'male', 'female', or None.
    """
    if pd.isnull(gender_value):
        return None
    gender_value = gender_value.strip().lower()
    if gender_value in ['male', 'm']:
        return 'male'
    elif gender_value in ['female', 'f']:
        return 'female'
    else:
        return None

if __name__ == "__main__":
    # Set paths relative to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, '../data/movies.db')
    output_csv = os.path.join(script_dir, '../data/first_names_and_gender.csv')

    # Run the extraction
    extract_first_names(db_path, output_csv)
