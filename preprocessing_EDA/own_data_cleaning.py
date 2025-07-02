import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/chicswldrg/Desktop/UZH 2024:25/Summer School Juli 2025/UCL-Summer-School-Intro-to-AI-Course-1/datasets/ai_adoption_dataset.csv')

###################### EXPLORING THE DATA #######################################################################################################################

# first 5 rows
print(f"The first 5 rows of the dataset:")
print(df.head())
print("\n")

# basic information
print("Basic information about the dataset:")
print(df.info())
print("\n")

# statistics for numerical columns
print("Statistics for numerical columns:")
print(df.describe())
print("\n")

# get all column names
print("Column names in our dataset:")
print(df.columns.tolist())
print("\n")

print("Data types for each column:")
print(df.dtypes)
print("\n")

##################### CHECK FOR DUPLICATES, NULL VALUES, UNIQUE VALUES, OUTLIERS ###########################################################################################################

# check for duplicates, and remove
def remove_duplicates():
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    if duplicates > 0:
        print(f"Removed {duplicates} duplicate rows.\n")
    else:
        print("No duplicate rows found.\n")

def check_null_values():
    null_counts = df.isnull().sum()
    print("Number of null values in each column:")
    print(null_counts[null_counts > 0])
    print("\n")

    columns_with_missing = df.columns[df.isnull().any()].tolist()
    print("Columns with missing values:")
    print(columns_with_missing)

    return null_counts

def find_unique_values():
    for column in df.columns:
        if df[column].dtype == 'object':
            unique_values = df[column].unique()
            print(f"Unique values in '{column}': {unique_values}\n")

def handle_missing_values():
    df_filled = df.fillna('Unknown')
    print(f"\nAfter filling missing values with 'Unknown':")
    print(df_filled.isnull().sum().sum(), "missing values remain\n")
    #print(f"Missing values: {df.isnull().sum().sum()}") # final check
# alternative option to handle missing values: remove

##################### VISUALISATION OF OUTLIERS ############################################################################################################

# Create box plots for numerical columns to spot outliers
numerical_cols = df.select_dtypes(include=[np.number]).columns

print("Creating box plots to spot outliers...")
print("Look for dots outside the boxes - these might be outliers!")

# Create a plot for each numerical column
def boxplot_for_numerical_columns():
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        plt.boxplot(df[col].dropna())
        plt.title(f'Box Plot: {col}')
        plt.ylabel(col)
        plt.show()

############ FINISH CLEANING DATA FILE ###########################################################################################################################

def clean_data_file():
    print(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns\n")
    remove_duplicates()
    check_null_values()
    find_unique_values()
    handle_missing_values()

def visualise_data_file():
    boxplot_for_numerical_columns()

clean_data_file()
visualise_data_file()

# Save the cleaned DataFrame to a new CSV file
df.to_csv('/Users/chicswldrg/Desktop/UZH 2024:25/Summer School Juli 2025/UCL-Summer-School-Intro-to-AI-Course-1/datasets/ai_adoption_dataset_cleaned.csv', index=False)
print("\nCleaned dataset saved to 'ai_adoption_dataset_cleaned.csv'.")