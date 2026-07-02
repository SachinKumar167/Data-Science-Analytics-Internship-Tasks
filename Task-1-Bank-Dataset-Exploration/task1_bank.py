import pandas as pd

# Load the dataset
bank = pd.read_csv("bank.csv")

# Display dataset information
print("Shape:")
print(bank.shape)

print("\nColumns:")
print(bank.columns)

print("\nFirst 5 Rows:")
print(bank.head())

print("\nDataset Information:")
print(bank.info())

print("\nMissing Values:")
print(bank.isnull().sum())