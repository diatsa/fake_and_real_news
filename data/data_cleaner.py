"""
Fake News Dataset Preparation
This script loads fake and true news datasets, merges them with labels,
and splits the data into training and testing sets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the datasets
print("Loading datasets...")
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Display basic information about the datasets
print(f"\nFake news dataset shape: {fake_df.shape}")
print(f"True news dataset shape: {true_df.shape}")

# Check the column headers
print("\nFake.csv columns:", fake_df.columns.tolist())
print("True.csv columns:", true_df.columns.tolist())

# Display first few rows to understand the data
print("\nFirst few rows of Fake.csv:")
print(fake_df.head())
print("\nFirst few rows of True.csv:")
print(true_df.head())

# Add label columns
# 0 = Fake, 1 = True
fake_df['label'] = 0
true_df['label'] = 1

print(f"\nAdded labels \nFake: 0, True: 1")

# Check missing values
print("\nMissing values in Fake.csv:")
print(fake_df.isna().sum()[fake_df.isna().sum() > 0])

print("\nMissing values in True.csv:")
print(true_df.isna().sum()[true_df.isna().sum() > 0])

# Merge the datasets
merged_df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)
print(f"\nMerged dataset shape: {merged_df.shape}")

# Drop redundant columns
merged_df = merged_df.drop(columns=["subject", "date"])

# Merge title and text into a single column
merged_df['content'] = (
    merged_df['title'].str.strip() + ' - ' + merged_df['text'].str.strip()
).str.strip()

# Drop original columns after merge
merged_df = merged_df.drop(columns=['title', 'text'])

# Inspect result
print(merged_df.head())
# Check class distribution
print("\nClass distribution:")
print(merged_df['label'].value_counts())

# Shuffle the data to ensure random distribution
# This is important because we concatenated all fake news first, then all true news
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
print("\nData shuffled randomly")

# Split the data into training (80%) and testing (20%) sets
# random_state ensures reproducibility
# stratify ensures both sets have similar proportions of fake/true news
X = merged_df.drop('label', axis=1)  # Features (all columns except label)
y = merged_df['label']  # Target (the label column)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20,  # 20% for testing, 80% for training
    random_state=42,  # For reproducibility
    stratify=y  # Maintains class distribution in both sets
)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# Verify class distribution in both sets
print("\nTraining set class distribution:")
print(y_train.value_counts())
print("\nTesting set class distribution:")
print(y_test.value_counts())

# Save the full merged dataset
merged_df.to_csv('merged_news_data.csv', index=False)
print("Saved merged_news_data.csv")

print("\nData preparation complete")