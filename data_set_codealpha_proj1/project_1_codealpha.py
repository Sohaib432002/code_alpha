import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# List of encodings to try
encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
df = None

# Try reading the file with different encodings
for enc in encodings:
    try:
        df = pd.read_csv("/content/Most Streamed Spotify Songs 2024.csv", encoding=enc)
        print(f'Successfully read the file with encoding: {enc}')
        break
    except UnicodeDecodeError as e:
        print(f'Failed to read the file with encoding: {enc}')
        print(e)

# Check if the dataframe was read successfully
if df is not None:
    print('The dataset Sample :')
    print(df.sample(10).to_string())
else:
    print('Failed to read the file with all attempted encodings.')

print('The Total Rows and Columns of that dataset are:', df.shape)

print('The Basic Information of the Dataset is:')
print(df.info())

print('The Total Null Values in the dataset is:', df.isnull().sum().sum())

print('The Null Values in the Columns is:\n', df.isnull().sum())

# Calculate the percentage of null values per column
null_data_per = df.isnull().sum() / df.shape[0] * 100

# Identify columns with more than 10% null values
columns_with_many_nulls = null_data_per[null_data_per > 10].index

# Print and drop columns with many null values
print('The dataset shape before removing the null columns', df.shape)
df.drop(columns=columns_with_many_nulls, axis=1, inplace=True)
print('The dataset shape after removing the null columns', df.shape)

print(df.info())
print('The null values in the columns is:\n', df.select_dtypes(include=['int', 'float', 'object']).isnull().sum())

# Convert columns with commas to float and handle date conversion
df['YouTube Views'] = df['YouTube Views'].str.replace(',', '').astype(float)
df['Spotify Streams'] = df['Spotify Streams'].str.replace(',', '').astype(float)
df['Spotify Playlist Count'] = df['Spotify Playlist Count'].str.replace(',', '').astype(float)
df['Spotify Playlist Reach'] = df['Spotify Playlist Reach'].str.replace(',', '').astype(float)
df['YouTube Likes'] = df['YouTube Likes'].str.replace(',', '').astype(float)
df['Release Dates'] = pd.to_datetime(df['Release Date'])
df['Release Date'] = df['Release Dates'].dt.date
df['Release Month'] = df['Release Dates'].dt.month
df['Release Year'] = df['Release Dates'].dt.year

# Drop the 'Release Dates' column
df.drop(columns=['Release Dates'], inplace=True)

# Drop rows with any null values
df.dropna(inplace=True)

# Visualize distributions and boxplots for numeric columns
for column in df.select_dtypes(include=['int', 'float']).columns:
    sns.histplot(df[column], kde=True)
    plt.title(column)
    plt.show()

for column in df.select_dtypes(include=['int', 'float']).columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(column)
    plt.show()


# Detect and handle outliers using IQR method
def detect_outliers_iqr(df, columns):
    outliers = {}
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[column] = (lower_bound, upper_bound)
    return outliers


def remove_outliers(df, outliers):
    df_cleaned = df.copy()
    for column, (lower, upper) in outliers.items():
        df_cleaned = df_cleaned[(df_cleaned[column] >= lower) & (df_cleaned[column] <= upper)]
    return df_cleaned


# Function to apply cube root transformation
def apply_cbrt_transformation(df):
    df_cbrt_transformed = df.copy()
    for column in df.select_dtypes(include=['int', 'float']).columns:
        df_cbrt_transformed[column] = np.cbrt(df[column])
    return df_cbrt_transformed


# Function to apply Winsorization
def apply_winsorization(df):
    df_winsorized = df.copy()
    for column in df.select_dtypes(include=['int', 'float']).columns:
        lower_limit = df[column].quantile(0.05)
        upper_limit = df[column].quantile(0.95)
        df_winsorized[column] = np.clip(df[column], lower_limit, upper_limit)
    return df_winsorized


# Function to visualize data
def visualize_data(df, transformation_name):
    print(f'\n\n\n Applying {transformation_name} Transformation')
    for column in df.select_dtypes(include=['int', 'float']).columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot of {column} after {transformation_name} Transformation')
        plt.show()


# Function to summarize data
def summarize_data(df, transformation_name):
    print(f'\n\n\n Summary statistics after {transformation_name} Transformation')
    summary = df.describe(include=[np.number])
    print(summary)


# Main function to handle transformations and visualizations
def process_data(df):
    # Detect and remove outliers
    numeric_columns = df.select_dtypes(include=['int', 'float']).columns
    outliers = detect_outliers_iqr(df, numeric_columns)
    df_cleaned = remove_outliers(df, outliers)

    # Cube root transformation
    df_cbrt_transformed = apply_cbrt_transformation(df_cleaned)
    summarize_data(df_cbrt_transformed, 'Cube Root')
    visualize_data(df_cbrt_transformed, 'Cube Root')

    # Winsorization transformation
    df_winsorized = apply_winsorization(df_cleaned)
    summarize_data(df_winsorized, 'Winsorization')
    visualize_data(df_winsorized, 'Winsorization')

    return df_cbrt_transformed, df_winsorized


# Process the data and apply transformations
df_cbrt_transformed, df_winsorized = process_data(df.select_dtypes(include=['int', 'float']))

for column in df_winsorized.select_dtypes(include=['int', 'float']).columns:
    sns.histplot(df[column], kde=True)
    plt.title(column)
    plt.show()

# Prepare string columns
df_str_columns = df[['Track', 'Album Name', 'Artist', 'ISRC']]

# Combine the data
df_combined = pd.concat([df_str_columns, df_winsorized.reset_index(drop=True)], axis=1)

# Save the combined dataset

# print('Combined dataset saved to combined_data.csv')

df_combined['Repeatable'] = ((df_combined['Spotify Streams'] > 50000000) * 0.3 +
                             (df_combined['YouTube Views'] > 50000000) * 0.3 +
                             (df_combined['Spotify Playlist Count'] > 1000) * 0.2 +
                             (df_combined['Spotify Playlist Reach'] > 1000000) * 0.2) > 0.5

# Convert Boolean to integer (1 for True, 0 for False)
df_combined['Repeatable'] = df_combined['Repeatable'].astype(int)

print(df_combined[['Track', 'Repeatable']])

print(df_combined.head())

df_combined['Repeatable'].value_counts().plot(kind='bar')

plt.title('Repeatable')
plt.show()
# df_combined.to_csv('combined_data.csv', index=False)


y = df_combined['Repeatable']
X = df_combined.drop(columns=['Repeatable'])

from sklearn.preprocessing import LabelEncoder

# Initialize label encoders
track_encoder = LabelEncoder()
album_encoder = LabelEncoder()
artist_encoder = LabelEncoder()

# Fit and transform the categorical data
df_combined['Track_encoded'] = track_encoder.fit_transform(df_combined['Track'])
df_combined['Album Name_encoded'] = album_encoder.fit_transform(df_combined['Album Name'])
df_combined['Artist_encoded'] = artist_encoder.fit_transform(df_combined['Artist'])

# Drop original categorical columns
df_encoded = df_combined.drop(columns=['Track', 'Album Name', 'Artist'])

print(df_encoded)

print(df_encoded.columns)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print('The Null values are :')
print(df_encoded.isnull().sum())
df_encoded.dropna(inplace=True)
print('The Null values are :')
print(df_encoded.isnull().sum())
df_encoded = pd.get_dummies(df_combined, columns=['Track', 'Album Name', 'Artist'])
X = df_encoded.drop(columns=['Repeatable', 'ISRC'])

y = df_encoded['Repeatable']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import numpy as np

# Check for NaNs in the original features
print("NaNs in X_train:", np.any(np.isnan(X_train)))
print("NaNs in X_test:", np.any(np.isnan(X_test)))

# Check for NaNs in the scaled features
print("NaNs in X_train_scaled:", np.any(np.isnan(X_train_scaled)))
print("NaNs in X_test_scaled:", np.any(np.isnan(X_test_scaled)))

from sklearn.impute import SimpleImputer

# Initialize the imputer
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent', etc.

# Fit and transform the imputer on training data
X_train_imputed = imputer.fit_transform(X_train)

# Transform the test data
X_test_imputed = imputer.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# List of models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC()
}


# Function to train and evaluate models
def evaluate_models(X_train, X_test, y_train, y_test, models):
    results = {}
    for name, model in models.items():
        print(f'\nEvaluating {name}...')
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        results[name] = {
            'Accuracy': accuracy,
            'Classification Report': report,
            'Confusion Matrix': conf_matrix
        }

        # Print results
        print(f'Accuracy: {accuracy}')
        print('Classification Report:\n', report)
        print('Confusion Matrix:\n', conf_matrix)

    return results


# Evaluate all models
results = evaluate_models(X_train_imputed, X_test_imputed, y_train, y_test, models)
