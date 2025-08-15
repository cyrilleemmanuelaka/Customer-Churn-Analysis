# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Display first few rows to understand the data
print(df.head())

# Check basic info about the dataset
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values (there's one in TotalCharges)
# Convert TotalCharges to numeric, coerce errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing TotalCharges with 0 (likely new customers)
df['TotalCharges'].fillna(0, inplace=True)

# Check unique values in each column to understand categorical variables
print("\nUnique Values:")
for column in df.columns:
    if df[column].dtype == 'object':
        print(f"{column}: {df[column].unique()}")

# Set style for plots
sns.set(style="whitegrid")

# 1. Churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title('Customer Churn Distribution')
plt.show()

# Percentage of churn
churn_percentage = df['Churn'].value_counts(normalize=True) * 100
print("\nChurn Percentage:")
print(churn_percentage)

# 2. Numeric features analysis
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
plt.figure(figsize=(12, 4))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x='Churn', y=feature, data=df)
    plt.title(f'{feature} vs Churn')
plt.tight_layout()
plt.show()

# 3. Categorical features analysis
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                       'PhoneService', 'MultipleLines', 'InternetService',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaperlessBilling', 'PaymentMethod']

plt.figure(figsize=(20, 30))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(6, 3, i)
    sns.countplot(x=feature, hue='Churn', data=df)
    plt.title(f'{feature} vs Churn')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Correlation matrix for numeric features
plt.figure(figsize=(8, 6))
corr_matrix = df[numeric_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Drop customerID as it's not useful for prediction
df.drop('customerID', axis=1, inplace=True)

# Convert Churn to binary (1 for Yes, 0 for No)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Encode categorical variables
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Split data into features (X) and target (y)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale numeric features
scaler = StandardScaler()
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

print("\nData shapes:")
print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

# 1. Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

print("\nLogistic Regression Results:")
print(classification_report(y_test, y_pred_logreg))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logreg))
print(f"Accuracy: {accuracy_score(y_test, y_pred_logreg):.2f}")

# 2. Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Results:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")

# Feature importance from Random Forest
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Top 15 Important Features')
plt.show()

# Save cleaned data for Power BI
df.to_csv('telco_churn_cleaned.csv', index=False)

# Create aggregated data for visualizations
# 1. Churn by tenure groups
df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], 
                          labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])
churn_by_tenure = df.groupby('tenure_group')['Churn'].mean().reset_index()
churn_by_tenure.to_csv('churn_by_tenure.csv', index=False)

# 2. Churn by contract type
churn_by_contract = df.groupby('Contract')['Churn'].mean().reset_index()
churn_by_contract.to_csv('churn_by_contract.csv', index=False)

# 3. Churn by internet service
churn_by_internet = df.groupby('InternetService')['Churn'].mean().reset_index()
churn_by_internet.to_csv('churn_by_internet.csv', index=False)

# 4. Monthly charges distribution by churn
monthly_charges_stats = df.groupby('Churn')['MonthlyCharges'].describe()
monthly_charges_stats.to_csv('monthly_charges_stats.csv')

print("\nData prepared for Power BI dashboard saved to CSV files.")