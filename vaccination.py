#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


# In[3]:


df=pd.read_csv('h1n1_vaccine_prediction.csv')


# In[4]:



# Separate numeric and non-numeric columns
numeric_cols = df.select_dtypes(include='number').columns
non_numeric_cols = df.select_dtypes(exclude='number').columns

# Impute missing values for numeric columns
numeric_imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

# Impute missing values for non-numeric columns
non_numeric_imputer = SimpleImputer(strategy='most_frequent')
df[non_numeric_cols] = non_numeric_imputer.fit_transform(df[non_numeric_cols])

# Convert categorical variables to numerical format
categorical_cols = ['age_bracket', 'qualification', 'race', 'sex', 'income_level', 'marital_status', 'housing_status', 'employment', 'census_msa']
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Split the dataset into features (X) and target variable (y)
X = df_encoded.drop('h1n1_vaccine', axis=1)
y = df_encoded['h1n1_vaccine']

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
predictions = model.predict(X_valid)

# Evaluate the model
accuracy = accuracy_score(y_valid, predictions)
report = classification_report(y_valid, predictions)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')


#  our model's output aligns with the task of determining whether a person is vaccinated or not. In the classification report, we have two classes: 0 and 1, where 0 typically represents "not vaccinated" and 1 represents "vaccinated" (assuming that the positive class is the one labeled '1' in our dataset).
