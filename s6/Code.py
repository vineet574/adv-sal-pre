import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import numpy as np

# Step 1: Load Data
data = pd.read_csv('jobs_in_data.csv')

# Display basic information about the dataset
print("First few rows of the dataset:")
print(data.head())

print("\nInformation about columns, data types, and null values:")
print(data.info())

print("\nStatistical summary of numerical columns:")
print(data.describe())

# Step 2: Feature Engineering

# Experience Level Encoding
experience_mapping = {'Entry-level': 0, 'Mid-level': 1, 'Senior': 2, 'Executive': 3}
data['experience_level_encoded'] = data['experience_level'].map(experience_mapping)

# Company Size Encoding
company_size_mapping = {'S': 1, 'M': 2, 'L': 3}
data['company_size_encoded'] = data['company_size'].map(company_size_mapping)

# Remote Work Indicator
data['remote_work'] = data['work_setting'].apply(lambda x: 1 if x == 'Remote' else 0)

# Adding 'work_year' to capture temporal trends
data['work_year_encoded'] = data['work_year'] - data['work_year'].min()

# Seniority Indicator from Job Titles (e.g., 'Junior', 'Senior', 'Lead')
def seniority_from_job_title(title):
    if 'junior' in title.lower():
        return 1
    elif 'senior' in title.lower() or 'lead' in title.lower():
        return 2
    elif 'manager' in title.lower() or 'director' in title.lower():
        return 3
    else:
        return 0

data['seniority'] = data['job_title'].apply(seniority_from_job_title)

# Handle missing data with SimpleImputer
categorical_features = ['job_category', 'company_location', 'employment_type']
numerical_features = ['experience_level_encoded', 'company_size_encoded', 'remote_work', 'work_year_encoded', 'seniority']

# Build ColumnTransformer to handle encoding and imputations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Step 3: Define Target Variable and Features
X = data[categorical_features + numerical_features]
y = data['salary_in_usd']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training and Hyperparameter Tuning

# Build a pipeline for training
model = RandomForestRegressor(random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', RFE(estimator=model, n_features_to_select=10)),
    ('regressor', model)
])

# Define parameter grid for GridSearch
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [10, 20],
    'regressor__min_samples_split': [2, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')

# Train the model
grid_search.fit(X_train, y_train)

# Best parameters found by GridSearch
print(f"Best parameters: {grid_search.best_params_}")

# Predictions
y_pred = grid_search.predict(X_test)

# Step 5: Model Evaluation

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R-squared: {r2}")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Step 6: Data Visualization

# Residual Analysis
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred - y_test, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('True Values')
plt.ylabel('Residuals')
plt.show()

# Feature Importance (Tree-based model)
importances = grid_search.best_estimator_.named_steps['regressor'].feature_importances_
features = preprocessor.transformers_[1][1].get_feature_names_out().tolist() + numerical_features
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.show()


