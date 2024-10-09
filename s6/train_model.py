import pandas as pd                           # For handling data (think of it like a tool to open and work with Excel files)
from sklearn.model_selection import train_test_split  # To split our data into learning data and test data
from sklearn.ensemble import RandomForestRegressor    # Our smart prediction model
import joblib                                      # A tool to save the trained model

# Step 1: Load the data (the 'jobs_in_data.csv' file is like a spreadsheet with job information)
data = pd.read_csv('jobs_in_data.csv')

# Step 2: Select the important columns that will help us predict salaries
selected_features = ['job_title', 'job_category', 'experience_level', 'company_location']  # These are the things that affect salary
X = data[selected_features]  # This is our input data (features)
y = data['salary_in_usd']    # This is what we want to predict (the salary)

# Step 3: Turn text columns (like job titles) into numbers (so the computer can understand them)
X = pd.get_dummies(X, drop_first=True)  # Converts words into numbers that the model can learn from

# Step 4: Split the data into learning data (train) and testing data (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% for learning, 20% for testing

# Step 5: Create the RandomForest model (our smart predictor)
model = RandomForestRegressor(n_estimators=100, random_state=42)  # 100 trees will "vote" to predict salary

# Step 6: Train the model (we show it examples from the training data)
model.fit(X_train, y_train)

# Step 7: Save the trained model to a file (so we don’t have to teach it again)
joblib.dump(model, 'salary_prediction_model.pkl')  # This file contains the learned patterns to predict salaries
