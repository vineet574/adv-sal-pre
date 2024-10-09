# Salary Prediction Project

This project aims to predict salaries based on various job-related features using a Random Forest Regressor model on a job dataset.

## Project Overview

The dataset contains information about job titles, categories, experience levels, employment types, and more. The goal is to predict salaries by analyzing these factors.

### Dataset Information

The dataset includes columns such as:
- `work_year`: Year of data record
- `job_title`: Title of the job
- `job_category`: Category of the job
- `salary_currency`: Currency used for salary
- `salary`: Salary in the local currency
- `salary_in_usd`: Salary converted to USD
- `experience_level`: Level of experience (Junior, Mid-level, Senior, etc.)
- `employment_type`: Type of employment
- `company_location`: Location of the company

## Project Workflow

### Data Preprocessing

- Load the job dataset from a CSV file.
- Clean the data by handling missing values and encoding categorical features using one-hot encoding.
- Split the data into features (X) and the target variable (y).

### Model Building

- Split the data into training and testing sets (80% train, 20% test).
- Utilize a Random Forest Regressor model for predicting salaries.
- Train the model using the training data.
- Evaluate model performance using R-squared and Mean Squared Error (MSE).

### Visualization

- Visualize temporal trends in salaries using line plots.
- Analyze salary distributions across job categories with box plots.
- Display scatter plots comparing actual vs. predicted salaries.
- Create histograms to illustrate the distribution of median house values.

## Tools Used

- Python
- Pandas for data manipulation
- Matplotlib for data visualization
- Scikit-learn for machine learning modeling

## Instructions

To run the project:

1. Clone or download this repository to your local machine.
2. Ensure you have Python installed along with necessary libraries (Pandas, Matplotlib, Scikit-learn).
3. Run the provided code in your Python environment to perform the salary prediction analysis.

## License
