import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Load the trained model
model = joblib.load('salary_prediction_model.pkl')

# Load the dataset
data = pd.read_csv('jobs_in_data.csv')

st.title("Salary Prediction Dashboard")

# User input for prediction
experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=0)
education = st.selectbox("Education Level", options=["High School", "Bachelor's", "Master's", "PhD"])
user_email = st.text_input("Your Email Address")

# Create a button to make predictions
if st.button("Predict Salary"):
    # Prepare input for model
    input_data = [[experience, education]]
    prediction = model.predict(input_data)
    
    st.success(f"The predicted salary is ${prediction[0]:,.2f}")

    # Send email notification
    if user_email:
        send_email(user_email, prediction[0])
        st.success("Email notification sent!")

# Job Recommendation
st.subheader("Job Recommendations")
recommended_jobs = data[data['experience_level'] == education]  # Simple filter based on experience level
if not recommended_jobs.empty:
    st.write("Based on your education level and experience, we recommend the following jobs:")
    st.write(recommended_jobs[['job_title', 'salary']])
else:
    st.write("No job recommendations available for your criteria.")

# Data Visualization
st.subheader("Data Visualization")

# Salary Distribution
st.subheader("Salary Distribution")
plt.figure(figsize=(10, 6))
sns.histplot(data['salary'], bins=30, kde=True)
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Frequency')
st.pyplot(plt)

# Job Count by Experience Level
st.subheader("Job Count by Experience Level")
job_count = data['experience_level'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=job_count.index, y=job_count.values)
plt.title('Job Count by Experience Level')
plt.xlabel('Experience Level')
plt.ylabel('Count')
st.pyplot(plt)

# Display data for visualization
st.subheader("Job Data")
st.write(data)

# Function to send email notification
def send_email(to_email, salary_prediction):
    from_email = "your_email@example.com"  # Replace with your email
    from_password = "your_password"  # Replace with your email password

    # Create the email content
    subject = "Salary Prediction Notification"
    body = f"Your predicted salary is ${salary_prediction:,.2f}."

    # Set up the email server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, from_password)

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Send the email
    server.send_message(msg)
    server.quit()



