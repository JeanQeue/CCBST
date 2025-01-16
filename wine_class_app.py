import streamlit as st
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load the Wine dataset
data = load_wine()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Train Decision Tree model
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

# Streamlit app
st.title("Welcome to Johnfredrick's Wine Classification App")

# User input
st.header("Enter Wine Features")
user_input = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}", value=0.0)
    user_input.append(value)

# Adding the submit button
if st.button("Submit"):
    # Checking if user input is provided
    if any(user_input):
        # Standardize user input
        user_input = scaler.transform([user_input])

        # Predictions
        log_reg_pred = log_reg.predict(user_input)
        tree_pred = tree_clf.predict(user_input)

        # Displaying the predictions
        st.write(f"Logistic Regression Prediction: {target_names[log_reg_pred[0]]}")
        st.write(f"Decision Tree Prediction: {target_names[tree_pred[0]]}")
    else:
        # Display blank predictions if no input is provided
        st.write("Logistic Regression Prediction:  ")
        st.write("Decision Tree Prediction:  ")