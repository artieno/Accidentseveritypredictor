# create a linear regression model for analyzing road accident severity using the relevant dataset related to
# the scenario. Please specify the dependent variable (the variable you want to predict) and the independent 
# variables (the factors that influence the accident severity). After creating the model, save it for future use
# Then, provide an example of using the model to predict accident severity for a hypothetical set of independent 
# variables, and explain how such a model could be beneficial for traffic accident analysis and prevention in 
# underdevolped countries. Add all relevant screen shots as well from your program. Also share the URL of your 
# GITHUB(Where you have uploaded your work) so that I can simulate the same.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv('accident_dataset.csv')

# Perform one-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['Weather', 'Road Condition', 'Time of Day', 'Vehicle Type', 'Traffic Signals', 'Pedestrians'], drop_first=True)

# Convert Accident Severity to numerical values (optional, if not already)
severity_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['Accident Severity'] = df['Accident Severity'].map(severity_mapping)

# Split data into features and target variable
X = df.drop('Accident Severity', axis=1)
y = df['Accident Severity']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("R-squared: %.2f" % r2_score(y_test, y_pred))

# Save the model for future use (optional)
joblib.dump(model, 'accident_severity_model.pkl')

# Define a hypothetical set of independent variables for prediction
new_data = pd.DataFrame(columns=X_train.columns)

# Check the number of columns in new_data
print("Number of columns in new_data:", len(new_data.columns))

# Add data to new_data with the appropriate feature values
new_row_data = [1, 0, 0,  # Weather: Clear
                1, 0, 0,  # Road Condition: Dry
                1, 0, 0,  # Time of Day: Afternoon
                0]       # Vehicle Type: Car
                
# Check the length of new_row_data
print("Length of new_row_data:", len(new_row_data))

# Ensure the length of new_row_data matches the number of columns in new_data
if len(new_row_data) == len(new_data.columns):
    new_data.loc[0] = new_row_data
    # Making prediction using the trained model
    predicted_severity = model.predict(new_data)
    print("Predicted accident severity:", predicted_severity)
else:
    print("Error: The number of elements in new_row_data does not match the number of columns in new_data.")