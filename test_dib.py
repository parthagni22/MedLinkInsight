import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
df = pd.read_csv("CSV\diabetes.csv")

# Define features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, rf.predict(X_test))
print("Model Accuracy:", accuracy)

# Save the model as a pickle file
with open("Dib_model.pkl", "wb") as f:
    pickle.dump(rf, f)
