import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the Breast Cancer dataset
df = pd.read_csv("CSV\B.cancer.csv")

# Define features and target variable
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score
accuracy = rf.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save the model as a pickle file
with open("B_cancer_model.pkl", "wb") as f:
    pickle.dump(rf, f)
