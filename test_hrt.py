import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df =pd.read_csv("CSV\heart_disease_data.csv")

X = df.drop('target', axis=1)
y=df['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, rf.predict(X_test))
print("Model Accuracy:", accuracy)

with open("hrt_model.pkl", "wb") as f:
    pickle.dump(rf, f)