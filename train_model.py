import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("student_data.csv")

# Clean target
data["Result"] = data["Result"].str.strip().str.capitalize()
data["Result"] = data["Result"].map({"Fail": 0, "Pass": 1})

# Features & target
X = data[["Study_Hours", "Attendance", "Assignment marks", "Previous_Marks"]]
y = data["Result"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Save model + accuracy
joblib.dump(model, "model.pkl")
joblib.dump(accuracy, "accuracy.pkl")

# Confusion matrix graph
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("static/confusion_matrix.png")
plt.close()

print("Model training completed successfully!")