import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load data
student_data = pd.read_csv("student_data.csv")
student_data["Result"] = student_data["Result"].str.strip().str.capitalize().map({"Fail": 0, "Pass": 1})

x = student_data[["Study_Hours", "Attendance", "Assignment marks", "Previous_Marks"]]
y = student_data["Result"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LogisticRegression()
model.fit(x_train, y_train)

# Create graph
plt.figure(figsize=(6,5))

pass_students = student_data[student_data["Result"] == 1]
fail_students = student_data[student_data["Result"] == 0]

plt.scatter(
    pass_students["Study_Hours"],
    pass_students["Previous_Marks"],
    color="green",
    label="Pass",
    s=80
)

plt.scatter(
    fail_students["Study_Hours"],
    fail_students["Previous_Marks"],
    color="red",
    label="Fail",
    s=80
)

plt.xlabel("Study Hours", fontsize=12)
plt.ylabel("Previous Marks", fontsize=12)

plt.title("Student Performance Analysis", fontsize=14)

plt.grid(True)

plt.legend()

plt.savefig("static/graph.png")
plt.close()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    study = float(request.form['study'])
    attendance = float(request.form['attendance'])
    assignment = float(request.form['assignment'])
    marks = float(request.form['marks'])

    sample = pd.DataFrame([[study, attendance, assignment, marks]], columns=["Study_Hours", "Attendance", "Assignment marks", "Previous_Marks"])

    prediction = model.predict(sample)

    probability = model.predict_proba(sample)[0][prediction[0]]

    if prediction[0] == 1:
        result = "Prediction: PASS"
        result_class = "pass"
        recommendation = "Student performance is satisfactory."
    else:
        result = "Prediction: FAIL"
        result_class = "fail"
        recommendation = "Improve study hours and attendance."

    return render_template(
        'index.html',
        prediction_text=result,
        probability=round(probability * 100, 2),
        recommendation=recommendation,
        result_class=result_class,
        
        study=study,
        attendance=attendance,
        assignment=assignment,
        marks=marks
    )
    
if __name__ == "__main__":
    app.run(debug=True)