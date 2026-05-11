from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model & accuracy
model = joblib.load("model.pkl")
accuracy = joblib.load("accuracy.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    study = float(request.form['study'])
    attendance = float(request.form['attendance'])
    assignment = float(request.form['assignment'])
    marks = float(request.form['marks'])

    input_data = pd.DataFrame([[study, attendance, assignment, marks]],
                              columns=["Study_Hours", "Attendance", "Assignment marks", "Previous_Marks"])

    prediction = model.predict(input_data)[0]
    probability = max(model.predict_proba(input_data)[0]) * 100

    if prediction == 1:
        result = "Prediction: PASS"
        result_class = "pass"
        recommendation = "Student performance is satisfactory."
    else:
        result = "Prediction: FAIL"
        result_class = "fail"
        recommendation = "Improve study hours and attendance."
    
    return render_template(
    "index.html",
    prediction_text=result,
    probability=round(probability, 2),
    recommendation=recommendation,
    result_class=result_class,
    accuracy=round(accuracy * 100, 2),
    study=study,
    attendance=attendance,
    assignment=assignment,
    marks=marks
)

if __name__ == "__main__":
    app.run(debug=True)