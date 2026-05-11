import pandas as pd
import numpy as np
import random

data = []

for i in range(500):

    # realistic random values
    study = np.random.randint(0, 10)          
    attendance = np.random.randint(30, 100)   
    assignment = np.random.randint(20, 100)   
    previous = np.random.randint(20, 100)     

    # scoring logic (important for realism)
    score = (
        study * 6 +
        attendance * 0.4 +
        assignment * 0.3 +
        previous * 0.3
    )

    # add randomness (VERY IMPORTANT for realism)
    noise = random.randint(-15, 15)
    score += noise

    # final label
    result = "Pass" if score >= 60 else "Fail"

    data.append([study, attendance, assignment, previous, result])

# create dataframe
df = pd.DataFrame(data, columns=[
    "Study_Hours",
    "Attendance",
    "Assignment marks",
    "Previous_Marks",
    "Result"
])

# save file
df.to_csv("student_data.csv", index=False)

print("500-row dataset generated successfully!")