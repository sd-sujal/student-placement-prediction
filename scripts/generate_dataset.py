import numpy as np
import pandas as pd

np.random.seed(42)
N = 600

cgpa = np.round(np.random.normal(7.2, 0.8, N), 2)
cgpa = np.clip(cgpa, 5.0, 10.0)

backlogs = np.random.choice([0, 1, 2, 3, 4, 5], N, p=[0.45, 0.25, 0.15, 0.08, 0.05, 0.02])

internships = np.random.choice([0, 1, 2, 3, 4], N, p=[0.35, 0.30, 0.20, 0.10, 0.05])

projects = np.random.randint(1, 7, N)
certifications = np.random.randint(0, 6, N)

aptitude_score = np.random.randint(40, 101, N)
communication_score = np.random.randint(40, 101, N)
coding_skill = np.random.randint(40, 101, N)

placement_score = (
    0.35 * cgpa +
    0.25 * (coding_skill / 10) +
    0.15 * (aptitude_score / 10) +
    0.10 * (communication_score / 10) +
    0.10 * internships -
    0.20 * backlogs
)

probability = 1 / (1 + np.exp(-placement_score + 3))
placed = (probability >= 0.5).astype(int)

salary = (
    1.5 * cgpa +
    0.04 * coding_skill +
    0.6 * internships +
    0.3 * projects -
    0.5 * backlogs +
    np.random.normal(0, 0.7, N)
)

salary = np.round(np.clip(salary, 3.0, 12.0), 2)
salary[placed == 0] = 0.0

df = pd.DataFrame({
    "cgpa": cgpa,
    "backlogs": backlogs,
    "internships": internships,
    "projects": projects,
    "certifications": certifications,
    "aptitude_score": aptitude_score,
    "communication_score": communication_score,
    "coding_skill": coding_skill,
    "placed": placed,
    "salary_lpa": salary
})

df.to_csv("student_data.csv", index=False)
print("Dataset generated: student_data.csv")
