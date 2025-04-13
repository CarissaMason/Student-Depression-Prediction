# Student Depression Prediction

This project uses machine learning (logistic regression) to predict the likelihood of depression among students based on academic, personal, and lifestyle factors.

## Dataset

Source: [Kaggle - Student Depression Dataset](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset)

## Features Used
- Gender
- Age
- Academic Pressure
- Work Pressure
- Job Satisfaction
- Work/Study Hours
- Financial Stress
- Suicidal Thoughts (Yes/No)

## Model

- Model: Logistic Regression (L2 regularization)
- Accuracy: ~84%
- Tuned using GridSearchCV
- Custom thresholding applied for high-risk detection

## Ethical Considerations

This model is for **educational purposes only**. It is not a diagnostic tool and should never replace professional mental health care.

## Usage

Clone the repo and open the notebook:
```bash
git clone https://github.com/yourusername/student-depression-prediction.git
cd student-depression-prediction
jupyter notebook student_depression_model.ipynb
