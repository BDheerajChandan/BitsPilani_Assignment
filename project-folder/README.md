# Employee Attrition Prediction – ML Assignment 2

## a. Problem statement
Predict whether an employee will leave the company (attrition) from demographic, job and workplace features, by comparing five classification models and deploying them in an interactive Streamlit app.

## b. Dataset description
- Source: <Kaggle link>
- Instances: <rows> | Features: <n> | Target: `<target>` (binary: <classes>)
- Preprocessing: ID/constant columns dropped, median/mode imputation, standard scaling for numeric, one-hot encoding for categorical, 80/20 stratified split (seed 42).

## c. GitHub Repository Link
<repo url>

## d. Models used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| **Overall winner** | |

## Run locally
pip install -r requirements.txt
python train.py
streamlit run app.py

## Live app
<streamlit url>