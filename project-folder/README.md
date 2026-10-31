# Employee Attrition Prediction – ML Assignment 2

M.Tech (AIML) | Machine Learning | BITS Pilani WILP

## a. Problem statement
Predict whether an employee will leave the company (`Attrition` = Left / Stayed) from demographic, job and workplace features. Five classification models are trained and compared on the same dataset using six evaluation metrics, and deployed in an interactive Streamlit app that evaluates the saved models on an uploaded test CSV without retraining.

## b. Dataset description
- **Source:** https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset
- **Instances:** 59,598 | **Columns:** 24 | **Input features used:** 22
- **Target:** `Attrition` (binary: `Left` = positive class, `Stayed`)
- **Problem type:** Binary classification
- **Preprocessing:**
  - ID column dropped (unique value per row)
  - Numeric features: median imputation + standard scaling
  - Categorical features: most-frequent imputation + one-hot encoding
  - 80/20 stratified train/test split (`random_state=42`): 47,678 training rows, 11,920 test rows
  - The test split is saved as `test_data.csv` and is used in the app
- **Model settings:** Logistic Regression (`max_iter=1000`), Decision Tree (`max_depth=12`), KNN (`k=7`), Gaussian Naive Bayes, SVM (RBF kernel, trained on a 20,000-row random sample of the training set for speed)

## c. GitHub Repository Link
https://github.com/BDheerajChandan/BitsPilani_Assignment

## d. Models used

All five models were evaluated on the same 11,920-row hold-out test set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7499 | 0.8396 | 0.7394 | 0.7320 | 0.7357 | 0.4984 |
| Decision Tree | 0.7187 | 0.7792 | 0.6917 | 0.7369 | 0.7136 | 0.4386 |
| KNN | 0.6803 | 0.7418 | 0.6596 | 0.6771 | 0.6682 | 0.3599 |
| Naive Bayes | 0.7218 | 0.8048 | 0.6862 | 0.7646 | 0.7233 | 0.4479 |
| Support Vector Machine | 0.7483 | 0.8349 | 0.7386 | 0.7285 | 0.7335 | 0.4952 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer: highest accuracy (0.7499), AUC (0.8396), precision, F1 and MCC (0.4984). Its balanced precision and recall make it the most reliable at separating employees who leave from those who stay. It is also fast to train and easy to interpret. |
| Decision Tree | Moderate performance (accuracy 0.7187, AUC 0.7792). Recall (0.7369) is slightly above Logistic Regression's, but precision is lower (0.6917), so it flags more false alarms. A single depth-limited tree cannot capture the feature relationships as well as the linear models. |
| KNN | Weakest model on every metric except recall against SVM (accuracy 0.6803, MCC 0.3599). Distance-based voting suffers after one-hot encoding raises the dimensionality, and it is also the slowest at prediction time on a large dataset. |
| Naive Bayes | Highest recall (0.7646), so it catches the most employees who actually leave, but at the cost of lower precision (0.6862). Its AUC (0.8048) is decent despite the strong feature-independence assumption, which is only partly valid here. It is a good choice if missing a leaver is costlier than a false alarm. |
| Support Vector Machine | Almost matches Logistic Regression (accuracy 0.7483, AUC 0.8349, MCC 0.4952) even though it was trained on only 20,000 samples. The RBF kernel gave no advantage over the linear model, which suggests the classes are largely linearly separable. Training and inference are slower. |
| **Overall winner** | **Logistic Regression.** It leads on accuracy, AUC, precision, F1 and MCC, with SVM a very close second. It is also the simplest, fastest and most interpretable model. Naive Bayes is the best alternative when recall is the priority. |

## Streamlit App Features
- Upload a test CSV (`test_data.csv` from this repo). The CSV must contain all input features and the `Attrition` column.
- Model selection dropdown (5 models)
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix and classification report
- Side-by-side comparison of all 5 models on the uploaded data
- Models are loaded from saved `.joblib` files; no retraining happens in the app

## Live app
https://bitspilani-ml-assignment2.streamlit.app/

## Model Evaluation metrics
- project-folder/model_comparision.csv
- project-folder/model_observation.csv

## Run locally
```
pip install -r requirements.txt
python train.py
streamlit run app.py
```