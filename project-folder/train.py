# train.py

import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import MODEL_FILES, prepare, evaluate

DATA = "train.csv"
SEED = 42
SVM_MAX_ROWS = 20000

df = pd.read_csv(DATA)
print("Shape:", df.shape)

# target detection
target = next((c for c in df.columns if "attrition" in c.lower()), df.columns[-1])
df = df.dropna(subset=[target])

# drop ID-like and constant columns
drop_cols = [c for c in df.columns if c != target and (df[c].nunique() == len(df) or df[c].nunique() <= 1)]
features = [c for c in df.columns if c not in drop_cols + [target]]

# positive class
vals = sorted(df[target].astype(str).unique())
assert len(vals) == 2, f"Binary target expected, got {vals}"
hints = ["left", "yes", "1", "true", "attrited", "churn"]
positive = next((v for v in vals if v.lower() in hints), vals[-1])

meta = {"target": target, "features": features, "dropped": drop_cols,
        "classes": vals, "positive": positive}
print("Target:", target, "| Positive:", positive, "| Features:", len(features))
assert len(features) >= 12 and len(df) >= 500, "Dataset below assignment minimums"

# split
train_df, test_df = train_test_split(
    df[features + [target]], test_size=0.2, stratify=df[target], random_state=SEED)

X_train, y_train = prepare(train_df, meta)
X_test, y_test = prepare(test_df, meta)

num = X_train.select_dtypes(include="number").columns.tolist()
cat = [c for c in features if c not in num]


def pre():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])


estimators = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", random_state=SEED),
}

Path("model").mkdir(exist_ok=True)
rows = []
for name, est in estimators.items():
    model = Pipeline([("pre", pre()), ("clf", est)])
    if name == "Support Vector Machine" and len(X_train) > SVM_MAX_ROWS:
        Xs = X_train.sample(SVM_MAX_ROWS, random_state=SEED)
        model.fit(Xs, y_train.loc[Xs.index])
    else:
        model.fit(X_train, y_train)
    m, _ = evaluate(model, X_test, y_test)
    rows.append({"ML Model Name": name, **m})
    joblib.dump(model, MODEL_FILES[name], compress=3)
    print("done:", name)

res = pd.DataFrame(rows).round(4)
res.to_csv("model/metrics.csv", index=False)
with open("model/meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(res.to_string(index=False))