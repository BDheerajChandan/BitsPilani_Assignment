import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from utils import MODEL_FILES, load_meta, prepare, evaluate

st.set_page_config(page_title="Employee Attrition Classifier", page_icon="📉", layout="wide")


@st.cache_resource
def get_model(name):
    return joblib.load(MODEL_FILES[name])


@st.cache_data
def get_meta():
    return load_meta()


meta = get_meta()
neg, pos = [c for c in meta["classes"] if c != meta["positive"]][0], meta["positive"]

st.title("📉 Employee Attrition – Model Evaluation Dashboard")
st.caption("Upload test data, pick a pre-trained model, and inspect its performance. No retraining happens here.")

with st.sidebar:
    st.header("Controls")
    file = st.file_uploader("Upload test CSV", type="csv")
    model_name = st.selectbox("Select model", list(MODEL_FILES))
    st.markdown(f"**Target column:** `{meta['target']}`")
    st.markdown(f"**Positive class:** `{pos}`")
    with st.expander("Required columns"):
        st.write(meta["features"] + [meta["target"]])

if file is None:
    st.info("Upload a CSV (e.g. `test_data.csv` from the repo) to begin.")
    st.stop()

try:
    df = pd.read_csv(file)
    X, y = prepare(df, meta)
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head(10), use_container_width=True)
st.write(f"Rows: **{len(df):,}** | Positive rate: **{y.mean():.2%}**")

metrics, pred = evaluate(get_model(model_name), X, y)

tab1, tab2, tab3 = st.tabs(["Metrics", "Confusion Matrix & Report", "Compare All Models"])

with tab1:
    st.subheader(model_name)
    cols = st.columns(6)
    for c, (k, v) in zip(cols, metrics.items()):
        c.metric(k, f"{v:.4f}")

with tab2:
    left, right = st.columns(2)
    with left:
        cm = confusion_matrix(y, pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=[neg, pos], yticklabels=[neg, pos])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        rep = classification_report(y, pred, target_names=[neg, pos], output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

with tab3:
    if st.button("Evaluate all 5 models"):
        rows = []
        with st.spinner("Scoring..."):
            for n in MODEL_FILES:
                m, _ = evaluate(get_model(n), X, y)
                rows.append({"ML Model Name": n, **m})
        comp = pd.DataFrame(rows).set_index("ML Model Name")
        st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"),
                     use_container_width=True)
        st.bar_chart(comp[["Accuracy", "AUC", "F1", "MCC"]])