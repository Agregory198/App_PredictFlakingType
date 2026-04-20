import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

@st.cache_resource
def load_model():
    return joblib.load("flaking_pipeline.pkl")

model = load_model()

st.title("Lithic Flaking Classifier (Batch Mode)")


uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview")
    st.write(df.head())

    LABEL_MAP = {
       0: "Bipolar",
       1: "Freehand"
   }

    REQUIRED_COLUMNS = ['FlakeTermination', 'Lipping', 'TechLength', 'MaxTechWidth',
       'MaxThickness', 'PlatformWidth', 'PlatformThickness', 'Mass']

    if all(col in df.columns for col in REQUIRED_COLUMNS):

        X = df[REQUIRED_COLUMNS]

        preds = model.predict(X)
        probs = model.predict_proba(X)

        df["prediction"] = preds

        df["prediction_label"] = df["prediction"].map(LABEL_MAP)

        df["confidence"] = probs.max(axis=1)

        df["prediction"] = df["prediction"].map(LABEL_MAP)

        st.subheader("Full Results")
        st.write(df)

        # download is always available once computed
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Predictions",
            data=csv,
            file_name="classified_flakes.csv",
            mime="text/csv"
        )

    else:
        st.error("Missing required columns")