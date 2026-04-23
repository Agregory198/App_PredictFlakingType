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

MODEL_REGISTRY = [
    {
        "name": "full_model",
        "path": "models/model_v1_all_features.pkl",
        "features": [
            'FlakeProfile', 'CortexArea', 'PlatformCortex',
           'CortexLocation', 'DorsalDirection', 'ArisOrientation',
           'CrossSectionType', 'ProfileTwisted?', 'FlakeTermination',
           'PlatformPrep', 'PlatAbrasion', 'FractureInitiationPoint',
           'PlatformDelineation', 'FissuringOnPlatform', 'MarksVentralSurface',
           'Lipping', 'PlatformMorphology', 'EdgeDamage', 'EPACaliper',
           'DorsalScarCount', 'MaxThickness', 'ExteriorPlatAngle', 'Mass',
           'Curvature', 'Elong', 'PlatElong', 'Flaking condition'
        ],
        "accuracy": 0.95
    },
    {
        "name": "Steenbokfontein_Data",
        "path": "models/model_v1_sbf_features.pkl",
        "features": [
            'FlakeTermination', 'PlatformPrep', 'PlatAbrasion',
            'Lipping', 'MaxThickness', 'Mass', 'Elong', 'PlatElong'
        ],
        "accuracy": 0.84
    },
    {
        "name": "basic_high_performance",
        "path": "models/model_v1_simp_highPerform.pkl",
        "features": [
            'Lipping', 'MaxThickness', 'Mass', 'Elong', 'PlatElong', 
            'DorsalDirection', 'FlakeTermination', 'PlatformMorphology'
        ],
        "accuracy": 0.95
    },
    {
        "name": "simplest_model",
        "path": "models/model_v1_simp_numeric",
        "features": [
            'Lipping', 'MaxThickness', 'Mass', 'Elong', 'PlatElong'
        ],
        "accuracy": 0.72
    }
]


def load_model():
    return joblib.load("flaking_pipeline.pkl")

#model = load_model()

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


    def get_compatible_models(df, registry):
        compatible = []
    
        for model in registry:
            if set(model["features"]).issubset(df.columns):
                compatible.append(model)
    
        return compatible
    
    compatible_models = get_compatible_models(df, MODEL_REGISTRY)

    if len(compatible_models) == 0:
        st.error("No compatible models found for your dataset.")
        st.stop()

    best_model_info = sorted(
        compatible_models,
        key=lambda x: x["accuracy"],
        reverse=True
    )[0]

    model = joblib.load(best_model_info["path"])
    X = df[best_model_info["features"]]

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

