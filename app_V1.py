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


def get_compatible_models(df, registry):
    compatible = []
    
    for mod in registry:
        if set(mod["features"]).issubset(df.columns):
            compatible.append(mod)

        missing = set(mod["features"]) - set(df.columns)
        
        #print("\nMODEL:", mod["name"])
        #print("missing:", missing)
        
        if len(missing) == 0:
            compatible.append(mod)
            #print("✔ accepted")
    
    return compatible
        
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
            'FlakeTermination', 'Lipping', 'MaxThickness', 'Mass',
            'Elong', 'PlatElong'
        ],
        "accuracy": 0.85
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
        "path": "models/model_v1_simp_numeric.pkl",
        "features": [
            'Lipping', 'MaxThickness', 'Mass', 'Elong', 'PlatElong'
        ],
        "accuracy": 0.72
    }
]


st.title("Lithic Flaking Classifier")


uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

        
    df = pd.read_csv(uploaded_file)
    df = df.replace("#VALUE!", pd.NA)

    debug = st.checkbox("Debug mode")

    if debug:
        st.write("Columns in data:", df.columns.tolist())

    

    st.subheader("Preview")
    st.write(df.head())

    LABEL_MAP = {
        0: "Bipolar",
        1: "Freehand"
    }

    
    compatible_models = get_compatible_models(df, MODEL_REGISTRY)

    if len(compatible_models) == 0:
        st.error("No compatible models found for your dataset.")
        st.stop()

    best_model_info = sorted(
        compatible_models,
        key=lambda x: x["accuracy"],
        reverse=True
    )[0]

    
    if debug:
        st.write("Selected model:", best_model_info["name"])
        st.write("Features used:", best_model_info["features"])

    model_features = best_model_info["features"]
    if "STRAT" in model_features:
        st.error("STRAT should NOT be in model features. Fix registry.")
        st.stop()


    missing = set(best_model_info["features"]) - set(df.columns)

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    model = joblib.load(best_model_info["path"])

    for mod in MODEL_REGISTRY:
        features = mod["features"]
    
        available = [f for f in features if f in df.columns]
    
        if len(available) < 2:
            continue  # too weak
    
    #X = df[available].copy()

    #preds = model.predict(X)
    #probs = model.predict_proba(X)
    preds = model.predict(df)
    probs = model.predict_proba(df)
    df["prediction_numeric"] = preds
    df["prediction_label"] = pd.Series(preds).map(LABEL_MAP)

    df["confidence"] = probs.max(axis=1)

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

