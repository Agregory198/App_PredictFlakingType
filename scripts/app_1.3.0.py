import streamlit as st
import pandas as pd
import joblib
import numpy as np
from difflib import get_close_matches


def get_compatible_models(df, registry):
    compatible = []
    
    for mod in registry:
        if set(mod["features"]).issubset(df.columns):
            compatible.append(mod)
    
    return compatible
        
MODEL_REGISTRY_QUARTZ = [
    {
        "name": "full_model",
        "path": "models/model_v1_all_features_quartz.pkl",
        "features": [
            'TechnicalCategory', 'FlakeProfile', 'CortexArea', 'PlatformCortex',
            'CortexLocation', 'DorsalDirection', 'ArisOrientation',
            'CrossSectionType', 'ProfileTwisted?', 'FlakeTermination',
            'PlatformPrep', 'PlatAbrasion', 'FractureInitiationPoint',
            'PlatformDelineation', 'FissuringOnPlatform', 'MarksVentralSurface',
            'Lipping', 'PlatformMorphology', 'EdgeDamage', 'EPACaliper',
            'DorsalScarCount', 'MaxThickness', 'ExteriorPlatAngle', 'Mass',
            'Curvature', 'Elong', 'PlatElong'
        ],
        "accuracy": 0.93
    },
    {
        "name": "categorical_model",
        "path": "models/model_v1_all_categorical_features_quartz.pkl",
        "features": [
            'TechnicalCategory', 'FlakeProfile', 'CortexArea', 'PlatformCortex',
            'CortexLocation', 'DorsalDirection', 'ArisOrientation',
            'CrossSectionType', 'ProfileTwisted?', 'FlakeTermination',
            'PlatformPrep', 'PlatAbrasion', 'FractureInitiationPoint',
            'PlatformDelineation', 'FissuringOnPlatform', 'MarksVentralSurface',
            'Lipping', 'PlatformMorphology', 'EdgeDamage', 'EPACaliper'
        ],
        "accuracy": 0.95
    },
    {
        "name": "numeric_model",
        "path": "models/model_v1_all_numeric_features_quartz.pkl",
        "features": [
            'DorsalScarCount', 'MaxThickness', 'ExteriorPlatAngle', 'Mass',
            'Curvature', 'Elong', 'PlatElong'
        ],
        "accuracy": 0.88
    },
    {
        "name": "simple_model",
        "path": "models/model_v1_simplified_features_quartz.pkl",
        "features": [
            'Lipping', 'MaxThickness', 'Mass', 'Elong', 'PlatElong', 'FlakeTermination'
        ],
        "accuracy": 0.79
    }
]

MODEL_REGISTRY_FLINT = [
    {
        "name": "full_model",
        "path": "models/model_v1_all_features_flint.pkl",
        "features": [
            'TechnicalCategory', 'CortexArea', 'PlatformCortex', 'CortexLocation',
            'DorsalDirection', 'ArisOrientation', 'FlakeProfile', 'ProfileTwisted?',
            'FlakeTermination', 'PlatformPrep', 'PlatAbrasion',
            'FractureInitiationPoint', 'PlatformDelineation', 'FissuringOnPlatform',
            'MarksVentralSurface', 'Lipping', 'PlatformMorphology', 'EdgeDamage',
            'DorsalScarCount', 'Mass', 'ExteriorPlatAngle', 'Curvature', 'Elong',
            'PlatElong'
        ],
        "accuracy": 0.87
    },
    {
        "name": "categorical_model",
        "path": "models/model_v1_all_categorical_features_flint.pkl",
        "features": [
            'TechnicalCategory', 'CortexArea', 'PlatformCortex', 'CortexLocation',
            'DorsalDirection', 'ArisOrientation', 'FlakeProfile', 'ProfileTwisted?',
            'FlakeTermination', 'PlatformPrep', 'PlatAbrasion',
            'FractureInitiationPoint', 'PlatformDelineation', 'FissuringOnPlatform',
            'MarksVentralSurface', 'Lipping', 'PlatformMorphology', 'EdgeDamage'
        ],
        "accuracy": 0.85
    },
    {
        "name": "numeric_model",
        "path": "models/model_v1_all_numeric_features_flint.pkl",
        "features": [
            'DorsalScarCount', 'Mass', 'ExteriorPlatAngle', 'Curvature', 'Elong',
            'PlatElong'
        ],
        "accuracy": 0.75
    },
    {
        "name": "simple_model",
        "path": "models/model_v1_simplified_features_flint.pkl",
        "features": [
            'Lipping', 'Mass', 'Elong', 'PlatElong', 'FlakeTermination'
        ],
        "accuracy": 0.68
    }
]




st.title("Lithic Flaking Classifier")

st.markdown(''' 
            This app will accept specific flake attributes to make predictions on whether 
            the flake was produced via bipolar or freehand technqiues.
            I ran different subsets of flake attributes through logistic regression, decisions trees,
            and RGX ensemble models. The app will choose the model with the best accuracy that best
            match the lithic attributes you import into the application.
            If you have column names that are not identical to those used in the trained models,
            you can choose which columns to rename/reassign once you import your dataset.
            \n
            \n
            This app currently allows for quartz or opaline raw materials. You may choose whichever
            raw material you wish to use for model prediction. If your data do not have a designated 
            raw material column, the default material used is quartz, but you may switch this 
            raw material manually to opaline.
         ''')


uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    material = st.selectbox(
        "Select raw material",
        ["Quartz", "Flint; Chert; Chalcedony"]
    )

        
    df = pd.read_csv(uploaded_file)
    df = df.replace(["#VALUE!", "", " ", "NA", "N/A"], np.nan)
    
    
    
    st.subheader("Raw Material Handling")

    raw_material_col = None

    # Case 1: RawMaterial already exists
    if "RawMaterial" in df.columns:
        raw_material_col = "RawMaterial"

    # Case 2: Let user map another column
    else:

        possible_matches = [
            c for c in df.columns
            if "material" in c.lower()
            or "raw" in c.lower()
            or "lith" in c.lower()
            ]

        selected_col = st.selectbox(
            "Select the column containing raw material information (optional)",
            options=["-- None --"] + list(df.columns),
            index=(list(df.columns).index(possible_matches[0]) + 1)
            if possible_matches else 0
            )

        if selected_col != "-- None --":
            raw_material_col = selected_col
    
    if 'Quartz' in material:

        MODEL_REGISTRY = MODEL_REGISTRY_QUARTZ

        if raw_material_col is not None:

            df = df[
                (df[raw_material_col].astype(str).str.contains('Quartz', na=False)) &
                (df[raw_material_col] != 'Quartzite')
                ]

        else:
            st.warning(
                "No raw material column selected. "
                "All rows will be treated as Quartz."
                )

    elif any(m in material for m in ['Flint', 'Chert', 'Chalcedony']):

        MODEL_REGISTRY = MODEL_REGISTRY_FLINT

        if raw_material_col is not None:

            df = df[
                df[raw_material_col]
                .astype(str)
                .str.contains('Flint|Chert|Chalcedony', na=False)
                ]

        else:
            st.warning(
                "No raw material column selected. "
                "All rows will be treated as Flint/Chert/Chalcedony."
                )

    else:
        st.error(f'There exists a raw material column but this column does not contain {material}')
    

        
    ALL_EXPECTED_COLUMNS = sorted(list(set(
        MODEL_REGISTRY_QUARTZ[0]["features"] +
        MODEL_REGISTRY_FLINT[0]["features"] +
        ["RawMaterial"]
        )))
    
    # -----------------------------
    # COLUMN MAPPER
    # -----------------------------

    st.subheader("Column Mapping")
    with st.expander("Optional: Rename / map columns"):
    
        rename_dict = {}

        for expected_col in ALL_EXPECTED_COLUMNS:

            # Skip if already present
            if expected_col in df.columns:
                continue

            # Find close matches
            matches = get_close_matches(expected_col, df.columns, n=1, cutoff=0.6)

            suggested = matches[0] if matches else None

            selected = st.selectbox(
                f"Map column for: {expected_col}",
                options=["-- Ignore --"] + list(df.columns),
                index=(list(df.columns).index(suggested) + 1) if suggested else 0,
                key=f"map_{expected_col}"
                )

            if selected != "-- Ignore --":
                rename_dict[selected] = expected_col

        # Apply renaming
        if rename_dict:

            if st.button("Apply Column Mapping"):

                df = df.rename(columns=rename_dict)

                st.success("Columns renamed successfully.")

                st.write("Updated columns:")
                st.write(df.columns.tolist())

        if st.checkbox("Show renamed columns"):
            st.write(df.columns.tolist())
    


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
        key=lambda x: (len(x["features"]), x["accuracy"]),
        reverse=True
    )[0]

    
    if debug:
        #st.write("Selected model:", best_model_info["name"])
        #st.write("Features used:", best_model_info["features"])

        for mod in MODEL_REGISTRY:
            missing = set(mod["features"]) - set(df.columns)
            st.write(f"Model: {mod['name']}")
            st.write("Missing:", missing)

    model_features = best_model_info["features"]
    
    if "STRAT" in model_features:
        st.error("STRAT should NOT be in model features. Fix registry.")
        st.stop()


    missing = set(best_model_info["features"]) - set(df.columns)

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    model = joblib.load(best_model_info["path"])



    if debug:
        st.write("Selected model:", best_model_info["name"])
        st.write("Features used:", model_features)

    
    try:
        X = df[model_features].copy()
        X = X.replace(r"^\s*$", np.nan, regex=True)

        preds = model.predict(X)
        probs = model.predict_proba(X)

        df["prediction_numeric"] = preds
        df["prediction_label"] = np.where(df['prediction_numeric'] == 0, 'Freehand', 'Bipolar')

        df["confidence"] = probs.max(axis=1)

        st.subheader("Full Results")
        st.write(df)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Predictions",
            data=csv,
            file_name="classified_flakes.csv",
            mime="text/csv"
        )

    except:
        st.error(f'There was an error, please check that {material} exists in your raw material column')
    
    


