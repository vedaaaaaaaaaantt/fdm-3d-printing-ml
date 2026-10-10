from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "optimized_xgboost_peak_stress.joblib"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fdm_cleaned.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "Printer Type",
    "Material",
    "Layer Thickness (mm)",
    "Infill Density (%)",
    "Infill Pattern",
    "Speed ​​(mm/s)",
    "Width (mm)",
    "Thickness (mm)",
]

TARGET_COLUMN = "Peak Stress (kPa)"


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)


# ============================================================
# FILTER PLA+
# ============================================================

pla_df = df[
    df["Material"].astype(str).str.strip() == "PLA+"
].copy()


if pla_df.empty:
    raise RuntimeError(
        "No PLA+ observations were found in the dataset."
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X = pla_df[FEATURE_COLUMNS]
y = pla_df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


# ============================================================
# GET PIPELINE STEPS
# ============================================================

if not hasattr(model, "named_steps"):
    raise RuntimeError(
        "Saved model is not a scikit-learn Pipeline."
    )


preprocessor = None
estimator = None


for _, step in model.named_steps.items():

    if (
        hasattr(step, "transform")
        and hasattr(step, "get_feature_names_out")
    ):
        preprocessor = step

    if hasattr(step, "get_booster"):
        estimator = step


if preprocessor is None:
    raise RuntimeError(
        "Could not locate preprocessing step."
    )


if estimator is None:
    raise RuntimeError(
        "Could not locate XGBoost estimator."
    )


# ============================================================
# TRANSFORM PLA+ TEST DATA
# ============================================================

X_test_transformed = preprocessor.transform(
    X_test
)


if hasattr(
    X_test_transformed,
    "toarray",
):
    X_test_transformed = (
        X_test_transformed.toarray()
    )


X_test_transformed = np.asarray(
    X_test_transformed,
    dtype=float,
)


# ============================================================
# SHAP ANALYSIS
# ============================================================

explainer = shap.TreeExplainer(
    estimator
)

shap_values = explainer.shap_values(
    X_test_transformed
)

shap_values = np.asarray(
    shap_values
)


if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 0]


# ============================================================
# TRANSFORMED FEATURE NAMES
# ============================================================

transformed_names = (
    preprocessor.get_feature_names_out()
)

transformed_names = [
    str(name)
    for name in transformed_names
]


# ============================================================
# GROUP SHAP VALUES
# ============================================================

original_features = {
    "Printer Type": [],
    "Material": [],
    "Layer Thickness (mm)": [],
    "Infill Density (%)": [],
    "Infill Pattern": [],
    "Speed ​​(mm/s)": [],
    "Width (mm)": [],
    "Thickness (mm)": [],
}


for index, name in enumerate(
    transformed_names
):

    clean_name = name

    if "__" in clean_name:
        clean_name = clean_name.split(
            "__",
            1,
        )[1]


    # Numerical features

    numerical_features = [
        "Layer Thickness (mm)",
        "Infill Density (%)",
        "Speed ​​(mm/s)",
        "Width (mm)",
        "Thickness (mm)",
    ]


    matched = False


    for feature in numerical_features:

        if clean_name == feature:

            original_features[
                feature
            ].append(index)

            matched = True

            break


    if matched:
        continue


    # Categorical features

    categorical_features = [
        "Printer Type",
        "Material",
        "Infill Pattern",
    ]


    for feature in categorical_features:

        prefix = feature + "_"

        if clean_name.startswith(prefix):

            original_features[
                feature
            ].append(index)

            matched = True

            break


# ============================================================
# AGGREGATE SHAP VALUES
# ============================================================

results = []


for feature, indices in (
    original_features.items()
):

    # Material is intentionally excluded
    # from PLA+-specific interpretation because
    # every PLA+ observation has the same material.

    if feature == "Material":
        continue


    if not indices:
        continue


    feature_values = shap_values[
        :,
        indices,
    ]


    # Combine one-hot encoded categories
    combined_values = np.sum(
        feature_values,
        axis=1,
    )


    mean_abs_shap = float(
        np.mean(
            np.abs(
                combined_values
            )
        )
    )


    mean_shap = float(
        np.mean(
            combined_values
        )
    )


    results.append(
        {
            "feature": feature,
            "mean_abs_shap": mean_abs_shap,
            "mean_shap": mean_shap,
        }
    )


# ============================================================
# SORT BY IMPORTANCE
# ============================================================

results.sort(
    key=lambda item:
        item["mean_abs_shap"],
    reverse=True,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("PLA+ PROCESS-PARAMETER SHAP EXPLAINABILITY")
print("=" * 70)

print(
    f"PLA+ observations: {len(pla_df)}"
)

print(
    f"Test samples analyzed: {len(X_test)}"
)

print(
    f"Transformed features: "
    f"{len(transformed_names)}"
)

print()
print(
    "Material excluded from ranking because "
    "all observations are PLA+."
)

print()

print(
    f"{'Feature':<30}"
    f"{'Mean |SHAP|':>18}"
    f"{'Mean SHAP':>18}"
)

print("-" * 70)


for item in results:

    print(
        f"{item['feature']:<30}"
        f"{item['mean_abs_shap']:>18.2f}"
        f"{item['mean_shap']:>18.2f}"
    )


print("=" * 70)