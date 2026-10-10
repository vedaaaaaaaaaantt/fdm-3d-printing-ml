from pathlib import Path
import os
import itertools

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import PredictionRequest, PredictionResponse


# =========================================================
# PATHS AND DATA
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "fdm_cleaned.csv"
)

MODEL_DIR = PROJECT_DIR / "models"

PEAK_LOAD_MODEL_PATH = MODEL_DIR / "xgboost_peak_load.joblib"
PEAK_STRESS_MODEL_PATH = (
    MODEL_DIR / "optimized_xgboost_peak_stress.joblib"
)
ELASTIC_MODULUS_MODEL_PATH = (
    MODEL_DIR / "xgboost_elastic_modulus.joblib"
)

REQUIRED_MODEL_PATHS = {
    "Peak Load": PEAK_LOAD_MODEL_PATH,
    "Peak Stress": PEAK_STRESS_MODEL_PATH,
    "Elastic Modulus": ELASTIC_MODULUS_MODEL_PATH,
}

for model_name, model_path in REQUIRED_MODEL_PATHS.items():
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_name} model not found at {model_path}. "
            "Run scripts/train_multi_target_models.py first."
        )

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found at {DATA_PATH}"
    )


# =========================================================
# LOAD MODELS
# =========================================================

peak_load_model = joblib.load(PEAK_LOAD_MODEL_PATH)
peak_stress_model = joblib.load(PEAK_STRESS_MODEL_PATH)
elastic_modulus_model = joblib.load(ELASTIC_MODULUS_MODEL_PATH)


# =========================================================
# LOAD AND NORMALIZE DATASET
# =========================================================

df = pd.read_csv(DATA_PATH)

df.columns = [
    column.replace("\u200b", "").strip()
    for column in df.columns
]

for column in [
    "Printer Type",
    "Material",
    "Infill Pattern",
]:
    df[column] = df[column].astype(str).str.strip()


# =========================================================
# MODEL FEATURES
# =========================================================

FEATURE_COLUMNS = [
    "Printer Type",
    "Material",
    "Layer Thickness (mm)",
    "Infill Density (%)",
    "Infill Pattern",
    "Speed (mm/s)",
    "Width (mm)",
    "Thickness (mm)",
]

CATEGORICAL_FEATURES = [
    "Printer Type",
    "Material",
    "Infill Pattern",
]

NUMERICAL_FEATURES = [
    "Layer Thickness (mm)",
    "Infill Density (%)",
    "Speed (mm/s)",
    "Width (mm)",
    "Thickness (mm)",
]

required_columns = FEATURE_COLUMNS + [
    "Peak Load (N)",
    "Peak Stress (kPa)",
    "Modulus (MPa)",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset is missing required columns: {missing_columns}"
    )


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="FDMML - 3D Printing Intelligence API",
    description=(
        "Machine learning API for predicting Peak Load, "
        "Peak Stress and Elastic Modulus of FDM-printed "
        "components, with material-specific explainability "
        "and PLA+ Peak Stress process optimization."
    ),
    version="3.1.0",
)


# =========================================================
# CORS CONFIGURATION
# =========================================================

frontend_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_URL", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        *frontend_origins,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# INPUT PREPARATION
# =========================================================

def build_input_dataframe(request):
    return pd.DataFrame(
        [
            {
                "Printer Type": request.printer_type.strip(),
                "Material": request.material.strip(),
                "Layer Thickness (mm)": request.layer_thickness,
                "Infill Density (%)": request.infill_density,
                "Infill Pattern": request.infill_pattern.strip(),
                "Speed (mm/s)": request.speed,
                "Width (mm)": request.width,
                "Thickness (mm)": request.thickness,
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def predict_all_properties(input_data):
    peak_load = float(
        peak_load_model.predict(input_data)[0]
    )

    peak_stress = float(
        peak_stress_model.predict(input_data)[0]
    )

    elastic_modulus = float(
        elastic_modulus_model.predict(input_data)[0]
    )

    return {
        "predicted_peak_load_n": peak_load,
        "predicted_peak_stress_kpa": peak_stress,
        "predicted_peak_stress_mpa": peak_stress / 1000.0,
        "predicted_elastic_modulus_mpa": elastic_modulus,
        "peak_load_model": "XGBoost",
        "peak_stress_model": "Optimized XGBoost",
        "elastic_modulus_model": "XGBoost",
    }


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "name": "FDMML - 3D Printing Intelligence API",
        "status": "online",
        "version": "3.1.0",
        "predictions": [
            "Peak Load (N)",
            "Peak Stress (kPa)",
            "Elastic Modulus (MPa)",
        ],
        "optimization_objective": "Peak Stress",
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "peak_load": True,
            "peak_stress": True,
            "elastic_modulus": True,
        },
        "dataset_rows": int(len(df)),
    }


# =========================================================
# MULTI-PROPERTY PREDICTION
# =========================================================

@app.post(
    "/api/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):
    try:
        input_data = build_input_dataframe(request)
        predictions = predict_all_properties(input_data)

        return PredictionResponse(**predictions)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )


# =========================================================
# MATERIAL LIST FOR SHAP DROPDOWN
# =========================================================

@app.get("/api/materials")
def available_materials():
    """Return the materials present in the processed dataset."""

    materials = sorted(
        df["Material"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist(),
        key=str.casefold,
    )

    return {"materials": materials}


# =========================================================
# MATERIAL-SPECIFIC SHAP EXPLAINABILITY
# =========================================================

@app.get("/api/pla-explainability")
def pla_explainability(material: str = "PLA+"):
    """
    Explain Peak Stress predictions for the selected material.

    The existing endpoint name is retained to avoid breaking
    frontend integrations. The material query parameter allows
    selection of any material present in the dataset.
    """

    try:
        import shap

        selected_material = material.strip()

        if not selected_material:
            raise HTTPException(
                status_code=400,
                detail="Select a material.",
            )

        available_materials = (
            df["Material"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        matched_material = next(
            (
                value
                for value in available_materials
                if value.casefold() == selected_material.casefold()
            ),
            None,
        )

        if matched_material is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Material '{selected_material}' was not found "
                    "in the dataset."
                ),
            )

        material_df = df[
            df["Material"]
            .astype(str)
            .str.strip()
            .str.casefold()
            == matched_material.casefold()
        ].copy()

        if material_df.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No samples found for material "
                    f"'{matched_material}'."
                ),
            )

        sample_count = min(48, len(material_df))

        X_sample = material_df[FEATURE_COLUMNS].sample(
            n=sample_count,
            random_state=42,
        )

        preprocessor = peak_stress_model.named_steps["preprocessor"]
        estimator = peak_stress_model.named_steps["model"]

        transformed = preprocessor.transform(X_sample)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        transformed_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(estimator)

        shap_values = np.asarray(
            explainer.shap_values(transformed)
        )

        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 0]

        mean_absolute_values = np.abs(shap_values).mean(axis=0)

        grouped_values = {
            feature: []
            for feature in FEATURE_COLUMNS
        }

        # Group encoded columns under their original input parameters.
        for index, transformed_name in enumerate(transformed_names):
            clean_name = str(transformed_name)

            if clean_name.startswith("numerical__"):
                original_name = clean_name.replace(
                    "numerical__", "", 1
                )

                if original_name in grouped_values:
                    grouped_values[original_name].append(
                        float(mean_absolute_values[index])
                    )

            elif clean_name.startswith("categorical__"):
                encoded_name = clean_name.replace(
                    "categorical__", "", 1
                )

                for original_name in CATEGORICAL_FEATURES:
                    if encoded_name.startswith(original_name + "_"):
                        grouped_values[original_name].append(
                            float(mean_absolute_values[index])
                        )
                        break

        # Material is constant within this subset, so it cannot
        # explain differences between samples of that same material.
        grouped_values.pop("Material", None)

        features = [
            {
                "parameter": parameter,
                "mean_absolute_shap": float(sum(values)),
            }
            for parameter, values in grouped_values.items()
            if values
        ]

        features.sort(
            key=lambda item: item["mean_absolute_shap"],
            reverse=True,
        )

        return {
            "status": "success",
            "material": matched_material,
            "samples_analyzed": int(sample_count),
            "total_material_observations": int(len(material_df)),
            "material_excluded": True,
            "transformed_features": int(len(transformed_names)),
            "features": features,
            "interpretation": (
                "Feature importance is calculated from the current "
                f"Peak Stress model using {matched_material} samples. "
                "Larger mean absolute SHAP values indicate greater "
                "overall influence on model predictions."
            ),
            "caution": (
                "SHAP values describe model contributions and do not "
                "establish physical causality. Mean absolute SHAP values "
                "do not indicate whether a parameter increases or "
                "decreases Peak Stress."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SHAP analysis failed: {str(exc)}",
        )


# =========================================================
# PLA+ PROCESS OPTIMIZATION
# The objective remains maximizing Peak Stress.
# =========================================================

@app.post("/api/optimize")
def optimize():
    try:
        pla_df = df[
            df["Material"].str.casefold() == "pla+"
        ].copy()

        if pla_df.empty:
            raise HTTPException(
                status_code=404,
                detail="PLA+ data not found.",
            )

        printer_values = sorted(
            pla_df["Printer Type"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        layer_values = sorted(
            pla_df["Layer Thickness (mm)"]
            .dropna()
            .unique()
            .tolist()
        )

        infill_values = sorted(
            pla_df["Infill Density (%)"]
            .dropna()
            .unique()
            .tolist()
        )

        pattern_values = sorted(
            pla_df["Infill Pattern"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        speed_values = sorted(
            pla_df["Speed (mm/s)"]
            .dropna()
            .unique()
            .tolist()
        )

        representative_width = float(
            pla_df["Width (mm)"].median()
        )

        representative_thickness = float(
            pla_df["Thickness (mm)"].median()
        )

        combinations = list(
            itertools.product(
                printer_values,
                layer_values,
                infill_values,
                pattern_values,
                speed_values,
            )
        )

        optimization_df = pd.DataFrame(
            combinations,
            columns=[
                "Printer Type",
                "Layer Thickness (mm)",
                "Infill Density (%)",
                "Infill Pattern",
                "Speed (mm/s)",
            ],
        )

        optimization_df["Material"] = "PLA+"
        optimization_df["Width (mm)"] = representative_width
        optimization_df["Thickness (mm)"] = representative_thickness

        optimization_df = optimization_df[FEATURE_COLUMNS]

        predictions = peak_stress_model.predict(optimization_df)

        optimization_df["Predicted Peak Stress (kPa)"] = predictions

        optimization_df["Predicted Peak Stress (MPa)"] = (
            predictions / 1000.0
        )

        optimization_df = optimization_df.sort_values(
            by="Predicted Peak Stress (kPa)",
            ascending=False,
        ).reset_index(drop=True)

        best_unrestricted = optimization_df.iloc[0]

        closed_df = optimization_df[
            optimization_df["Printer Type"].str.casefold() == "closed"
        ]

        best_closed = (
            closed_df.iloc[0]
            if not closed_df.empty
            else best_unrestricted
        )

        best_actual = pla_df.sort_values(
            by="Peak Stress (kPa)",
            ascending=False,
        ).iloc[0]

        def build_predicted_configuration(row):
            predicted_kpa = float(
                row["Predicted Peak Stress (kPa)"]
            )

            return {
                "printer_type": row["Printer Type"],
                "material": "PLA+",
                "layer_thickness": float(
                    row["Layer Thickness (mm)"]
                ),
                "infill_density": float(
                    row["Infill Density (%)"]
                ),
                "infill_pattern": row["Infill Pattern"],
                "speed": float(row["Speed (mm/s)"]),
                "width": representative_width,
                "thickness": representative_thickness,
                "predicted_peak_stress_kpa": predicted_kpa,
                "predicted_peak_stress_mpa": predicted_kpa / 1000.0,
            }

        top_5 = [
            build_predicted_configuration(row)
            for _, row in optimization_df.head(5).iterrows()
        ]

        best_experimental_configuration = {
            "printer_type": best_actual["Printer Type"],
            "material": best_actual["Material"],
            "layer_thickness": float(
                best_actual["Layer Thickness (mm)"]
            ),
            "infill_density": float(
                best_actual["Infill Density (%)"]
            ),
            "infill_pattern": best_actual["Infill Pattern"],
            "speed": float(best_actual["Speed (mm/s)"]),
            "width": float(best_actual["Width (mm)"]),
            "thickness": float(best_actual["Thickness (mm)"]),
            "measured_peak_stress_kpa": float(
                best_actual["Peak Stress (kPa)"]
            ),
            "measured_peak_stress_mpa": float(
                best_actual["Peak Stress (kPa)"] / 1000.0
            ),
        }

        return {
            "status": "success",
            "optimization": "PLA+ bounded process optimization",
            "objective": "Maximize predicted Peak Stress",
            "candidates_evaluated": int(len(optimization_df)),
            "representative_width_mm": representative_width,
            "representative_thickness_mm": representative_thickness,
            "best_configuration": build_predicted_configuration(
                best_unrestricted
            ),
            "closed_printer_recommendation": (
                build_predicted_configuration(best_closed)
            ),
            "best_experimental_configuration": (
                best_experimental_configuration
            ),
            "top_5": top_5,
            "validation_note": (
                "The optimization is model-based. The experimentally "
                "observed configuration is reported separately and is "
                "not a model prediction. Physical testing is required "
                "to validate recommended settings."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(exc)}",
        )