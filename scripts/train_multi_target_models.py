
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "fdm_cleaned.csv"
)

MODEL_DIR = PROJECT_DIR / "models"
TABLE_DIR = PROJECT_DIR / "results" / "tables"
FIGURE_DIR = PROJECT_DIR / "results" / "figures"

for directory in [MODEL_DIR, TABLE_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------
# 2. FEATURES AND TARGETS
# ---------------------------------------------------------

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

CATEGORICAL_FEATURES = [
    "Printer Type",
    "Material",
    "Infill Pattern",
]

NUMERICAL_FEATURES = [
    "Layer Thickness (mm)",
    "Infill Density (%)",
    "Speed ​​(mm/s)",
    "Width (mm)",
    "Thickness (mm)",
]

TARGETS = {
    "peak_load": "Peak Load (N)",
    "peak_stress": "Peak Stress (kPa)",
    "elastic_modulus": "Modulus (MPa)",
}

# ---------------------------------------------------------
# 3. LOAD AND VALIDATE DATA
# ---------------------------------------------------------

print("=" * 70)
print("FDMML: MULTI-PROPERTY PREDICTION")
print("=" * 70)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

# Normalize column names by removing surrounding spaces.
df.columns = df.columns.str.strip()

# The speed column contains invisible zero-width characters.
# Normalize it safely while preserving the intended column name.
speed_candidates = [
    column
    for column in df.columns
    if "Speed" in column and "(mm/s)" in column
]

if len(speed_candidates) != 1:
    raise ValueError(
        "Could not uniquely identify the printing speed "
        f"column. Found: {speed_candidates}"
    )

df = df.rename(
    columns={
        speed_candidates[0]: "Speed (mm/s)"
    }
)

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

required_columns = (
    FEATURE_COLUMNS
    + list(TARGETS.values())
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Normalize categorical values.
for column in CATEGORICAL_FEATURES:
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

# Convert model features and target columns to numeric.
for column in NUMERICAL_FEATURES + list(TARGETS.values()):
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

# Remove rows missing any of the required features or targets.
# Strain at Break, Hardness, and Roughness are not required
# because they are not inputs or targets in this experiment.
initial_rows = len(df)

df = df.dropna(
    subset=FEATURE_COLUMNS + list(TARGETS.values())
).copy()

print(f"\nOriginal dataset rows: {initial_rows}")
print(f"Rows retained: {len(df)}")
print(f"Rows excluded: {initial_rows - len(df)}")
print(f"Number of input features: {len(FEATURE_COLUMNS)}")

if len(df) < 50:
    raise ValueError(
        "Too few complete rows remain for reliable evaluation."
    )

# ---------------------------------------------------------
# 4. DEFINE PREPROCESSING
# ---------------------------------------------------------

def create_preprocessor():
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def create_pipeline(regressor):
    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                regressor,
            ),
        ]
    )


# ---------------------------------------------------------
# 5. DEFINE COMPARISON MODELS
# ---------------------------------------------------------

def create_baseline_models():
    return {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=4,
            verbosity=0,
        ),

        "SVR": SVR(
            kernel="rbf",
            C=100,
            epsilon=0.1,
        ),
    }


XGB_PARAMETER_SPACE = {
    "model__n_estimators": [
        300, 500, 800
    ],
    "model__max_depth": [
        3, 4, 5, 6, 7
    ],
    "model__learning_rate": [
        0.01, 0.02, 0.03, 0.05
    ],
    "model__subsample": [
        0.7, 0.8, 0.9, 1.0
    ],
    "model__colsample_bytree": [
        0.7, 0.8, 0.9, 1.0
    ],
    "model__min_child_weight": [
        1, 3, 5
    ],
    "model__gamma": [
        0, 0.1, 0.3
    ],
}


# ---------------------------------------------------------
# 6. EVALUATION HELPER
# ---------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    return {
        "R2": float(
            r2_score(y_true, y_pred)
        ),
        "MAE": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        ),
    }


# ---------------------------------------------------------
# 7. TRAIN EACH TARGET INDEPENDENTLY
# ---------------------------------------------------------

all_results = []
summary = {}

for target_key, target_column in TARGETS.items():

    print("\n" + "=" * 70)
    print(f"TARGET: {target_column}")
    print("=" * 70)

    X = df[FEATURE_COLUMNS].copy()
    y = df[target_column].copy()

    # Each target uses the same random split strategy.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    print(f"Training observations: {len(X_train)}")
    print(f"Testing observations: {len(X_test)}")
    print(f"Target unit: {target_column}")

    target_results = []
    fitted_baselines = {}

    # -----------------------------------------------------
    # 7A. BASELINE COMPARISON
    # -----------------------------------------------------

    for model_name, regressor in create_baseline_models().items():

        print(f"\nTraining baseline: {model_name}")

        pipeline = create_pipeline(regressor)

        pipeline.fit(
            X_train,
            y_train,
        )

        predictions = pipeline.predict(X_test)

        metrics = calculate_metrics(
            y_test,
            predictions,
        )

        result = {
            "Target": target_column,
            "Target Key": target_key,
            "Model": model_name,
            **metrics,
        }

        target_results.append(result)
        all_results.append(result)

        fitted_baselines[model_name] = pipeline

        print(
            f"R²={metrics['R2']:.4f} | "
            f"MAE={metrics['MAE']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f}"
        )

    # -----------------------------------------------------
    # 7B. TUNE XGBOOST USING TRAINING DATA ONLY
    # -----------------------------------------------------

    print("\nTuning XGBoost with 5-fold CV...")

    xgb_pipeline = create_pipeline(
        XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=4,
            verbosity=0,
        )
    )

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=XGB_PARAMETER_SPACE,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
        refit=True,
        error_score="raise",
    )

    search.fit(
        X_train,
        y_train,
    )

    optimized_model = search.best_estimator_

    # The held-out test set is used only after tuning.
    test_predictions = optimized_model.predict(
        X_test
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    optimized_result = {
        "Target": target_column,
        "Target Key": target_key,
        "Model": "Optimized XGBoost",
        **test_metrics,
    }

    target_results.append(optimized_result)
    all_results.append(optimized_result)

    print("\nOptimized XGBoost test metrics:")
    print(
        f"R²={test_metrics['R2']:.4f} | "
        f"MAE={test_metrics['MAE']:.4f} | "
        f"RMSE={test_metrics['RMSE']:.4f}"
    )

    print(
        "\nBest CV RMSE:",
        round(-search.best_score_, 4),
    )

    print("\nBest parameters:")
    print(json.dumps(search.best_params_, indent=2))

    # -----------------------------------------------------
    # 7C. FIVE-FOLD CROSS-VALIDATION SUMMARY
    # -----------------------------------------------------

    cv_results = cross_validate(
        optimized_model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        },
        n_jobs=1,
        return_train_score=False,
    )

    cv_summary = {
        "cv_r2_mean": float(
            np.mean(cv_results["test_r2"])
        ),
        "cv_r2_std": float(
            np.std(cv_results["test_r2"])
        ),
        "cv_mae_mean": float(
            -np.mean(cv_results["test_mae"])
        ),
        "cv_rmse_mean": float(
            -np.mean(cv_results["test_rmse"])
        ),
    }

    print("\nFive-fold CV metrics:")
    print(
        f"R²: {cv_summary['cv_r2_mean']:.4f} "
        f"+/- {cv_summary['cv_r2_std']:.4f}"
    )
    print(
        f"MAE: {cv_summary['cv_mae_mean']:.4f}"
    )
    print(
        f"RMSE: {cv_summary['cv_rmse_mean']:.4f}"
    )

    # -----------------------------------------------------
    # 7D. SAVE MODEL AND TEST PREDICTIONS
    # -----------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"optimized_xgboost_{target_key}.joblib"
    )

    joblib.dump(
        optimized_model,
        model_path,
    )

    predictions_df = pd.DataFrame(
        {
            "Actual": y_test.to_numpy(),
            "Predicted": test_predictions,
            "Residual": (
                y_test.to_numpy()
                - test_predictions
            ),
        },
        index=y_test.index,
    )

    predictions_df.index.name = "Original Row Index"

    predictions_path = (
        TABLE_DIR
        / f"{target_key}_test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path
    )

    # Save baseline models too, for later analysis if needed.
    baseline_paths = {}

    for model_name, pipeline in fitted_baselines.items():
        safe_name = (
            model_name.lower()
            .replace(" ", "_")
        )

        baseline_path = (
            MODEL_DIR
            / f"{safe_name}_{target_key}.joblib"
        )

        joblib.dump(
            pipeline,
            baseline_path,
        )

        baseline_paths[model_name] = str(
            baseline_path.relative_to(PROJECT_DIR)
        )

    summary[target_key] = {
        "target_column": target_column,
        "unit": target_column.split("(")[-1].rstrip(")"),
        "test_rows": int(len(X_test)),
        "training_rows": int(len(X_train)),
        "test_metrics": test_metrics,
        "cross_validation": cv_summary,
        "best_cv_rmse": float(-search.best_score_),
        "best_parameters": search.best_params_,
        "model_path": str(
            model_path.relative_to(PROJECT_DIR)
        ),
        "baseline_model_paths": baseline_paths,
        "test_predictions_path": str(
            predictions_path.relative_to(PROJECT_DIR)
        ),
    }

    print(f"\nSaved optimized model: {model_path}")

# ---------------------------------------------------------
# 8. SAVE COMPARISON TABLES AND SUMMARY
# ---------------------------------------------------------

results_df = pd.DataFrame(
    all_results
)

results_df = results_df.sort_values(
    ["Target", "RMSE"],
    ascending=[True, True],
)

comparison_path = (
    TABLE_DIR
    / "multi_property_model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False,
)

summary_path = (
    TABLE_DIR
    / "multi_property_training_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=2,
    )

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    results_df[
        [
            "Target",
            "Model",
            "R2",
            "MAE",
            "RMSE",
        ]
    ].to_string(index=False)
)

print(f"\nComparison table: {comparison_path}")
print(f"Training summary: {summary_path}")
print(f"Model directory: {MODEL_DIR}")
print("\nAll metrics use the target's original unit.")
