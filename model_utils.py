"""
Shared code for training the Abalone age model.

This reuses the same pre-processing pipeline and the same five algorithms
as the notebook (abalone_age_prediction_10fold.ipynb).
"""
import json
import pickle
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

DATA_FILE = "abalone.csv"
MODEL_FILE = "abalone_age_model_full_dataset.pkl"
INFO_FILE = "model_info.json"

NUMERIC_FEATURES = [
    "Length", "Diameter", "Height", "Whole weight",
    "Shucked weight", "Viscera weight", "Shell weight",
]
CATEGORICAL_FEATURES = ["Sex"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path=DATA_FILE):
    """Load the CSV, create the Age (years) target, drop duplicates (same as notebook)."""
    df = pd.read_csv(path)
    df["Age (years)"] = df["Rings"] + 1.5
    return df.drop_duplicates().reset_index(drop=True)


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_algorithms():
    return {
        "Linear Regression": LinearRegression(),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }


def fit_final_model(df, algorithm_name):
    """Train the chosen algorithm on the full dataset (notebook Step 10.2)."""
    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", build_algorithms()[algorithm_name]),
    ])
    model.fit(df[ALL_FEATURES], df["Age (years)"])
    return model


def run_cross_validation(df, k=10):
    """10-fold CV for all five algorithms (notebook Step 8). Returns mean/std RMSE per algorithm."""
    X, y = df[ALL_FEATURES], df["Age (years)"]
    folds = list(KFold(n_splits=k, shuffle=True, random_state=42).split(X))
    results = {}
    for name, algorithm in build_algorithms().items():
        model = Pipeline([("preprocessor", build_preprocessor()), ("regressor", algorithm)])
        scores = []
        start = time.time()
        for train_idx, test_idx in folds:
            fold_model = clone(model)
            fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = fold_model.predict(X.iloc[test_idx])
            scores.append(float(np.sqrt(mean_squared_error(y.iloc[test_idx], pred))))
        results[name] = {
            "mean_rmse": float(np.mean(scores)),
            "std_rmse": float(np.std(scores, ddof=1)),
        }
        print(f"{name:<20} mean RMSE = {results[name]['mean_rmse']:.3f} years ({time.time() - start:.1f} s)")
    return results


def build_info(df, best_name, cv_results):
    """Information the app shows and uses for input ranges."""
    return {
        "best_algorithm": best_name,
        "cv_mean_rmse": cv_results[best_name]["mean_rmse"],
        "cv_std_rmse": cv_results[best_name]["std_rmse"],
        "cv_results": cv_results,
        "n_samples": int(len(df)),
        "sklearn_version": sklearn.__version__,
        "feature_ranges": {
            f: [float(df[f].min()), float(df[f].max()), float(df[f].median())]
            for f in NUMERIC_FEATURES
        },
        "sex_values": sorted(df["Sex"].unique().tolist()),
    }


def save_model(model, info):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(INFO_FILE, "w") as f:
        json.dump(info, f, indent=2)
