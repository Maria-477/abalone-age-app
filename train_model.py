"""
Run this once to (re)create the saved model:

    python train_model.py

It repeats the notebook workflow: 10-fold cross validation on 5 algorithms,
picks the lowest mean RMSE, trains it on the full dataset and saves it.
"""
from model_utils import (
    MODEL_FILE, INFO_FILE, build_info, fit_final_model,
    load_data, run_cross_validation, save_model,
)

df = load_data()
print(f"Loaded {len(df):,} abalones\n")

cv_results = run_cross_validation(df)
best_name = min(cv_results, key=lambda n: cv_results[n]["mean_rmse"])
print(f"\nBest algorithm: {best_name}")

final_model = fit_final_model(df, best_name)
save_model(final_model, build_info(df, best_name, cv_results))
print(f"Saved '{MODEL_FILE}' and '{INFO_FILE}'")
