# 🐚 Abalone Age Prediction System

Predicts an abalone's age in years from its physical measurements (sex, length, diameter, height, whole / shucked / viscera / shell weight).

- **Live Streamlit app:** {{LIVE_APP_URL}}
- **GitHub repository:** {{GITHUB_REPO_URL}}

## How it works

The dataset (`abalone.csv`, 4,177 abalones) stores a ring count. Age is estimated as **Rings + 1.5**.
The notebook compares five regression algorithms with 10-fold cross validation (RMSE) and keeps the best one:

| Algorithm | Mean RMSE (years) |
|---|---|
| **Gradient Boosting (selected)** | **2.161** |
| Random Forest | 2.171 |
| Linear Regression | 2.209 |
| K-Nearest Neighbors | 2.282 |
| Decision Tree | 2.984 |

The selected model is trained on the full dataset. Its pipeline includes median imputation and standard scaling for numeric inputs and one-hot encoding for `Sex`, so the app sends raw measurements straight to the model.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `model_utils.py` | Pre-processing pipeline, algorithms and training helpers (same as the notebook) |
| `train_model.py` | Re-creates the saved model (`python train_model.py`) |
| `abalone_age_model_full_dataset.pkl` / `model_info.json` | Saved model and its summary |
| `abalone_age_prediction_10fold.ipynb` | Original notebook |
| `abalone.csv` | Dataset |
| `requirements.txt` | Python dependencies |

If the saved model cannot be loaded (for example a different scikit-learn version on the server), the app automatically re-trains it from `abalone.csv`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io, sign in with GitHub and click **Create app**.
3. Choose the repository, branch `main`, and main file path `app.py`, then click **Deploy**.
