# src/train_optuna_compare.py
import pandas as pd
import numpy as np
import optuna
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

# --- Convert SMILES to Morgan fingerprint ---
def smiles_to_morgan(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# --- Compute RDKit descriptors ---
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(Descriptors.descList))
    return np.array([f(mol) for _, f in Descriptors.descList], dtype=float)

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("./data/surfpro_imputed.csv").dropna(subset=["SMILES", "AW_ST_CMC"])

    # Convert SMILES → features
    fp_features = np.array([smiles_to_morgan(s) for s in df["SMILES"]])
    desc_features = np.array([smiles_to_descriptors(s) for s in df["SMILES"]])

    # Combine
    X = np.hstack([fp_features, desc_features])
    y = df["AW_ST_CMC"].values

# --- FIX: clean NaN and Inf ---
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Optional: clip huge values to avoid float32 overflow
    X = np.clip(X, -1e6, 1e6)

 

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    results = {}

    # --- 1. XGBoost ---
    def objective_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42,
            "n_jobs": -1,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)

    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(objective_xgb, n_trials=10)
    best_model_xgb = xgb.XGBRegressor(**study_xgb.best_params)
    best_model_xgb.fit(X_train, y_train)
    y_pred = best_model_xgb.predict(X_test)
    results["XGBoost"] = {
        "R2": r2_score(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Best Params": study_xgb.best_params
    }

    # --- 2. RandomForest ---
    def objective_rf(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "random_state": 42,
            "n_jobs": -1,
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)

    study_rf = optuna.create_study(direction="minimize")
    study_rf.optimize(objective_rf, n_trials=30)
    best_model_rf = RandomForestRegressor(**study_rf.best_params)
    best_model_rf.fit(X_train, y_train)
    y_pred = best_model_rf.predict(X_test)
    results["RandomForest"] = {
        "R2": r2_score(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Best Params": study_rf.best_params
    }

    # --- 3. SVM ---
    def objective_svm(trial):
        params = {
            "C": trial.suggest_float("C", 0.1, 100, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
        }
        model = SVR(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)

    study_svm = optuna.create_study(direction="minimize")
    study_svm.optimize(objective_svm, n_trials=30)
    best_model_svm = SVR(**study_svm.best_params)
    best_model_svm.fit(X_train, y_train)
    y_pred = best_model_svm.predict(X_test)
    results["SVM"] = {
        "R2": r2_score(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Best Params": study_svm.best_params
    }
    
    bad_rows = np.where(~np.isfinite(X).all(axis=1))[0]
    print("Number of problematic molecules:", len(bad_rows))
    print("Example problematic SMILES:", df.iloc[bad_rows]["SMILES"].head())


    # --- Save Results ---
    with open("results.txt", "w") as f:
        f.write("Model Comparison Results\n")
        f.write("="*50 + "\n")
        for model, metrics in results.items():
            f.write(f"{model}:\n")
            f.write(f"  R²   : {metrics['R2']:.4f}\n")
            f.write(f"  RMSE : {metrics['RMSE']:.4f}\n")
            f.write(f"  Best Params: {metrics['Best Params']}\n\n")

        # Find best model (lowest RMSE)
        best_model = min(results.items(), key=lambda x: x[1]["RMSE"])
        f.write("="*50 + "\n")
        f.write(f"Best Model: {best_model[0]}\n")
        f.write(f"  R²   : {best_model[1]['R2']:.4f}\n")
        f.write(f"  RMSE : {best_model[1]['RMSE']:.4f}\n")
        f.write(f"  Params: {best_model[1]['Best Params']}\n")
