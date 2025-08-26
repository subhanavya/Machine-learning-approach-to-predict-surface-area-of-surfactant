# Machine Learning Approaches for Predicting Surfactant Properties

## Project Overview

This project applies machine learning models to predict physicochemical
properties of surfactants (e.g., **surface area, CMC, HLB**) from
molecular structures. The pipeline converts molecular **SMILES strings**
into numerical features using:

-   **Morgan Fingerprints** (captures molecular topology)\
-   **RDKit Descriptors** (quantitative molecular descriptors)

We then evaluate and compare three supervised regression algorithms:

-   **XGBoost (Extreme Gradient Boosting)**\
-   **Random Forest (RF)**\
-   **Support Vector Machines (SVM)**

Hyperparameter tuning is performed using **Optuna**, an automated
optimization framework.

------------------------------------------------------------------------

## Motivation

Surfactants play a critical role in detergents, emulsifiers,
pharmaceuticals, and materials science. Accurately predicting their
properties directly from chemical structure reduces experimental costs
and accelerates discovery.

------------------------------------------------------------------------

## Approach

1.  **Data Preparation**
    -   Input dataset: `surfpro_imputed.csv`\
    -   Required columns: `SMILES`, `AW_ST_CMC` (or target property)\
    -   Missing or invalid SMILES are skipped.
2.  **Feature Engineering**
    -   Generate **Morgan fingerprints** (binary molecular encoding)\
    -   Compute **RDKit descriptors**\
    -   Concatenate both to create a comprehensive feature set
3.  **Model Training and Validation**
    -   Split dataset into train/validation/test\
    -   Normalize features with `StandardScaler`\
    -   Train three ML models: XGBoost, Random Forest, and SVM
4.  **Hyperparameter Tuning with Optuna**
    -   Optuna runs multiple trials to optimize key parameters:
        -   **XGBoost**: n_estimators, max_depth, learning_rate,
            subsample, colsample_bytree, gamma, min_child_weight\
        -   **Random Forest**: n_estimators, max_depth,
            min_samples_split, min_samples_leaf\
        -   **SVM**: kernel, C, epsilon, gamma\
    -   Objective function minimizes **Mean Squared Error (MSE)**
5.  **Evaluation Metrics**
    -   **R² (Coefficient of Determination)**\
    -   **RMSE (Root Mean Squared Error)**
6.  **Results Recording**
    -   A results table (`results.txt`) stores performance of all
        models\
    -   The **best-performing model** is identified and saved

------------------------------------------------------------------------

## Why XGBoost?

-   Handles **nonlinearities** and **high-dimensional feature spaces**
    effectively\
-   Built-in **regularization** reduces overfitting compared to Random
    Forest\
-   Efficient for **sparse molecular features** (like fingerprints)\
-   Often outperforms classical methods in **chemoinformatics tasks**

------------------------------------------------------------------------

## Usage

### Install dependencies

``` bash
pip install -r requirements.txt
```

### Run training & comparison

``` bash
python train_optuna_compare.py
```

### Expected output

-   `results.txt`: Contains R² and RMSE for all models + best model
    selected

Example snippet from results file:

    Model        R²      RMSE
    XGBoost     0.78    0.45
    RandomForest 0.72   0.52
    SVM         0.65    0.61

    Best model: XGBoost

------------------------------------------------------------------------

## Project Description (Portfolio-Ready)

**Machine Learning Approaches for Surfactant Property Prediction (Feb
2025 -- Present)**\
- Developed a **chemoinformatics pipeline** to predict surfactant
properties from SMILES using molecular fingerprints and descriptors\
- Implemented **XGBoost, Random Forest, and SVM models** with
Optuna-driven hyperparameter tuning\
- Achieved **R² up to 0.78** with XGBoost, demonstrating its superior
ability to capture structure--property relationships\
- Delivered a reproducible workflow with automated results comparison
and model selection
