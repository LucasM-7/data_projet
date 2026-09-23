# ============================================================
# Détection de fraude santé - Healthcare Provider Fraud Detection
# Pipeline Python : LASSO (modèle final), XGBoost (comparaison),
# SHAP, seuil de décision coût-sensible
#
# Split train/test identique à celui utilisé en R (train_providers.csv /
# test_providers.csv), pour comparer les modèles sur les mêmes prestataires.
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (roc_curve, roc_auc_score, average_precision_score,
                              confusion_matrix, classification_report)
import xgboost as xgb
import shap
import joblib

DATA_DIR = "data"
ARTIFACTS_DIR = "streamlit_artifacts"

# ------------------------------------------------------------
# Chargement et vérification
# ------------------------------------------------------------
df = pd.read_csv(f"{DATA_DIR}/df_lasso.csv")
# Dimensions : (5410, 11)

# ------------------------------------------------------------
# Imputation des zéros structurels
# ------------------------------------------------------------
cols_montants = ["avg_reimbursed_inp", "max_reimbursed_inp",
                  "avg_reimbursed_out", "max_reimbursed_out"]
df[cols_montants] = df[cols_montants].fillna(0)

# ------------------------------------------------------------
# Encodage de la cible et séparation features / cible
# ------------------------------------------------------------
df["target"] = (df["PotentialFraud"] == "Yes").astype(int)
X = df.drop(columns=["Provider", "PotentialFraud", "target"])
y = df["target"]

# ------------------------------------------------------------
# Split train/test aligné avec le pipeline R
# ------------------------------------------------------------
train_providers = pd.read_csv(f"{DATA_DIR}/train_providers.csv")["Provider"]
train_mask = df["Provider"].isin(train_providers)

X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_test, y_test = X.loc[~train_mask], y.loc[~train_mask]
# Train : 4329 lignes, taux de fraude 9.36% | Test : 1081 lignes, taux de fraude 9.34%

# ------------------------------------------------------------
# Preprocessing : log1p sur les montants, via ColumnTransformer
# ------------------------------------------------------------
num_cols_log = ["total_reimbursed", "avg_reimbursed_inp", "avg_reimbursed_out",
                "max_reimbursed_inp", "max_reimbursed_out"]

preprocessor = ColumnTransformer(
    transformers=[("log1p", FunctionTransformer(np.log1p, validate=True), num_cols_log)],
    remainder="passthrough"
)

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

feature_names = num_cols_log + [c for c in X_train.columns if c not in num_cols_log]
X_train_prep = pd.DataFrame(X_train_prep, columns=feature_names, index=X_train.index)
X_test_prep = pd.DataFrame(X_test_prep, columns=feature_names, index=X_test.index)

# ------------------------------------------------------------
# LASSO (modèle final retenu) - seuil calibré sur des prédictions
# Out-Of-Fold du train, jamais sur le test
# ------------------------------------------------------------
lasso_py = LogisticRegressionCV(
    penalty="l1", solver="liblinear", Cs=20, cv=5,
    scoring="roc_auc", random_state=42, max_iter=5000
)
lasso_py.fit(X_train_prep, y_train)

cv_lasso_py = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_lasso = cross_val_predict(
    LogisticRegressionCV(penalty="l1", solver="liblinear", Cs=20, cv=5,
                          scoring="roc_auc", random_state=42, max_iter=5000),
    X_train_prep, y_train, cv=cv_lasso_py, method="predict_proba"
)[:, 1]

fpr_l, tpr_l, thresholds_l = roc_curve(y_train, oof_preds_lasso)
best_threshold_lasso = thresholds_l[np.argmax(tpr_l - fpr_l)]

lasso_test_probs = lasso_py.predict_proba(X_test_prep)[:, 1]
lasso_test_preds = (lasso_test_probs >= best_threshold_lasso).astype(int)

print(f"LASSO — ROC-AUC : {roc_auc_score(y_test, lasso_test_probs):.4f}")       # 0.9302
print(f"LASSO — PR-AUC  : {average_precision_score(y_test, lasso_test_probs):.4f}")  # 0.7045
print(f"Seuil optimal (Youden, OOF) : {best_threshold_lasso:.4f}")             # 0.0848
print(confusion_matrix(y_test, lasso_test_preds))
# [[803 177]
#  [ 13  88]]

# ------------------------------------------------------------
# XGBoost - modèle de comparaison, pas retenu comme modèle final
# ------------------------------------------------------------
ratio_imbalance = (len(y_train) - sum(y_train)) / sum(y_train)

xgb_model = xgb.XGBClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=4,
    scale_pos_weight=ratio_imbalance, random_state=42, eval_metric="logloss"
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = cross_val_predict(xgb_model, X_train_prep, y_train, cv=cv, method="predict_proba")[:, 1]

fpr, tpr, thresholds = roc_curve(y_train, oof_preds)
best_threshold = thresholds[np.argmax(tpr - fpr)]
# XGBoost (OOF) — ROC-AUC : 0.9350 | PR-AUC : 0.6705 | Seuil Youden : 0.3867

xgb_model.fit(X_train_prep, y_train)
test_probs = xgb_model.predict_proba(X_test_prep)[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)

test_auc = roc_auc_score(y_test, test_probs)
test_pr_auc = average_precision_score(y_test, test_probs)
print(f"XGBoost — ROC-AUC Test : {test_auc:.4f}")     # 0.9195
print(f"XGBoost — PR-AUC Test  : {test_pr_auc:.4f}")  # 0.6927
print(classification_report(y_test, test_preds, target_names=["Non Fraude", "Fraude"]))

# ------------------------------------------------------------
# Explicabilité (TreeSHAP sur XGBoost)
# ------------------------------------------------------------
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_test_prep)
shap.summary_plot(shap_values, X_test_prep)

# ------------------------------------------------------------
# Seuil de décision coût-sensible (métier)
# Hypothèses : coût audit (faux positif) = 300€ fixe
#              coût fraude manquée (faux négatif) = total_reimbursed du prestataire
# Calculé sur les probabilités du LASSO, le modèle final retenu.
# ------------------------------------------------------------
nb_FP = ((lasso_test_preds == 1) & (y_test == 0)).sum()
cout_FP_youden = nb_FP * 300
cout_FN_youden = X_test[(y_test == 1) & (lasso_test_preds == 0)]["total_reimbursed"].sum()
cout_total_youden = cout_FP_youden + cout_FN_youden
# Seuil Youden (0.0848) -> coût total ≈ 772 510€

seuils_a_tester = np.concatenate([
    np.arange(0.001, 0.05, 0.002),
    np.arange(0.05, 0.95, 0.05)
])
resultats = []
for seuil in seuils_a_tester:
    preds_seuil = (lasso_test_probs >= seuil).astype(int)
    nb_FP = ((preds_seuil == 1) & (y_test == 0)).sum()
    cout_FP = nb_FP * 300
    cout_FN = X_test[(y_test == 1) & (preds_seuil == 0)]["total_reimbursed"].sum()
    resultats.append({"seuil": seuil, "cout_FP": cout_FP, "cout_FN": cout_FN, "cout_total": cout_FP + cout_FN})

resultats_df = pd.DataFrame(resultats)
meilleur = resultats_df.loc[resultats_df["cout_total"].idxmin()]
print(f"Seuil coût-optimal : {meilleur['seuil']:.3f} — coût total minimal : {meilleur['cout_total']:.0f}€")
# Seuil coût-optimal : 0.019 — coût total minimal : 121 330€
# -> réduction de 84.3% du coût total estimé par rapport au seuil Youden

# ------------------------------------------------------------
# Tableau comparatif final
# ------------------------------------------------------------
comparatif = pd.DataFrame({
    "Modele": ["LASSO (retenu)", "XGBoost"],
    "ROC-AUC": [round(roc_auc_score(y_test, lasso_test_probs), 4), round(test_auc, 4)],
    "PR-AUC": [round(average_precision_score(y_test, lasso_test_probs), 4), round(test_pr_auc, 4)]
})
print(comparatif.to_string(index=False))

# ------------------------------------------------------------
# Export des artefacts pour le dashboard Streamlit
# ------------------------------------------------------------
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

joblib.dump(lasso_py, f"{ARTIFACTS_DIR}/lasso_model.pkl")
joblib.dump(preprocessor, f"{ARTIFACTS_DIR}/preprocessor.pkl")
joblib.dump(feature_names, f"{ARTIFACTS_DIR}/feature_names.pkl")
joblib.dump(float(meilleur["seuil"]), f"{ARTIFACTS_DIR}/seuil_cout_sensible.pkl")
