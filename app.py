import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------
# CHARGEMENT DES ARTEFACTS (modèle déjà entraîné, pas de ré-entraînement ici)
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('streamlit_artifacts/lasso_model.pkl')
    preprocessor = joblib.load('streamlit_artifacts/preprocessor.pkl')
    feature_names = joblib.load('streamlit_artifacts/feature_names.pkl')
    seuil = joblib.load('streamlit_artifacts/seuil_cout_sensible.pkl')
    return model, preprocessor, feature_names, seuil

model, preprocessor, feature_names, seuil = load_artifacts()

# ------------------------------------------------------------
# TITRE ET CONTEXTE
# ------------------------------------------------------------
st.title("Détection de fraude santé — Healthcare Provider Fraud Detection")
st.markdown("""
Modèle **LASSO** (régression logistique pénalisée L1) entraîné sur des données Kaggle
de claims santé US, pour prédire la probabilité de fraude d'un prestataire à partir
de 9 variables agrégées (volume de claims, montants remboursés, franchise).
""")

# ------------------------------------------------------------
# FORMULAIRE DE SAISIE
# ------------------------------------------------------------
st.header("Simuler un prestataire")

col1, col2, col3 = st.columns(3)
with col1:
    total_claims = st.number_input("Nombre total de claims", min_value=0, value=100)
    nb_hospit = st.number_input("Nombre d'hospitalisations", min_value=0, value=5)
    nb_outpatient = st.number_input("Nombre de claims ambulatoires", min_value=0, value=95)
with col2:
    total_reimbursed = st.number_input("Total remboursé (€)", min_value=0, value=50000)
    avg_reimbursed_inp = st.number_input("Moyenne remboursée hospit. (€)", min_value=0, value=8000)
    avg_reimbursed_out = st.number_input("Moyenne remboursée ambulatoire (€)", min_value=0, value=300)
with col3:
    max_reimbursed_inp = st.number_input("Max remboursé hospit. (€)", min_value=0, value=20000)
    max_reimbursed_out = st.number_input("Max remboursé ambulatoire (€)", min_value=0, value=1500)
    total_deductible = st.number_input("Franchise totale (€)", min_value=0, value=3000)

SEUIL_YOUDEN = 0.0848  # seuil statistique par défaut (indice de Youden, calibré en OOF)

if st.button("Évaluer ce prestataire"):
    input_df = pd.DataFrame([{
        'total_claims': total_claims,
        'nb_hospit': nb_hospit,
        'nb_outpatient': nb_outpatient,
        'total_reimbursed': total_reimbursed,
        'avg_reimbursed_inp': avg_reimbursed_inp,
        'avg_reimbursed_out': avg_reimbursed_out,
        'max_reimbursed_inp': max_reimbursed_inp,
        'max_reimbursed_out': max_reimbursed_out,
        'total_deductible': total_deductible
    }])

    input_prep = preprocessor.transform(input_df)
    input_prep = pd.DataFrame(input_prep, columns=feature_names)

    proba = model.predict_proba(input_prep)[:, 1][0]

    st.subheader("Résultat")
    st.metric("Probabilité de fraude estimée", f"{proba:.1%}")

    st.markdown("### Comparaison des deux approches de seuil")
    col_youden, col_cout = st.columns(2)

    with col_youden:
        st.markdown("**Seuil statistique (Youden)**")
        st.caption(f"Seuil : {SEUIL_YOUDEN:.3f}")
        if proba >= SEUIL_YOUDEN:
            st.error("Signalé pour audit")
        else:
            st.success("Pas de signalement")

    with col_cout:
        st.markdown("**Seuil coût-sensible (métier)**")
        st.caption(f"Seuil : {seuil:.3f}")
        if proba >= seuil:
            st.error("Signalé pour audit")
        else:
            st.success("Pas de signalement")

    st.caption(
        "Le seuil coût-sensible est calibré pour minimiser un coût métier estimé "
        "(300€ par audit inutile vs. montant remboursé en cas de fraude manquée), "
        "contrairement au seuil Youden qui traite les deux types d'erreur de façon égale."
    )

# ------------------------------------------------------------
# PERFORMANCES DU MODÈLE
# ------------------------------------------------------------
st.header("Performance du modèle (test set)")
perf_col1, perf_col2 = st.columns(2)
perf_col1.metric("ROC-AUC", "0.930")
perf_col2.metric("PR-AUC", "0.705")
st.caption("Évalué sur un test set de 1081 prestataires jamais vus à l'entraînement (9.3% de fraudeurs).")